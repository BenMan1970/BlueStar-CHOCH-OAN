"""CHoCH Scanner — couche UI Streamlit (version : voir SCANNER_VERSION).

Architecture production grade : le MOTEUR (constantes, logging, domaine,
I/O OANDA, orchestration, contrat JSON, exports) vit dans
choch_scanner_v5_25.py — importable sans aucune dependance UI, auditable et
testable hors Streamlit. Ce fichier ne contient que la presentation et le
branchement des caches Streamlit sur le moteur.

Les fonctions cachees (bougies, precisions) sont reaffectees DANS l'espace
de noms du moteur : run_scan() les resout via les globaux de ce module.

Lancer : streamlit run app.py
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import streamlit as st

import choch_scanner_v5_25 as choch
from choch_scanner_v5_25 import (ConfigError, DISPLAY_COLS, INSTRUMENTS,
                                MIN_SCORE, RULE_VERSION, SCAN_GLOBAL_TIMEOUT,
                                SCAN_MAX_WORKERS, SCANNER_VERSION,
                                TIMEFRAMES, _log, create_pdf, generate_png,
                                resolve_credentials, run_scan,
                                serialize_pipeline)


def _streamlit_secret(key: str) -> Optional[str]:
    """Lecture tolerante de st.secrets (absence de secrets.toml != crash)."""
    try:
        value = st.secrets.get(key)
    except Exception:  # noqa: BLE001 — fichier de secrets absent/illisible
        return None
    return None if value is None else str(value)


# --- Branchement des caches Streamlit sur le moteur -------------------------
# Le moteur definit des fonctions pures (testables hors Streamlit) ; on leur
# ajoute ici le cache Streamlit. La reaffectation dans l'espace de noms du
# moteur est OBLIGATOIRE : run_scan() appelle ces fonctions via les globaux
# du module moteur.
choch._SECRET = _streamlit_secret
choch.get_candles_cached = st.cache_data(
    ttl=choch.CANDLES_CACHE_TTL_SECONDS, show_spinner=False,
    max_entries=512)(choch.get_candles_cached)
choch._fetch_precisions = st.cache_data(
    ttl=choch.PRECISION_CACHE_TTL_SECONDS, show_spinner=False,
    max_entries=8)(choch._fetch_precisions)


# =====================================================================
# SECTION 8 — COUCHE UI
# =====================================================================

@dataclass
class StoredScan:
    """Tout ce que l'UI affiche vient d'ici ; construit UNE fois par scan."""
    scan_time: datetime
    df: pd.DataFrame
    json_bytes: Optional[bytes]
    doc: Optional[dict[str, Any]]
    json_error: Optional[str]
    errors: list[str]
    timed_out: int
    auth_aborted: bool
    exports: dict[str, bytes] = field(default_factory=dict)


def _build_stored_scan(result: ScanResult) -> StoredScan:
    if result.rows:
        df = (pd.DataFrame(result.rows)
              .sort_values(["_status_rank", "_time_sort", "Instrument",
                            "Timeframe"],
                           ascending=[True, False, True, True],
                           kind="mergesort")
              .loc[:, list(DISPLAY_COLS)]
              .reset_index(drop=True))
    else:
        df = pd.DataFrame(columns=list(DISPLAY_COLS))
    json_bytes: Optional[bytes] = None
    doc: Optional[dict[str, Any]] = None
    json_error: Optional[str] = None
    try:
        json_bytes = serialize_pipeline(result.payloads, result.scan_time,
                                        result.errors,
                                        result.coverage_counts,
                                        result.precision, result.env)
        doc = json.loads(json_bytes)
    except (RuntimeError, TypeError, ValueError) as exc:
        json_error = str(exc)
        _log(logging.ERROR, "pipeline_json_failed", err=json_error)
    # E1 : invariant UI == JSON. Les EXPORTS (CSV/PDF/PNG) doivent
    # contenir EXACTEMENT les signaux valides du pipeline JSON. Avant,
    # df_export filtrait seulement sur Statut : un signal rejete par le
    # contrat (Invalidated) y restait, brisant l'invariant.
    # AUD-03 : on filtre en LISTE BLANCHE (valid_ids + Stale), pas en
    # liste noire sur invalid_signals, car cette derniere est tronquee
    # a 50 en serialize_pipeline -> au-dela, des rejets restaient affiches.
    # Les Stale restent affiches (visibles, non emis) : comportement
    # documente l.1640-1643 preserve.
    if doc is not None and not df.empty:
        valid_ids = {s["signal_id"] for s in doc["signals"]}
        n_before = len(df)
        df = df[
            df["signal_id"].isin(valid_ids) | (df["Statut"] == "Stale")
            ].reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped:
            _log(logging.WARNING, "ui_json_desync",
                 dropped=n_dropped,
                 raison="signaux rejetes par le contrat retires du tableau")
    return StoredScan(
        scan_time=result.scan_time, df=df, json_bytes=json_bytes, doc=doc,
        json_error=json_error, errors=list(result.errors),
        timed_out=result.coverage_counts["timed_out"],
        auth_aborted=result.auth_aborted,
    )


def _style_bb(val: object) -> str:
    s = str(val)
    if "Squeeze" in s:
        return "color:#ff9800;font-weight:bold"
    if "Expansion" in s:
        return "color:#ab47bc;font-weight:bold"
    return "color:#90a4ae"


def _style_distance(val: object) -> str:
    try:
        v = float(str(val).replace("%", ""))
    except ValueError:
        return "color:#90a4ae"
    if v <= 0.15:
        return "color:#00c853;font-weight:bold"
    if v <= 0.40:
        return "color:#ff9800;font-weight:bold"
    return "color:#ff5252;font-weight:bold"


_STATUS_STYLE: Final[Mapping[str, str]] = {
    "Fresh": "color:#00c853;font-weight:bold",
    "Aged": "color:#ff9800;font-weight:bold",
    "Stale": "color:#ff5252;font-weight:bold",
    "Invalidated": "color:#9e9e9e;text-decoration:line-through",
}


def _render_dataframe(df_all: pd.DataFrame) -> None:
    # F4 : signal_id reste dans df_all (filtrage UI == JSON, exports) mais
    # n'est pas affiche dans le tableau Streamlit.
    df_view = df_all.drop(columns=["signal_id"], errors="ignore")
    styled = (
        df_view.style
        .map(lambda x: "color:#e879f9;font-weight:bold" if x == "CHoCH"
             else "color:#94a3b8", subset=["Type"])
        .map(lambda x: "color:#00c853;font-weight:bold" if x == "Achat"
             else "color:#ff5252;font-weight:bold", subset=["Ordre"])
        .map(lambda x: "color:#00c853" if str(x).startswith("Bull")
             else "color:#ff5252", subset=["Signal"])
        .map(lambda x: "color:#00c853;font-weight:bold" if x == "Fort"
             else "color:#ff9800", subset=["Force"])
        .map(_style_bb, subset=["BB_Width"])
        .map(_style_distance, subset=["Distance%"])
        .map(lambda x: _STATUS_STYLE.get(str(x), ""), subset=["Statut"])
    )
    st.dataframe(styled, hide_index=True, width="stretch")


def _render_downloads(scan: StoredScan, df_export: pd.DataFrame) -> None:
    ts = scan.scan_time.strftime("%Y%m%d_%H%M%S")
    if "csv" not in scan.exports:
        # to_csv() sans chemin renvoie une chaine : on encode en utf-8-sig
        # pour que la BOM soit presente (Excel, accents).
        scan.exports["csv"] = df_export.to_csv(index=False).encode(
            "utf-8-sig")
    if "png" not in scan.exports:
        scan.exports["png"] = generate_png(df_export)
    if "pdf" not in scan.exports:
        scan.exports["pdf"] = create_pdf(df_export, scan.scan_time)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("CSV", scan.exports["csv"], f"choch_{ts}.csv",
                           "text/csv", key=f"dl_csv_{ts}")
    with c2:
        st.download_button("PNG", scan.exports["png"], f"choch_{ts}.png",
                           "image/png", key=f"dl_png_{ts}")
    with c3:
        st.download_button("PDF", scan.exports["pdf"],
                           f"choch_signaux_{ts}.pdf", "application/pdf",
                           key=f"dl_pdf_{ts}")
    with c4:
        if scan.json_bytes is not None:
            st.download_button("JSON", scan.json_bytes,
                               f"choch_pipeline_{ts}.json",
                               "application/json", key=f"dl_json_{ts}")
        else:
            st.error(f"JSON non généré (invariant violé) : {scan.json_error}")


def _render_results(scan: StoredScan) -> None:
    if scan.auth_aborted:
        st.error("Scan interrompu — trop d'erreurs d'authentification "
                 "OANDA. Vérifiez le token.")
    if scan.timed_out:
        st.warning(f"Timeout global — {scan.timed_out} requête(s) "
                   "non terminée(s), résultats partiels.")
    if scan.errors:
        st.warning(f"{len(scan.errors)} erreur(s) : "
                   f"{'; '.join(scan.errors[:5])}")
    df_all = scan.df
    # E1 : les exports ne contiennent que les signaux VALIDES du pipeline
    # JSON. AUD-03/m3 : si le JSON a echoue (doc is None), on n'exporte
    # RIEN (echec ferme) : exporter un CSV non valide brisait l'invariant
    # UI == JSON et livrait des donnees non contractuelles.
    if scan.doc is not None:
        valid_ids = {s["signal_id"] for s in scan.doc["signals"]}
        df_export = df_all[df_all["signal_id"].isin(valid_ids)]
    else:
        df_export = df_all.iloc[0:0]   # vide : pas d'export non valide
    _render_downloads(scan, df_export)

    if scan.doc is not None:
        meta = scan.doc["meta"]
        cov = meta["coverage"]
        sc = meta["status_counts"]
        n_invalid = len(meta["invalid_signals"])
        st.caption(
            f"Pipeline JSON : {meta['signal_count']} signal(s) "
            f"(Fresh {sc['Fresh']}, Aged {sc['Aged']}, "
            f"Invalidated {sc['Invalidated']})"
            + (f", {n_invalid} rejeté(s) par le contrat" if n_invalid else "")
            + f" | schema {meta['schema_version']} | rule "
            f"{meta['rule_version']} | couverture {cov['pairs_requested']} = "
            f"{cov['ok_signal']} signal + {cov['ok_no_signal']} sans signal"
            f" + {cov['ok_not_emitted']} Stale + "
            f"{cov['invalid_contract']} rejetés + {cov['no_data']} no_data"
            f" + {cov['failed']} failed + {cov['aborted']} aborted + "
            f"{cov['timed_out']} timed_out"
        )

    if df_all.empty:
        st.info(f"Aucun signal CHoCH/BOS qualifié (Score ≥ {MIN_SCORE}).")
        return
    n_stale = int((df_all["Statut"] == "Stale").sum())
    if n_stale:
        st.info(f"{n_stale} signal(s) Stale visible(s) dans le tableau — "
                "exclus des exports et du pipeline JSON.")
    _render_dataframe(df_all)
    if scan.doc is not None and scan.doc["signals"]:
        with st.expander("Aperçu JSON Pipeline (premier signal)"):
            st.json(scan.doc["signals"][0])


def _init_session_state() -> None:
    for k, v in {"cache_bust": 0, "scan": None}.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _trigger_scan() -> None:
    try:
        creds = resolve_credentials()
    except ConfigError as exc:
        st.error(str(exc))
        return
    pb = st.progress(0.0, text="Initialisation du scan…")
    info = st.empty()
    t0 = time.monotonic()
    total = len(INSTRUMENTS) * len(TIMEFRAMES)
    done = 0

    def _tick(inst: str, tf: str) -> None:
        nonlocal done
        done += 1
        elapsed = time.monotonic() - t0
        eta = elapsed / done * (total - done)
        pb.progress(done / total,
                    text=f"[{done}/{total}] {inst.replace('_', '/')} "
                         f"({tf}) — ETA {int(eta)}s")
        info.caption(f"Workers: {SCAN_MAX_WORKERS} | Timeout: "
                     f"{SCAN_GLOBAL_TIMEOUT}s | Écoulé: {elapsed:.1f}s")

    try:
        result = run_scan(creds, st.session_state.cache_bust,
                          progress_callback=_tick)
    except Exception as exc:  # noqa: BLE001 — barriere finale
        _log(logging.ERROR, "scan_fatal", err=repr(exc))
        st.error(f"Erreur critique du scan : {exc}")
        return
    finally:
        pb.empty()
        info.empty()
    st.session_state.scan = _build_stored_scan(result)


# =====================================================================
# SECTION 9 — POINT D'ENTREE STREAMLIT
# =====================================================================

def main() -> None:
    """Point d'entree Streamlit (5.25 H2)."""
    st.set_page_config(page_title=f"CHoCH Scanner v{SCANNER_VERSION}",
                       layout="wide")
    st.title(f"Scanner Change of Character (CHoCH) — v{SCANNER_VERSION} "
             f"({RULE_VERSION})")
    _init_session_state()

    col_a, col_b = st.columns([3, 1])
    with col_a:
        scan_clicked = st.button("Lancer le Scan", type="primary",
                                 width="stretch", key="btn_scan")
    with col_b:
        force_refresh = st.button("Force refresh (bust cache)", width="stretch",
                                  key="btn_force_refresh")

    if force_refresh:
        # Nouvelle cle de cache pour CETTE session ; n'efface pas le cache des
        # autres sessions (pas de get_candles_cached.clear() global).
        st.session_state.cache_bust += 1
        st.toast("Le prochain scan rechargera les bougies depuis OANDA.")

    if scan_clicked:
        _trigger_scan()

    if st.session_state.scan is not None:
        _render_results(st.session_state.scan)


if __name__ == "__main__":
    main()
