"""
app.py – Point d'entrée Streamlit du scanner CHoCH v5.13.
"""
import logging
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from oandapyV20.exceptions import V20Error
from requests.exceptions import RequestException

from config import INSTRUMENTS, SCANNER_VERSION, TIMEFRAMES, SCAN_GLOBAL_TIMEOUT
from detection import detect_choch_v58, build_pipeline_payload_v58
from indicators import (
    _auth_counter,
    calc_distance_pct,
    compute_bb_width,
    compute_statut,
    format_bb_width,
    format_distance,
    format_niveau,
    get_candles,
)
from ui import render_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title=f"CHoCH Scanner v{SCANNER_VERSION}", layout="wide")
st.title(f"Scanner Change of Character (CHoCH) — v{SCANNER_VERSION} Intraday")

if "scanning" not in st.session_state:
    st.session_state.scanning = False

if st.button(
    "Lancer le Scan",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.scanning,
):
    st.session_state.scanning = True
    NUM_COMBOS = len(INSTRUMENTS) * len(TIMEFRAMES)
    st.session_state.scan_time = datetime.now(timezone.utc)
    _scan_time = st.session_state.scan_time
    _auth_counter.reset()

    try:
        for key in ("df", "pipeline_signals", "png_buf", "pdf_buf"):
            st.session_state.pop(key, None)

        with st.spinner(f"Scan en cours sur {NUM_COMBOS} combinaisons…"):
            _results: list[dict] = []
            _pipeline_signals: list[dict] = []
            _errors: list[str] = []
            scan_aborted = False

            executor = ThreadPoolExecutor(max_workers=6)
            try:
                futures = {
                    executor.submit(get_candles, inst, tf_code): (inst, tf_name)
                    for inst in INSTRUMENTS
                    for tf_name, tf_code in TIMEFRAMES.items()
                }

                done, not_done = wait(futures.keys(), timeout=SCAN_GLOBAL_TIMEOUT)
                if not_done:
                    st.warning(
                        f"Timeout global – {len(not_done)} requête(s) ignorée(s), "
                        "résultats partiels."
                    )
                    for f in not_done:
                        f.cancel()

                for future in done:
                    _inst, _tf_name = futures[future]
                    try:
                        _df = future.result()
                    except CancelledError:
                        continue
                    except SystemError as e:
                        scan_aborted = True
                        st.error(str(e))
                        break
                    except (V20Error, RequestException, ValueError, KeyError) as e:
                        _errors.append(f"{_inst}/{_tf_name}: {e}")
                        continue

                    if _df is None:
                        continue

                    sig = detect_choch_v58(_df, _tf_name, _inst)
                    if not sig:
                        continue

                    _trend = sig["trend"]
                    _vol = sig["volatilite"]
                    _force = sig["force"]
                    _inst_disp = _inst.replace("_", "/")
                    _statut = compute_statut(sig["idx_break"], len(_df), _tf_name)

                    bb_needed = 40
                    bb_start = max(0, sig["idx_break"] + 1 - bb_needed)
                    _df_bb = _df.iloc[bb_start:sig["idx_break"] + 1]
                    _bb_res = compute_bb_width(_df_bb)
                    _bb_str = format_bb_width(_bb_res)

                    _dist_pct = calc_distance_pct(sig["level"], sig["close_price"])
                    signal_time = _df.index[sig["idx_break"]]

                    signal_id = (
                        f"{_inst}__{_tf_name}__{signal_time.strftime('%Y%m%dT%H%M')}"
                        f"__{sig['sig_type']}__{sig['direction']}"
                    )

                    _results.append({
                        "Instrument": _inst_disp,
                        "Paire": _inst_disp,
                        "_time_sort": signal_time,
                        "Timeframe": _tf_name,
                        "Type": sig["sig_type"],
                        "Ordre": "Achat" if sig["direction"] == "Bullish" else "Vente",
                        "Signal": f"{sig['direction']} {sig['sig_type']}",
                        "Niveau": format_niveau(sig["level"], _inst),
                        "Distance%": format_distance(_dist_pct),
                        "Volatilité": _vol,
                        "Force": _force,
                        "BB_Width": _bb_str,
                        "Statut": _statut,
                        "Heure (UTC)": signal_time.strftime("%Y-%m-%d %H:%M"),
                        "signal_id": signal_id,
                    })

                    if _statut in ("Fresh", "Aged"):
                        _pipeline_signals.append(
                            build_pipeline_payload_v58(
                                _df, _inst, _inst_disp, _tf_name, sig, _trend,
                                _scan_time, len(_df), _bb_res, _vol, _force,
                            )
                        )
            finally:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=False)

            if scan_aborted:
                raise SystemError(
                    "Scan interrompu à cause d'une erreur d'authentification."
                )

            if _errors:
                st.warning(f"{len(_errors)} erreur(s) : {'; '.join(_errors[:5])}")
            if _results:
                _df_sorted = pd.DataFrame(_results).sort_values(
                    "_time_sort", ascending=False
                )
                _df_result = (
                    _df_sorted
                    .drop_duplicates(subset="signal_id", keep="first")
                    .drop(columns=["_time_sort", "signal_id"])
                    .reset_index(drop=True)
                )
                st.session_state.df = _df_result
                st.session_state.pipeline_signals = _pipeline_signals
                st.session_state.png_buf = None
                st.session_state.pdf_buf = None
                st.success(
                    f"Scan terminé – {len(_df_result)} signaux sur {NUM_COMBOS} "
                    f"combinaisons | {len(_pipeline_signals)} dans le pipeline JSON"
                )
            else:
                st.info("Aucun signal CHoCH/BOS récent qualifié (Score ≥ 65)")
    except SystemError:
        pass
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.error(f"Erreur critique : {exc}")
        logger.exception(exc)
    finally:
        st.session_state.scanning = False

if "df" in st.session_state:
    render_results()
