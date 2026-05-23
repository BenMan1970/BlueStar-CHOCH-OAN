# -*- coding: utf-8 -*-
"""
CHoCH Scanner v5.10 — Change of Character & Break of Structure detector.
Audited and hardened (concurrency, auth circuit breaker, signal deduplication,
gap-aware ATR, session bonus freshness).
"""

import io
import json
import logging
import threading
import functools
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import Literal, Optional, TypedDict

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

import numpy as np
import pandas as pd
import requests
import streamlit as st
from oandapyV20 import API
from oandapyV20.endpoints import instruments
from oandapyV20.exceptions import V20Error
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
SCANNER_VERSION = "5.10"

# ===================== CONFIG (inchangé) =====================
INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "DE30_EUR", "XAU_USD", "SPX500_USD", "NAS100_USD", "US30_USD",
]

VOLATILITY_STATIC = { ... }  # inchangé

TIMEFRAMES = {"H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"}
SWING_LOOKBACK = {"H1": 5, "H4": 5, "D1": 4, "Weekly": 3}
SWING_HISTORY = {"H1": 120, "H4": 90, "D1": 60, "Weekly": 26}
ATR_DIST_MULT = 1.8
MIN_SCORE = 65

SCAN_GLOBAL_TIMEOUT = 180

TF_STATUT = {
    "H1":     {"Fresh": 4, "Aged": 12},
    "H4":     {"Fresh": 3, "Aged": 8},
    "D1":     {"Fresh": 2, "Aged": 5},
    "Weekly": {"Fresh": 2, "Aged": 4},
}

DISPLAY_COLS = [
    "Instrument", "Timeframe", "Type", "Ordre", "Signal",
    "Niveau", "Distance%", "Volatilité", "Force", "BB_Width", "Statut", "Heure (UTC)",
]
EXPORT_COLS = DISPLAY_COLS.copy()

GRAN_COUNT = {"H1": 400, "H4": 300, "D": 200, "W": 120}

# ===================== TYPING (inchangé) =====================
class SwingDict(TypedDict):
    idx: int
    price: float
    kind: Literal["HH", "LH", "HL", "LL"]

class SignalDict(TypedDict, total=False):
    sig_type: Literal["CHoCH", "BOS"]
    direction: Literal["Bullish", "Bearish"]
    level: float
    idx_break: int
    close_price: float
    current_price: float
    has_sweep: bool
    atr_val: float
    volatilite: str
    trend: Literal["Bullish", "Bearish", "Range"]
    force: Literal["Fort", "Moyen", "Faible"]
    dist_atr: float
    score: int

# ===================== API THREAD-SAFE (modifié) =====================
_thread_local = threading.local()
_api_lock = threading.Lock()


class _AuthCounter:
    def __init__(self) -> None:
        self._count = 0
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._count += 1
            return self._count

    def reset(self) -> None:
        with self._lock:
            self._count = 0

    def get(self) -> int:
        with self._lock:
            return self._count


_auth_counter = _AuthCounter()


def _get_api() -> API:
    if not hasattr(_thread_local, "api"):
        with _api_lock:
            if not hasattr(_thread_local, "api"):
                try:
                    _thread_local.api = API(
                        access_token=st.secrets["OANDA_ACCESS_TOKEN"],
                        request_params={"timeout": 12},
                    )
                except Exception as exc:
                    logger.critical("Impossible d'initialiser l'API OANDA : %s", exc)
                    raise
    return _thread_local.api


try:
    _ = st.secrets["OANDA_ACCESS_TOKEN"]
except KeyError as exc:
    st.error(f"Clé API OANDA manquante dans les secrets : {exc}")
    st.stop()

# ===================== UTILITAIRES (gap-aware ATR) =====================

# Suppression de @st.cache_data — le cache n'est plus utilisé
def calc_atr_bundle(data: pd.DataFrame, inst: str, period: int = 14) -> tuple[float, str]:
    """ATR avec gestion des gaps temporels."""
    tr = _compute_true_range(data)
    if len(tr) < period * 3:
        return float("nan"), VOLATILITY_STATIC.get(inst, "Moyenne")
    atr_val = float(pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().iloc[-1])
    window = tr[-100:] if len(tr) >= 100 else tr
    median_tr = float(np.median(window))
    if np.isnan(atr_val) or np.isclose(median_tr, 0, atol=1e-10):
        return atr_val, VOLATILITY_STATIC.get(inst, "Moyenne")
    ratio = atr_val / median_tr
    if ratio >= 1.8:
        return atr_val, "Très Haute"
    if ratio >= 1.2:
        return atr_val, "Haute"
    if ratio >= 0.7:
        return atr_val, "Moyenne"
    return atr_val, "Basse"


def _compute_true_range(data: pd.DataFrame) -> np.ndarray:
    """True Range avec détection de gap temporel : si >2x la granularité,
    on ignore close précédent pour ne pas fausser le TR."""
    high = data["high"].values
    low = data["low"].values
    close = data["close"].values
    n = len(data)
    tr = np.zeros(n - 1)
    # Déterminer la granularité typique à partir des premiers écarts (suppose index régulier)
    if n >= 2:
        typical_diff = (data.index[1] - data.index[0]).total_seconds()
    else:
        typical_diff = 3600  # fallback 1h
    for i in range(1, n):
        gap = (data.index[i] - data.index[i-1]).total_seconds()
        if gap > 2 * typical_diff:
            # Gap anormal → TR = high - low uniquement
            tr[i-1] = high[i] - low[i]
        else:
            tr[i-1] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
    return tr


def instrument_precision(inst: str) -> int:  # inchangé
    ...


def format_niveau(niveau: Optional[float], inst: str) -> str:  # inchangé
    ...


def calc_distance_pct(niveau, close_actuel) -> Optional[float]:  # inchangé
    ...


def format_distance(dist_pct) -> str:  # inchangé
    ...


def _local_hour(dt: datetime, tz_name: str) -> int:  # inchangé
    ...


def get_session(dt: datetime) -> Literal["London_NY_Overlap", ...]:  # inchangé
    ...


def is_premium_session(s: str) -> bool:  # inchangé
    ...


def _parse_candle_row(c: dict, inst: str, gran: str) -> Optional[dict]:  # inchangé
    ...


# Suppression du @st.cache_data pour get_candles
def get_candles(inst: str, gran: str) -> Optional[pd.DataFrame]:
    count = GRAN_COUNT.get(gran, 300)
    try:
        req = instruments.InstrumentsCandles(
            instrument=inst,
            params={"count": count, "granularity": gran},
        )
        _get_api().request(req)
        candles = [c for c in req.response.get("candles", []) if c.get("complete")]
        if len(candles) < 50:
            return None
        rows = [r for c in candles if (r := _parse_candle_row(c, inst, gran)) is not None]
        if len(rows) < 50:
            return None
        df = pd.DataFrame(rows)
        df.set_index("time", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        return df
    except V20Error as exc:
        if exc.code == 401:
            auth_cnt = _auth_counter.increment()
            if auth_cnt > 3:  # CIRCUIT BREAKER
                logger.critical("Trop d'erreurs 401 – arrêt du scan.")
                raise SystemError("Compte OANDA bloqué – vérifiez le token.") from exc
            if hasattr(_thread_local, "api"):
                del _thread_local.api
            logger.error("V20Error 401 [%s/%s] – auth failure #%d", inst, gran, auth_cnt)
        elif exc.code == 429:
            logger.warning("V20Error 429 [%s/%s] – rate limited", inst, gran)
        else:
            logger.warning("V20Error [%s/%s] code=%s: %s", inst, gran, exc.code, exc)
        return None
    except requests.RequestException as exc:
        logger.warning("Network error [%s/%s]: %s", inst, gran, exc)
        return None
    except Exception as exc:
        logger.error("Unexpected error in get_candles [%s/%s]: %s", inst, gran, exc, exc_info=True)
        return None


# Suppression du @st.cache_data pour compute_bb_width
def compute_bb_width(data: pd.DataFrame, length: int = 20, std: int = 2) -> tuple[Optional[float], str]:
    # inchangé
    ...


def format_bb_width(bb_result) -> str:  # inchangé
    ...


def compute_statut(idx_sig, len_df, tf) -> Literal["Fresh", "Aged", "Stale", "N/A"]:  # inchangé
    ...


# ===================== CORE V5.9 (modifié pour session bonus) =====================

# _classify_swings inchangé

# Suppression du @st.cache_data pour detect_swing_points
def detect_swing_points(data: pd.DataFrame, tf: str) -> list[SwingDict]:
    # inchangé, suppression du décorateur
    ...


def get_structural_trend(swings) -> Literal["Bullish", "Bearish", "Range"]:  # inchangé
    ...


# Les _resolve_* inchangés

def _detect_liquidity_sweep(...) -> bool:  # inchangé
    ...


def _compute_confluence_score(
    dist_atr: float,
    idx: int,
    df_index: pd.DatetimeIndex,
    has_sweep: bool,
    sig_type: Literal["CHoCH", "BOS"],
    len_df: int,
    tf: str,  # nouveau paramètre pour vérifier la fraîcheur
) -> int:
    score = 25
    if dist_atr <= 1.0:
        score += 15
    # Bonus session uniquement si signal Fresh
    candles_elapsed = len_df - 1 - idx
    thresholds = TF_STATUT.get(tf, {"Fresh": 2})
    if candles_elapsed <= thresholds["Fresh"]:
        if is_premium_session(get_session(df_index[idx])):
            score += 20
    if has_sweep:
        score += 15
    if sig_type == "CHoCH":
        score += 10
    return min(score, 100)


# Suppression du @st.cache_data pour detect_choch_v58
def detect_choch_v58(df: pd.DataFrame, tf: str, inst: str) -> Optional[SignalDict]:
    swings = detect_swing_points(df, tf)
    trend = get_structural_trend(swings)
    if trend == "Range":
        return None

    close_arr = df["close"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    open_arr = df["open"].values
    n = len(close_arr)

    atr_val, volatilite_label = calc_atr_bundle(df, inst)
    if np.isnan(atr_val) or atr_val <= 0:
        return None

    for offset in range(5):
        idx = n - 1 - offset
        if idx < 3:
            break
        prev_swings = [s for s in swings if s["idx"] < idx - 2]
        if not prev_swings:
            continue
        sig_type, direction, level = _resolve_signal(trend, close_arr, idx, prev_swings)
        if sig_type is None:
            continue

        rng_v = high_arr[idx] - low_arr[idx]
        if rng_v <= 0:
            continue
        body_ratio = abs(close_arr[idx] - open_arr[idx]) / rng_v
        if body_ratio < 0.40:
            continue
        force_label = "Fort" if body_ratio >= 0.60 else "Moyen"

        has_sweep = (
            sig_type == "CHoCH"
            and _detect_liquidity_sweep(high_arr, low_arr, idx, prev_swings, atr_val, direction)
        )

        dist_atr = abs(close_arr[idx] - level) / atr_val
        if dist_atr > ATR_DIST_MULT:
            continue

        score = _compute_confluence_score(
            dist_atr, idx, df.index, has_sweep, sig_type, n, tf
        )
        if score < MIN_SCORE:
            continue

        return {
            "sig_type": sig_type,
            "direction": direction,
            "level": float(level),
            "idx_break": int(idx),
            "close_price": float(close_arr[idx]),
            "current_price": float(close_arr[-1]),
            "has_sweep": bool(has_sweep),
            "atr_val": float(atr_val),
            "volatilite": volatilite_label,
            "trend": trend,
            "force": force_label,
            "dist_atr": float(dist_atr),
            "score": int(score),
        }
    return None


def build_pipeline_payload_v58(...) -> dict:  # inchangé
    ...


# ===================== EXPORT (inchangé) =====================

def _json_default(obj): ...
def create_pdf(df_export): ...
def generate_png(data, display_cols): ...

# ===================== UI (modifié pour déduplication et gestion des erreurs) =====================
st.set_page_config(page_title="CHoCH Scanner v5.10", layout="wide")
st.title("Scanner Change of Character (CHoCH) — v5.10 Intraday")

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
    _auth_counter.reset()  # nouveau scan, on réinitialise le compteur 401

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
                    st.warning(f"Timeout global – {len(not_done)} requête(s) ignorée(s), résultats partiels.")
                    for f in not_done:
                        f.cancel()

                for future in done:
                    _inst, _tf_name = futures[future]
                    try:
                        _df = future.result()
                    except CancelledError:
                        continue  # normal après timeout
                    except SystemError as e:
                        # Circuit breaker 401 → arrêt immédiat
                        scan_aborted = True
                        st.error(str(e))
                        break
                    except Exception as exc:
                        _errors.append(f"{_inst}/{_tf_name}: {exc}")
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

                    # BB width (sur fenêtre pertinente)
                    bb_needed = 40
                    bb_start = max(0, sig["idx_break"] + 1 - bb_needed)
                    _df_bb = _df.iloc[bb_start:sig["idx_break"] + 1]
                    _bb_res = compute_bb_width(_df_bb)
                    _bb_str = format_bb_width(_bb_res)
                    _dist_pct = calc_distance_pct(sig["level"], sig["close_price"])
                    signal_time = _df.index[sig["idx_break"]]

                    # Construire un identifiant unique pour déduplication réelle
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
                # Ne pas poursuivre l'affichage, le scan est avorté
                raise SystemError("Scan interrompu à cause d'une erreur d'authentification.")

            if _errors:
                st.warning(f"{len(_errors)} erreur(s) : {'; '.join(_errors[:5])}")
            if _results:
                _df_sorted = pd.DataFrame(_results).sort_values("_time_sort", ascending=False)
                # Déduplication par signal_id (supprime les vrais doublons accidentels)
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
        # Déjà affiché, on sort proprement
        pass
    except Exception as exc:
        st.error(f"Erreur critique : {exc}")
        logger.exception(exc)
    finally:
        st.session_state.scanning = False

# ... (suite de l'UI inchangée : téléchargements, style, affichage)
