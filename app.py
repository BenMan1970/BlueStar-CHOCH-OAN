"""
CHoCH Scanner v5.14 — Monolith, fully audited & hardened.
(Modules fusionnés pour compatibilité Streamlit Cloud)
"""
# pylint: disable=wrong-import-position, wrong-import-order, import-error
import matplotlib
matplotlib.use("Agg")

import io
import json
import logging
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import Literal, Optional, TypedDict

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

import numpy as np
import pandas as pd
import requests
import streamlit as st
from matplotlib.figure import Figure
from oandapyV20 import API
from oandapyV20.endpoints import instruments
from oandapyV20.exceptions import V20Error
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
SCANNER_VERSION = "5.14"

# ===================== CONFIG =====================
INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "DE30_EUR", "XAU_USD", "SPX500_USD", "NAS100_USD", "US30_USD",
]

VOLATILITY_STATIC = {
    "EUR_USD": "Basse", "GBP_USD": "Basse", "USD_JPY": "Basse",
    "USD_CHF": "Basse", "USD_CAD": "Basse", "AUD_USD": "Moyenne",
    "NZD_USD": "Moyenne", "EUR_GBP": "Moyenne", "EUR_JPY": "Moyenne",
    "EUR_CHF": "Moyenne", "EUR_AUD": "Moyenne", "EUR_CAD": "Moyenne",
    "EUR_NZD": "Moyenne", "GBP_JPY": "Haute", "GBP_CHF": "Haute",
    "GBP_AUD": "Haute", "GBP_CAD": "Haute", "GBP_NZD": "Haute",
    "AUD_JPY": "Haute", "AUD_CAD": "Moyenne", "AUD_CHF": "Haute",
    "AUD_NZD": "Moyenne", "CAD_JPY": "Haute", "CAD_CHF": "Haute",
    "CHF_JPY": "Haute", "NZD_JPY": "Haute", "NZD_CAD": "Moyenne",
    "NZD_CHF": "Haute", "DE30_EUR": "Très Haute", "XAU_USD": "Très Haute",
    "SPX500_USD": "Très Haute", "NAS100_USD": "Très Haute", "US30_USD": "Très Haute",
}

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

# ===================== TYPING =====================
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

# ===================== API THREAD-SAFE =====================
# threading.Lock est parfaitement adapté – pas d'asyncio dans ce projet.
_thread_local = threading.local()
_api_lock = threading.Lock()

class _AuthCounter:
    """Compteur thread-safe des erreurs 401."""

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

# ===================== UTILITAIRES =====================

def _compute_true_range(data: pd.DataFrame) -> np.ndarray:
    high = data["high"].values
    low = data["low"].values
    close = data["close"].values
    n = len(data)
    tr = np.zeros(n - 1)
    if n >= 2:
        typical_diff = (data.index[1] - data.index[0]).total_seconds()
    else:
        typical_diff = 3600
    for i in range(1, n):
        gap = (data.index[i] - data.index[i-1]).total_seconds()
        if gap > 2 * typical_diff:
            tr[i-1] = high[i] - low[i]
        else:
            tr[i-1] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1]),
            )
    return tr

def calc_atr_bundle(data: pd.DataFrame, inst: str, period: int = 14) -> tuple[float, str]:
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

def instrument_precision(inst: str) -> int:
    if any(k in inst for k in ["SPX500", "NAS100", "US30", "DE30", "XAU", "XAG"]):
        return 2
    if "JPY" in inst:
        return 3
    return 5

def format_niveau(niveau: Optional[float], inst: str) -> str:
    if niveau is None:
        return "N/A"
    return f"{niveau:.{instrument_precision(inst)}f}"

def calc_distance_pct(
    niveau: Optional[float], close_actuel: Optional[float]
) -> Optional[float]:
    if niveau is None or close_actuel is None or np.isclose(niveau, 0, atol=1e-8):
        return None
    dist = abs(close_actuel - niveau) / abs(niveau) * 100
    return dist if dist <= 100 else None

def format_distance(dist_pct: Optional[float]) -> str:
    if dist_pct is None:
        return "N/A"
    return f"{dist_pct:.3f}%"

def _local_hour(dt: datetime, tz_name: str) -> int:
    return dt.astimezone(ZoneInfo(tz_name)).hour

def get_session(
    dt: datetime,
) -> Literal["London_NY_Overlap", "London", "NewYork", "Tokyo", "Off"]:
    london_h = _local_hour(dt, "Europe/London")
    ny_h = _local_hour(dt, "America/New_York")
    tokyo_h = _local_hour(dt, "Asia/Tokyo")
    london = 8 <= london_h < 17
    ny = 9 <= ny_h < 17
    tokyo = 9 <= tokyo_h < 18
    if london and ny:
        return "London_NY_Overlap"
    if london:
        return "London"
    if ny:
        return "NewYork"
    if tokyo:
        return "Tokyo"
    return "Off"

def is_premium_session(s: str) -> bool:
    return s in ("London", "NewYork", "London_NY_Overlap")

def _parse_candle_row(c: dict, inst: str, gran: str) -> Optional[dict]:
    try:
        open_v = float(c["mid"]["o"])
        high_v = float(c["mid"]["h"])
        low_v = float(c["mid"]["l"])
        close_v = float(c["mid"]["c"])
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "Bougie malformée ignorée [%s/%s] t=%s: %s",
            inst, gran, c.get("time"), exc,
        )
        return None
    if not all(np.isfinite(v) for v in (open_v, high_v, low_v, close_v)):
        logger.warning("Prix non-fini ignoré [%s/%s] t=%s", inst, gran, c.get("time"))
        return None
    if low_v > min(open_v, close_v) or max(open_v, close_v) > high_v:
        logger.warning(
            "OHLC incohérent ignoré [%s/%s] t=%s", inst, gran, c.get("time")
        )
        return None
    return {
        "time": pd.to_datetime(c["time"], utc=True),
        "open": open_v,
        "high": high_v,
        "low": low_v,
        "close": close_v,
    }

def _handle_v20_error(exc: V20Error, inst: str, gran: str) -> None:
    if exc.code == 401:
        auth_cnt = _auth_counter.increment()
        if auth_cnt > 3:
            logger.critical("Trop d'erreurs 401 – arrêt du scan.")
            raise SystemError("Compte OANDA bloqué – vérifiez le token.") from exc
        if hasattr(_thread_local, "api"):
            del _thread_local.api
        logger.error(
            "V20Error 401 [%s/%s] – auth failure #%d", inst, gran, auth_cnt
        )
    elif exc.code == 429:
        logger.warning("V20Error 429 [%s/%s] – rate limited", inst, gran)
    else:
        logger.warning(
            "V20Error [%s/%s] code=%s: %s", inst, gran, exc.code, exc
        )

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
        rows = [
            r
            for c in candles
            if (r := _parse_candle_row(c, inst, gran)) is not None
        ]
        if len(rows) < 50:
            return None
        df = pd.DataFrame(rows)
        df.set_index("time", inplace=True)
        df = df[~df.index.duplicated(keep="last")]
        return df
    except V20Error as exc:
        _handle_v20_error(exc, inst, gran)
        return None
    except requests.RequestException as exc:
        logger.warning("Network error [%s/%s]: %s", inst, gran, exc)
        return None
    except (ValueError, KeyError, TypeError) as exc:
        logger.error(
            "Data parsing error in get_candles [%s/%s]: %s", inst, gran, exc
        )
        return None

def compute_bb_width(
    data: pd.DataFrame, length: int = 20, std: int = 2
) -> tuple[Optional[float], str]:
    close = data["close"]
    if len(close) < length * 2:
        return None, "N/A"
    sma = close.rolling(length).mean()
    std_dev = close.rolling(length).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    bb_w = (upper - lower) / sma
    bb_avg = bb_w.rolling(length).mean()
    avg_last = bb_avg.iloc[-1]
    if pd.isna(avg_last) or np.isclose(avg_last, 0, atol=1e-10):
        return None, "N/A"
    bb_avg_safe = bb_avg.replace(0, np.nan)
    pct = ((bb_w - bb_avg) / bb_avg_safe * 100).iloc[-1]
    if pd.isna(pct):
        return None, "N/A"
    if pct <= -25:
        regime = "Squeeze"
    elif pct >= 25:
        regime = "Expansion"
    else:
        regime = "Normal"
    return float(pct), regime

def format_bb_width(bb_result: tuple[Optional[float], str]) -> str:
    pct, regime = bb_result
    if pct is None:
        return "N/A"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%_{regime}"

def compute_statut(
    idx_sig: Optional[int], len_df: int, tf: str
) -> Literal["Fresh", "Aged", "Stale", "N/A"]:
    if idx_sig is None:
        return "N/A"
    candles_elapsed = (len_df - 1) - idx_sig
    thresholds = TF_STATUT.get(tf, {"Fresh": 2, "Aged": 5})
    if candles_elapsed <= thresholds["Fresh"]:
        return "Fresh"
    if candles_elapsed <= thresholds["Aged"]:
        return "Aged"
    return "Stale"

# ===================== DÉTECTION DES SWINGS =====================
def _classify_swings(pivots: list) -> list[SwingDict]:
    swings: list[SwingDict] = []
    prev_h: Optional[float] = None
    prev_l: Optional[float] = None
    for idx, price, k in pivots:
        if k == "H":
            kind = "HH" if (prev_h is None or price > prev_h) else "LH"
            swings.append({"idx": idx, "price": price, "kind": kind})
            prev_h = price
        else:
            kind = "HL" if (prev_l is None or price > prev_l) else "LL"
            swings.append({"idx": idx, "price": price, "kind": kind})
            prev_l = price
    return swings

def _build_pivot_mask(high_s: pd.Series, low_s: pd.Series, win: int):
    roll_max = high_s.rolling(window=win, center=True, min_periods=win).max()
    roll_min = low_s.rolling(window=win, center=True, min_periods=win).min()
    h_mask = (high_s == roll_max) & high_s.notna()
    l_mask = (low_s == roll_min) & low_s.notna()
    return h_mask, l_mask

def detect_swing_points(data: pd.DataFrame, tf: str) -> list[SwingDict]:
    lookback = SWING_LOOKBACK.get(tf, 5)
    history = SWING_HISTORY.get(tf, 60)
    high_arr = data["high"].values
    low_arr = data["low"].values
    n = len(high_arr)
    high_s = pd.Series(high_arr)
    low_s = pd.Series(low_arr)
    win = 2 * lookback + 1
    h_mask, l_mask = _build_pivot_mask(high_s, low_s, win)
    start = max(lookback, n - history - lookback)
    end = n - lookback - 1
    pivots = []
    for i in range(start, end):
        if h_mask.iloc[i]:
            pivots.append((i, float(high_arr[i]), "H"))
        if l_mask.iloc[i]:
            pivots.append((i, float(low_arr[i]), "L"))
    pivots.sort(key=lambda x: x[0])
    seen = {}
    for pos, (i, price, k) in enumerate(pivots):
        seen[(price, k)] = pos
    dedup_positions = set(seen.values())
    pivots = [p for pos, p in enumerate(pivots) if pos in dedup_positions]
    pivots.sort(key=lambda x: x[0])
    return _classify_swings(pivots)

def _last_high_low(swings: list[SwingDict]):
    highs = [s for s in swings if s["kind"] in ("HH", "LH")]
    lows = [s for s in swings if s["kind"] in ("HL", "LL")]
    if not highs or not lows:
        return None, None
    return highs[-1]["kind"], lows[-1]["kind"]

def get_structural_trend(
    swings: list[SwingDict],
) -> Literal["Bullish", "Bearish", "Range"]:
    if len(swings) < 4:
        return "Range"
    last_high, last_low = _last_high_low(swings[-6:])
    if last_high is None:
        return "Range"
    if last_high == "HH" and last_low == "HL":
        return "Bullish"
    if last_high == "LH" and last_low == "LL":
        return "Bearish"
    return "Range"

# ===================== CORE DETECTION =====================
_SigResult = tuple[
    Optional[Literal["CHoCH", "BOS"]],
    Optional[Literal["Bullish", "Bearish"]],
    Optional[float],
]
_NONE_SIG: _SigResult = (None, None, None)

def _resolve_bullish_trend(close_arr, idx: int, prev_swings) -> _SigResult:
    hl_list = [s for s in prev_swings if s["kind"] == "HL"]
    if hl_list:
        ref = hl_list[-1]["price"]
        if close_arr[idx] < ref <= close_arr[idx - 1]:
            return "CHoCH", "Bearish", ref
    hh_list = [s for s in prev_swings if s["kind"] == "HH"]
    if hh_list:
        ref = hh_list[-1]["price"]
        if close_arr[idx - 1] <= ref < close_arr[idx]:
            return "BOS", "Bullish", ref
    return _NONE_SIG

def _resolve_bearish_trend(close_arr, idx: int, prev_swings) -> _SigResult:
    lh_list = [s for s in prev_swings if s["kind"] == "LH"]
    if lh_list:
        ref = lh_list[-1]["price"]
        if close_arr[idx - 1] <= ref < close_arr[idx]:
            return "CHoCH", "Bullish", ref
    ll_list = [s for s in prev_swings if s["kind"] == "LL"]
    if ll_list:
        ref = ll_list[-1]["price"]
        if close_arr[idx] < ref <= close_arr[idx - 1]:
            return "BOS", "Bearish", ref
    return _NONE_SIG

def _resolve_signal(trend, close_arr, idx: int, prev_swings) -> _SigResult:
    if trend == "Bullish":
        return _resolve_bullish_trend(close_arr, idx, prev_swings)
    if trend == "Bearish":
        return _resolve_bearish_trend(close_arr, idx, prev_swings)
    return _NONE_SIG

# pylint: disable=too-many-arguments
def _detect_liquidity_sweep(
    high_arr, low_arr, idx, prev_swings, atr_val, direction
) -> bool:
    if direction == "Bearish":
        candidates = [s for s in prev_swings if s["kind"] in ("HH", "LH")]
        return any(
            high_arr[idx] > s["price"]
            and (high_arr[idx] - s["price"]) > (atr_val * 0.25)
            for s in candidates
        )
    candidates = [s for s in prev_swings if s["kind"] in ("HL", "LL")]
    return any(
        low_arr[idx] < s["price"]
        and (s["price"] - low_arr[idx]) > (atr_val * 0.25)
        for s in candidates
    )
# pylint: enable=too-many-arguments

# pylint: disable=too-many-arguments
def _compute_confluence_score(
    dist_atr, idx, df_index, has_sweep, sig_type, len_df, tf
) -> int:
    score = 25
    if dist_atr <= 1.0:
        score += 15
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
# pylint: enable=too-many-arguments

def _evaluate_candle(
    idx, close_arr, high_arr, low_arr, open_arr,
    prev_swings, atr_val, trend, df_index, n, tf,
) -> Optional[SignalDict]:
    sig_type, direction, level = _resolve_signal(
        trend, close_arr, idx, prev_swings
    )
    if sig_type is None:
        return None
    rng_v = high_arr[idx] - low_arr[idx]
    if rng_v <= 0:
        return None
    body_ratio = abs(close_arr[idx] - open_arr[idx]) / rng_v
    if body_ratio < 0.40:
        return None
    force_label = "Fort" if body_ratio >= 0.60 else "Moyen"
    has_sweep = (
        sig_type == "CHoCH"
        and _detect_liquidity_sweep(
            high_arr, low_arr, idx, prev_swings, atr_val, direction
        )
    )
    dist_atr = abs(close_arr[idx] - level) / atr_val
    if dist_atr > ATR_DIST_MULT:
        return None
    score = _compute_confluence_score(
        dist_atr, idx, df_index, has_sweep, sig_type, n, tf
    )
    if score < MIN_SCORE:
        return None
    return {
        "sig_type": sig_type,
        "direction": direction,
        "level": float(level),
        "idx_break": int(idx),
        "close_price": float(close_arr[idx]),
        "current_price": float(close_arr[-1]),
        "has_sweep": bool(has_sweep),
        "atr_val": float(atr_val),
        "trend": trend,
        "force": force_label,
        "dist_atr": float(dist_atr),
        "score": int(score),
    }

def _scan_window_for_signal(
    df, swings, trend, inst, tf
) -> Optional[SignalDict]:
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
        signal = _evaluate_candle(
            idx, close_arr, high_arr, low_arr, open_arr,
            prev_swings, atr_val, trend, df.index, n, tf,
        )
        if signal is not None:
            signal["volatilite"] = volatilite_label
            return signal
    return None

def detect_choch_v58(df, tf, inst) -> Optional[SignalDict]:
    swings = detect_swing_points(df, tf)
    trend = get_structural_trend(swings)
    if trend == "Range":
        return None
    return _scan_window_for_signal(df, swings, trend, inst, tf)

# pylint: disable=too-many-arguments
def build_pipeline_payload_v58(
    df, inst, inst_disp, tf_name, signal_info, trend,
    scan_time, len_df, bb_result, volatilite, force,
) -> dict:
    time_sig = df.index[signal_info["idx_break"]]
    session = get_session(time_sig)
    dist_pct = calc_distance_pct(
        signal_info["level"], signal_info["close_price"]
    )
    curr_dist_pct = calc_distance_pct(
        signal_info["level"], signal_info["current_price"]
    )
    candles_since = (len_df - 1) - signal_info["idx_break"]
    statut = compute_statut(signal_info["idx_break"], len_df, tf_name)
    prec = instrument_precision(inst)
    bb_pct, bb_regime = bb_result

    return {
        "signal_id": (
            f"{inst}__{tf_name}__{time_sig.strftime('%Y%m%dT%H%M')}"
            f"__scan{scan_time.strftime('%Y%m%dT%H%M')}"
        ),
        "scanner_version": SCANNER_VERSION,
        "generated_at": scan_time.isoformat(),
        "pair": inst_disp,
        "pair_oanda": inst,
        "timeframe": tf_name,
        "type": signal_info["sig_type"],
        "direction": signal_info["direction"],
        "is_bullish": signal_info["direction"] == "Bullish",
        "order": "buy" if signal_info["direction"] == "Bullish" else "sell",
        "trend": trend,
        "is_choch": signal_info["sig_type"] == "CHoCH",
        "status": statut,
        "confluence_score": signal_info["score"],
        "level": round(float(signal_info["level"]), prec),
        "close_price": round(float(signal_info["close_price"]), prec),
        "current_price": round(float(signal_info["current_price"]), prec),
        "distance_pct": round(dist_pct, 4) if dist_pct is not None else None,
        "current_distance_pct": (
            round(curr_dist_pct, 4) if curr_dist_pct is not None else None
        ),
        "distance_atr_multiple": round(signal_info["dist_atr"], 2),
        "volatility": volatilite,
        "force": force,
        "bb_width_pct": bb_pct,
        "bb_regime": bb_regime,
        "session": session,
        "signal_time": time_sig.isoformat(),
        "candles_elapsed": candles_since,
    }
# pylint: enable=too-many-arguments

# ===================== EXPORT =====================
def _json_default(obj: object) -> object:
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        val = float(obj)
        return None if (np.isnan(val) or np.isinf(val)) else val
    return str(obj)

def create_pdf(df_export: pd.DataFrame) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=20,
        rightMargin=20,
        topMargin=40,
        bottomMargin=40,
    )
    elements = []
    styles = getSampleStyleSheet()
    elements.append(
        Paragraph(
            f"Rapport des Signaux CHoCH v{SCANNER_VERSION}", styles["Title"]
        )
    )
    elements.append(
        Paragraph(
            f"Généré le {datetime.now(timezone.utc).strftime('%d/%m/%Y à %H:%M')} UTC",
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 20))

    cols_present = [c for c in EXPORT_COLS if c in df_export.columns]
    col_widths_map = {c: 60 for c in cols_present}
    col_widths_map.update(
        {"Instrument": 65, "Distance%": 52, "Statut": 45, "Heure (UTC)": 105}
    )
    col_widths = [col_widths_map.get(c, 60) for c in cols_present]

    data = [cols_present] + df_export[cols_present].values.tolist()
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.beige]),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

def generate_png(data: pd.DataFrame, display_cols: list[str]) -> io.BytesIO:
    fig = Figure(figsize=(22, min(max(5, len(data) * 0.35), 30)))
    ax = fig.add_subplot(111)
    ax.axis("off")
    disp = data[[c for c in display_cols if c in data.columns]]
    tbl = ax.table(
        cellText=disp.values,
        colLabels=disp.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    return buf

# ===================== UI RENDERING (complexité < 10) =====================
def _style_bb(val: object) -> str:
    val_str = str(val)
    if "Squeeze" in val_str:
        return "color:#ff9800;font-weight:bold"
    if "Expansion" in val_str:
        return "color:#ab47bc;font-weight:bold"
    return "color:#90a4ae"

def _style_distance(val: object) -> str:
    try:
        v = float(str(val).replace("%", ""))
        if v <= 0.15:
            return "color:#00c853;font-weight:bold"
        if v <= 0.40:
            return "color:#ff9800;font-weight:bold"
        return "color:#ff5252;font-weight:bold"
    except (ValueError, TypeError):
        return "color:#90a4ae"

def _render_downloads(
    df_all: pd.DataFrame, df_export: pd.DataFrame, pipeline_signals: list,
    scan_time_meta: datetime,
) -> None:
    """Affiche les boutons de téléchargement CSV, PNG, PDF, JSON."""
    _ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        csv_cols = [c for c in EXPORT_COLS if c in df_export.columns]
        st.download_button(
            "CSV",
            df_export[csv_cols].to_csv(index=False).encode(),
            f"choch_{_ts}.csv",
            "text/csv",
        )
    with c2:
        if st.session_state.get("png_buf") is None:
            st.session_state.png_buf = generate_png(df_all, DISPLAY_COLS)
        st.download_button(
            "PNG",
            st.session_state.png_buf.getvalue(),
            f"choch_{_ts}.png",
            "image/png",
        )
    with c3:
        if st.session_state.get("pdf_buf") is None:
            st.session_state.pdf_buf = create_pdf(df_export)
        st.download_button(
            "PDF",
            st.session_state.pdf_buf.getvalue(),
            f"choch_signaux_{_ts}.pdf",
            "application/pdf",
        )
    with c4:
        pipeline_json = json.dumps(
            {
                "meta": {
                    "scanner_version": SCANNER_VERSION,
                    "generated_at": scan_time_meta.isoformat(),
                    "signal_count": len(pipeline_signals),
                },
                "signals": pipeline_signals,
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ).encode("utf-8")
        st.download_button(
            "JSON",
            pipeline_json,
            f"choch_pipeline_{_ts}.json",
            "application/json",
        )

def _render_dataframe(df_all: pd.DataFrame) -> None:
    """Affiche le tableau principal avec le style coloré."""
    cols_disp = [c for c in DISPLAY_COLS if c in df_all.columns]
    styled = (
        df_all[cols_disp]
        .style
        .map(
            lambda x: (
                "color:#e879f9;font-weight:bold" if x == "CHoCH"
                else "color:#94a3b8" if x == "BOS"
                else ""
            ),
            subset=["Type"],
        )
        .map(
            lambda x: (
                "color:#00c853;font-weight:bold" if x == "Achat"
                else "color:#ff5252;font-weight:bold" if x == "Vente"
                else ""
            ),
            subset=["Ordre"],
        )
        .map(
            lambda x: (
                "color:#00c853" if "Bull" in str(x)
                else "color:#ff5252" if "Bear" in str(x)
                else ""
            ),
            subset=["Signal"],
        )
        .map(
            lambda x: (
                "color:#00c853;font-weight:bold" if x == "Fort"
                else "color:#ff5252" if x == "Faible"
                else "color:#ff9800"
            ),
            subset=["Force"],
        )
        .map(_style_bb, subset=["BB_Width"])
        .map(_style_distance, subset=["Distance%"])
        .map(
            lambda x: (
                "color:#00c853;font-weight:bold" if x == "Fresh"
                else "color:#ff9800;font-weight:bold" if x == "Aged"
                else "color:#ff5252;font-weight:bold" if x == "Stale"
                else ""
            ),
            subset=["Statut"],
        )
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)

def _render_pipeline_section(pipeline_signals: list) -> None:
    """Affiche l'aperçu JSON si des signaux sont présents."""
    if pipeline_signals:
        with st.expander(
            f"Aperçu JSON Pipeline ({len(pipeline_signals)} signaux Fresh/Aged)"
        ):
            st.json(pipeline_signals[0])

def render_results() -> None:
    """Point d'entrée principal pour l'affichage des résultats."""
    _df_all = st.session_state.df.copy()
    _df_export = _df_all[_df_all["Statut"].isin(["Fresh", "Aged"])].copy()
    _pipeline_signals = st.session_state.get("pipeline_signals", [])
    _scan_time_meta = st.session_state.get(
        "scan_time", datetime.now(timezone.utc)
    )

    _n_stale = len(_df_all[_df_all["Statut"] == "Stale"])
    if _n_stale > 0:
        st.info(
            f"{_n_stale} signal(s) Stale visible(s) dans le tableau "
            "— exclus des exports."
        )

    _render_downloads(_df_all, _df_export, _pipeline_signals, _scan_time_meta)
    _render_dataframe(_df_all)
    _render_pipeline_section(_pipeline_signals)

# ===================== MAIN APP =====================
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
                    except (
                        V20Error, requests.RequestException, ValueError, KeyError
                    ) as e:
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

                    _dist_pct = calc_distance_pct(
                        sig["level"], sig["close_price"]
                    )
                    signal_time = _df.index[sig["idx_break"]]

                    signal_id = (
                        f"{_inst}__{_tf_name}__"
                        f"{signal_time.strftime('%Y%m%dT%H%M')}"
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
                st.warning(
                    f"{len(_errors)} erreur(s) : {'; '.join(_errors[:5])}"
                )
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
