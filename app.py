"""
CHoCH Scanner v5.15 — Production-grade hardened build.

Monolithic deployment (Streamlit Cloud compatible) but architected as strict
layered modules in a single file:

    1.  Constants & rule registry (frozen, versioned)
    2.  Structured JSON logging
    3.  Cached resources (API, ThreadPool, TZ)
    4.  Pure domain layer (deterministic, side-effect free)
    5.  I/O layer (OANDA with retry, circuit breaker, dedup)
    6.  Orchestration (scan as pure function)
    7.  UI layer (presentation only)

Invariants enforced:
    - Single source of truth for every signal field (no UI/pipeline drift).
    - signal_id is deterministic and reproducible.
    - rule_version is embedded in every emitted payload.
    - No global mutable state shared across Streamlit sessions.
    - No silent JSON serialization failure (whitelisted converter).
    - No thread/pool leak on script reruns.
"""
from __future__ import annotations

# pylint: disable=wrong-import-position
import matplotlib
matplotlib.use("Agg")

import hashlib
import io
import json
import logging
import math
import os
import sys
import threading
import time
import uuid
from concurrent.futures import (
    CancelledError,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Final, Literal, Mapping, Optional, Sequence, TypedDict

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - py<3.9 fallback
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

# =====================================================================
# SECTION 1 — CONSTANTS & RULE REGISTRY
# =====================================================================

SCANNER_VERSION: Final[str] = "5.15"
RULE_VERSION: Final[str] = "choch.v58.r7"  # r7: revert BUG-B (detection window), BUG-E (CHoCH pivot over-restriction), BUG-F (session bonus coupling); preserve BUG-A (BOS), BUG-C (look-ahead), BUG-G (sweep scope)

INSTRUMENTS: Final[tuple[str, ...]] = (
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "DE30_EUR", "XAU_USD", "SPX500_USD", "NAS100_USD", "US30_USD",
)

VOLATILITY_STATIC: Final[Mapping[str, str]] = {
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
    "SPX500_USD": "Très Haute", "NAS100_USD": "Très Haute",
    "US30_USD": "Très Haute",
}

TIMEFRAMES: Final[Mapping[str, str]] = {
    "H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W",
}
SWING_LOOKBACK: Final[Mapping[str, int]] = {
    "H1": 5, "H4": 5, "D1": 4, "Weekly": 3,
}
SWING_HISTORY: Final[Mapping[str, int]] = {
    "H1": 120, "H4": 90, "D1": 60, "Weekly": 26,
}
GRAN_COUNT: Final[Mapping[str, int]] = {
    "H1": 400, "H4": 300, "D": 200, "W": 120,
}

# r7: explicit per-timeframe detection window.
# BUG-B (r6) falsely claimed the offset loop was redundant. _scan_one calls
# detect_choch() exactly ONCE per (inst, tf); there is no per-candle outer
# loop and no detect_at_idx() filter anywhere in the codebase. Removing the
# loop collapsed detection to the last closed candle only, losing ~80% of
# valid signals (especially D1/Weekly where CHoCHs rarely form on N-1).
# D1/Weekly use shorter windows because a 5-candle lookback would detect
# signals already Stale by definition (TF_STATUT["D1"]["Aged"] = 5,
# TF_STATUT["Weekly"]["Aged"] = 4).
DETECTION_LOOKBACK: Final[Mapping[str, int]] = {
    "H1": 5, "H4": 5, "D1": 3, "Weekly": 3,
}

TF_STATUT: Final[Mapping[str, Mapping[str, int]]] = {
    "H1":     {"Fresh": 4, "Aged": 12},
    "H4":     {"Fresh": 3, "Aged": 8},
    "D1":     {"Fresh": 2, "Aged": 5},
    "Weekly": {"Fresh": 2, "Aged": 4},
}

ATR_DIST_MULT: Final[float] = 1.8
MIN_SCORE: Final[int] = 65
SCAN_GLOBAL_TIMEOUT: Final[int] = 180
SCAN_MAX_WORKERS: Final[int] = 6
OANDA_REQUEST_TIMEOUT: Final[int] = 12
CANDLES_CACHE_TTL_SECONDS: Final[int] = 60  # short — quotes refresh quickly
MAX_AUTH_FAILURES: Final[int] = 3
OANDA_MAX_RETRIES: Final[int] = 2
OANDA_BACKOFF_BASE: Final[float] = 0.25

DISPLAY_COLS: Final[tuple[str, ...]] = (
    "Instrument", "Timeframe", "Type", "Ordre", "Signal",
    "Niveau", "Distance%", "Volatilité", "Force", "BB_Width",
    "Statut", "Heure (UTC)",
)
EXPORT_COLS: Final[tuple[str, ...]] = DISPLAY_COLS

TrendT = Literal["Bullish", "Bearish", "Range"]
DirectionT = Literal["Bullish", "Bearish"]
SigTypeT = Literal["CHoCH", "BOS"]
StatusT = Literal["Fresh", "Aged", "Stale", "N/A"]
SessionT = Literal[
    "London_NY_Overlap", "London", "NewYork", "Tokyo", "Off",
]


# =====================================================================
# SECTION 2 — STRUCTURED JSON LOGGING
# =====================================================================

class _JsonFormatter(logging.Formatter):
    """Minimal dependency-free JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
                .isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any structured extras
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                payload[key[4:]] = value
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            # Never let logging crash the process
            return json.dumps({"ts": payload["ts"], "level": "ERROR",
                               "msg": "log_serialization_failure"})


def _configure_root_logger() -> logging.Logger:
    root = logging.getLogger("choch")
    if getattr(root, "_choch_configured", False):
        return root
    root.setLevel(os.environ.get("CHOCH_LOG_LEVEL", "INFO"))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False
    setattr(root, "_choch_configured", True)
    return root


logger = _configure_root_logger()


def _log(level: int, msg: str, **ctx: Any) -> None:
    """Structured log helper — keys are flattened in JSON output."""
    logger.log(level, msg, extra={f"ctx_{k}": v for k, v in ctx.items()})


# =====================================================================
# SECTION 3 — CACHED RESOURCES (Streamlit-aware)
# =====================================================================

@lru_cache(maxsize=8)
def _tz(name: str) -> ZoneInfo:
    """Cached ZoneInfo factory — TZ objects are heavy to instantiate."""
    return ZoneInfo(name)


@st.cache_resource(show_spinner=False)
def _get_oanda_api() -> API:
    """
    One OANDA API client per Streamlit process (shared across reruns).
    Thread-safe by design (oandapyV20 uses requests.Session internally).
    """
    token = st.secrets.get("OANDA_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("OANDA_ACCESS_TOKEN missing from st.secrets")
    return API(
        access_token=token,
        request_params={"timeout": OANDA_REQUEST_TIMEOUT},
    )


@st.cache_resource(show_spinner=False)
def _get_scan_executor() -> ThreadPoolExecutor:
    """
    Single ThreadPoolExecutor reused across scans (prevents thread storm
    when Streamlit reruns trigger button re-evaluation).
    """
    return ThreadPoolExecutor(
        max_workers=SCAN_MAX_WORKERS,
        thread_name_prefix="choch-scan",
    )


# =====================================================================
# SECTION 4 — PURE DOMAIN LAYER
# =====================================================================

class SwingDict(TypedDict):
    idx: int
    price: float
    kind: Literal["HH", "LH", "HL", "LL"]


@dataclass(frozen=True)
class SignalCore:
    """Single source of truth for a detected signal — used by UI and JSON."""
    sig_type: SigTypeT
    direction: DirectionT
    level: float
    idx_break: int
    close_price: float
    current_price: float
    has_sweep: bool
    atr_val: float
    volatilite: str
    trend: TrendT
    force: Literal["Fort", "Moyen", "Faible"]
    dist_atr: float
    score: int
    bb_width_pct: Optional[float]
    bb_regime: str
    signal_time_utc: datetime
    session: SessionT
    statut: StatusT
    candles_elapsed: int
    distance_pct: Optional[float]
    current_distance_pct: Optional[float]


# ---- 4.1 numeric utilities ------------------------------------------------

def _safe_float(value: Any) -> Optional[float]:
    """Convert to float or return None for NaN/Inf/invalid."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _compute_true_range_vec(data: pd.DataFrame) -> np.ndarray:
    """
    Vectorised True Range. Resets TR to (high-low) across abnormal gaps
    (gap > 2 × median delta) to avoid spurious TR spikes across weekends.
    """
    high = data["high"].to_numpy(dtype=np.float64)
    low = data["low"].to_numpy(dtype=np.float64)
    close = data["close"].to_numpy(dtype=np.float64)
    n = high.size
    if n < 2:
        return np.empty(0, dtype=np.float64)

    deltas = np.diff(data.index.values.astype("datetime64[s]").astype(np.int64))
    if deltas.size == 0:
        return np.empty(0, dtype=np.float64)
    typical = float(np.median(deltas)) if deltas.size else 3600.0
    typical = max(typical, 1.0)

    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    tr = np.maximum(hl, np.maximum(hc, lc))
    # Across abnormal gaps -> use (high - low) only
    gap_mask = deltas > (2 * typical)
    tr = np.where(gap_mask, hl, tr)
    return tr


def calc_atr_bundle(
    data: pd.DataFrame, inst: str, period: int = 14,
) -> tuple[float, str]:
    """Returns (atr, regime). NaN-safe, fallback to static volatility."""
    tr = _compute_true_range_vec(data)
    fallback = VOLATILITY_STATIC.get(inst, "Moyenne")
    if tr.size < period * 3:
        return float("nan"), fallback

    # EWM-14 converge à 99.9% après 50 périodes — slice suffit
    tr_ewm = tr[-100:] if tr.size > 100 else tr
    atr_val = float(
        pd.Series(tr_ewm).ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    )
    if not math.isfinite(atr_val):
        return float("nan"), fallback

    window = tr[-100:] if tr.size >= 100 else tr
    median_tr = float(np.median(window))
    if not math.isfinite(median_tr) or median_tr < 1e-10:
        return atr_val, fallback

    ratio = atr_val / median_tr
    if ratio >= 1.8:
        regime = "Très Haute"
    elif ratio >= 1.2:
        regime = "Haute"
    elif ratio >= 0.7:
        regime = "Moyenne"
    else:
        regime = "Basse"
    return atr_val, regime


def instrument_precision(inst: str) -> int:
    if any(k in inst for k in (
            "SPX500", "NAS100", "US30", "DE30", "XAU", "XAG")):
        return 2
    if "JPY" in inst:
        return 3
    return 5


def format_niveau(niveau: Optional[float], inst: str) -> str:
    if niveau is None or not math.isfinite(niveau):
        return "N/A"
    return f"{niveau:.{instrument_precision(inst)}f}"


def calc_distance_pct(
    niveau: Optional[float], close_actuel: Optional[float],
) -> Optional[float]:
    if niveau is None or close_actuel is None:
        return None
    if not (math.isfinite(niveau) and math.isfinite(close_actuel)):
        return None
    if abs(niveau) < 1e-12:
        return None
    dist = abs(close_actuel - niveau) / abs(niveau) * 100.0
    return dist if 0.0 <= dist <= 100.0 else None


def format_distance(dist_pct: Optional[float]) -> str:
    return "N/A" if dist_pct is None else f"{dist_pct:.3f}%"


def _local_hour(dt: datetime, tz_name: str) -> int:
    return dt.astimezone(_tz(tz_name)).hour


def get_session(dt: datetime) -> SessionT:
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


# ---- 4.2 candle parsing ---------------------------------------------------

def _parse_candle_row(c: Mapping[str, Any], inst: str, gran: str) -> Optional[dict]:
    """Parse one OANDA candle dict; reject malformed or non-OHLC-consistent."""
    try:
        mid = c["mid"]
        open_v = float(mid["o"])
        high_v = float(mid["h"])
        low_v = float(mid["l"])
        close_v = float(mid["c"])
        t = c["time"]
    except (KeyError, ValueError, TypeError) as exc:
        _log(logging.WARNING, "candle_malformed",
             instrument=inst, granularity=gran, err=str(exc))
        return None

    if not all(math.isfinite(v) for v in (open_v, high_v, low_v, close_v)):
        return None
    # Canonical OHLC consistency: high >= max(o,c) >= min(o,c) >= low
    if not (high_v >= low_v
            and high_v >= max(open_v, close_v)
            and low_v <= min(open_v, close_v)):
        _log(logging.WARNING, "candle_inconsistent",
             instrument=inst, granularity=gran, t=str(t))
        return None

    return {
        "time": pd.to_datetime(t, utc=True),
        "open": open_v,
        "high": high_v,
        "low": low_v,
        "close": close_v,
    }


# ---- 4.3 swing & trend ----------------------------------------------------

def _classify_swings(pivots: Sequence[tuple[int, float, str]]) -> list[SwingDict]:
    swings: list[SwingDict] = []
    prev_h: Optional[float] = None
    prev_l: Optional[float] = None
    for idx, price, k in pivots:
        if k == "H":
            kind: Literal["HH", "LH", "HL", "LL"] = (
                "HH" if (prev_h is None or price > prev_h) else "LH"
            )
            swings.append({"idx": idx, "price": price, "kind": kind})
            prev_h = price
        else:
            kind = "HL" if (prev_l is None or price > prev_l) else "LL"
            swings.append({"idx": idx, "price": price, "kind": kind})
            prev_l = price
    return swings


def detect_swing_points(data: pd.DataFrame, tf: str) -> list[SwingDict]:
    """
    Pivot detection over a centered window. Dedup is performed by **index**
    (never by price) to avoid losing legitimate same-price pivots on
    indices/metals.
    """
    lookback = SWING_LOOKBACK.get(tf, 5)
    history = SWING_HISTORY.get(tf, 60)
    n = len(data)
    if n < 2 * lookback + 1:
        return []

    win = 2 * lookback + 1
    high_s = data["high"].reset_index(drop=True)
    low_s = data["low"].reset_index(drop=True)
    roll_max = high_s.rolling(window=win, center=True, min_periods=win).max()
    roll_min = low_s.rolling(window=win, center=True, min_periods=win).min()
    h_mask = (high_s == roll_max) & high_s.notna()
    l_mask = (low_s == roll_min) & low_s.notna()

    start = max(lookback, n - history - lookback)
    end = n - lookback - 1
    pivots: list[tuple[int, float, str]] = []
    seen_idx: set[int] = set()
    high_arr = high_s.to_numpy()
    low_arr = low_s.to_numpy()
    for i in range(start, end):
        if h_mask.iloc[i] and i not in seen_idx:
            pivots.append((i, float(high_arr[i]), "H"))
            seen_idx.add(i)
        if l_mask.iloc[i] and i not in seen_idx:
            # An index that is simultaneously a max AND a min in the same
            # window is a degenerate flat — emit high only, skip low.
            pivots.append((i, float(low_arr[i]), "L"))
            seen_idx.add(i)
    pivots.sort(key=lambda p: p[0])
    return _classify_swings(pivots)


def _last_high_low(
    swings: Sequence[SwingDict],
) -> tuple[Optional[str], Optional[str]]:
    highs = [s for s in swings if s["kind"] in ("HH", "LH")]
    lows = [s for s in swings if s["kind"] in ("HL", "LL")]
    if not highs or not lows:
        return None, None
    return highs[-1]["kind"], lows[-1]["kind"]


def get_structural_trend(swings: Sequence[SwingDict]) -> TrendT:
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


# ---- 4.4 signal resolution ------------------------------------------------

_SigResult = tuple[
    Optional[SigTypeT], Optional[DirectionT], Optional[float],
]
_NONE_SIG: _SigResult = (None, None, None)


def _resolve_bullish(
    close_arr: np.ndarray, idx: int, prev_swings: Sequence[SwingDict],
) -> _SigResult:
    # r7: revert BUG-E over-restriction. The protective pivot for a bearish
    # CHoCH is the most recent HL, regardless of its position relative to the
    # last HH. Requiring HL.idx > last_HH.idx eliminated the dominant right-
    # edge configuration where the protective pivot forms near the end of the
    # series and hasn't yet been followed by a new HH. This is the proven r3
    # logic, restored. BUG-A (BOS on last extreme) is preserved.
    hl = [s for s in prev_swings if s["kind"] == "HL"]
    if hl:
        ref = hl[-1]["price"]
        if close_arr[idx] < ref <= close_arr[idx - 1]:
            return "CHoCH", "Bearish", ref
    hh = [s for s in prev_swings if s["kind"] == "HH"]
    if hh:
        ref = hh[-1]["price"]
        if close_arr[idx - 1] <= ref < close_arr[idx]:
            return "BOS", "Bullish", ref
    return _NONE_SIG


def _resolve_bearish(
    close_arr: np.ndarray, idx: int, prev_swings: Sequence[SwingDict],
) -> _SigResult:
    # r7: symmetric revert of BUG-E (see _resolve_bullish comment).
    lh = [s for s in prev_swings if s["kind"] == "LH"]
    if lh:
        ref = lh[-1]["price"]
        if close_arr[idx - 1] <= ref < close_arr[idx]:
            return "CHoCH", "Bullish", ref
    ll = [s for s in prev_swings if s["kind"] == "LL"]
    if ll:
        ref = ll[-1]["price"]
        if close_arr[idx] < ref <= close_arr[idx - 1]:
            return "BOS", "Bearish", ref
    return _NONE_SIG


def _resolve_signal(
    trend: TrendT, close_arr: np.ndarray, idx: int,
    prev_swings: Sequence[SwingDict],
) -> _SigResult:
    if trend == "Bullish":
        return _resolve_bullish(close_arr, idx, prev_swings)
    if trend == "Bearish":
        return _resolve_bearish(close_arr, idx, prev_swings)
    return _NONE_SIG


def _detect_liquidity_sweep(
    high_arr: np.ndarray, low_arr: np.ndarray, idx: int,
    prev_swings: Sequence[SwingDict], atr_val: float, direction: DirectionT,
) -> bool:
    # BUG-G : seuls les 3 pivots les plus récents constituent des pools actifs.
    # any() sur toute l'histoire gonflait has_sweep → scores artificiels.
    if direction == "Bearish":
        cands = [s for s in prev_swings if s["kind"] in ("HH", "LH")]
        if not cands:
            return False
        return any(
            high_arr[idx] > s["price"]
            and (high_arr[idx] - s["price"]) > (atr_val * 0.25)
            for s in cands[-3:]
        )
    cands = [s for s in prev_swings if s["kind"] in ("HL", "LL")]
    if not cands:
        return False
    return any(
        low_arr[idx] < s["price"]
        and (s["price"] - low_arr[idx]) > (atr_val * 0.25)
        for s in cands[-3:]
    )


def compute_statut(idx_sig: Optional[int], len_df: int, tf: str) -> StatusT:
    if idx_sig is None:
        return "N/A"
    candles_elapsed = (len_df - 1) - idx_sig
    thr = TF_STATUT.get(tf, {"Fresh": 2, "Aged": 5})
    if candles_elapsed <= thr["Fresh"]:
        return "Fresh"
    if candles_elapsed <= thr["Aged"]:
        return "Aged"
    return "Stale"


def _compute_confluence_score(
    dist_atr: float, candle_time: datetime, has_sweep: bool,
    sig_type: SigTypeT, statut: StatusT,
) -> int:
    score = 25
    if dist_atr <= 1.0:
        score += 15
    # r7: revert BUG-F. The session bonus reflects the quality of the candle's
    # formation context, which does not change as the signal ages. Coupling it
    # to statut == "Fresh" pushed valid Aged signals below MIN_SCORE, amplifying
    # the zero-signal regression. 'statut' is kept in the signature for backward
    # stability (callers need not change), but is no longer used in scoring.
    if is_premium_session(get_session(candle_time)):
        score += 20
    if has_sweep:
        score += 15
    if sig_type == "CHoCH":
        score += 10
    return min(score, 100)


# ---- 4.5 BB width ---------------------------------------------------------

def compute_bb_width(
    data: pd.DataFrame, length: int = 20, std: int = 2,
) -> tuple[Optional[float], str]:
    close = data["close"]
    if len(close) < length * 2:
        return None, "N/A"
    sma = close.rolling(length).mean()
    std_dev = close.rolling(length).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev

    sma_safe = sma.where(sma.abs() > 1e-12)
    bb_w = (upper - lower) / sma_safe
    bb_avg = bb_w.rolling(length).mean()
    avg_last = bb_avg.iloc[-1]
    if pd.isna(avg_last) or abs(avg_last) < 1e-12:
        return None, "N/A"

    bb_avg_safe = bb_avg.where(bb_avg.abs() > 1e-12)
    pct_series = (bb_w - bb_avg) / bb_avg_safe * 100.0
    pct_val = pct_series.iloc[-1]
    if pd.isna(pct_val) or not math.isfinite(pct_val):
        return None, "N/A"

    if pct_val <= -25:
        regime = "Squeeze"
    elif pct_val >= 25:
        regime = "Expansion"
    else:
        regime = "Normal"
    return float(pct_val), regime


def format_bb_width(bb_result: tuple[Optional[float], str]) -> str:
    pct, regime = bb_result
    if pct is None:
        return "N/A"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%_{regime}"


# ---- 4.6 SignalCore builder (the single source of truth) -----------------

def _evaluate_candle(
    *, idx: int, df: pd.DataFrame, prev_swings: Sequence[SwingDict],
    atr_val: float, atr_regime: str, trend: TrendT, tf: str,
) -> Optional[SignalCore]:
    close_arr = df["close"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    open_arr = df["open"].to_numpy()
    n = close_arr.size

    sig_type, direction, level = _resolve_signal(
        trend, close_arr, idx, prev_swings,
    )
    if sig_type is None or direction is None or level is None:
        return None

    rng_v = high_arr[idx] - low_arr[idx]
    if rng_v <= 0:
        return None
    body_ratio = abs(close_arr[idx] - open_arr[idx]) / rng_v
    if body_ratio < 0.40:
        return None
    force_label: Literal["Fort", "Moyen", "Faible"] = (
        "Fort" if body_ratio >= 0.60 else "Moyen"
    )

    has_sweep = sig_type == "CHoCH" and _detect_liquidity_sweep(
        high_arr, low_arr, idx, prev_swings, atr_val, direction,
    )
    dist_atr = abs(close_arr[idx] - level) / atr_val
    if dist_atr > ATR_DIST_MULT:
        return None

    candle_time = df.index[idx].to_pydatetime()
    statut = compute_statut(idx, n, tf)
    score = _compute_confluence_score(
        dist_atr, candle_time, has_sweep, sig_type, statut,
    )
    if score < MIN_SCORE:
        return None

    # BB width over a bounded window ending at signal candle
    bb_window = df.iloc[max(0, idx + 1 - 40): idx + 1]
    bb_pct, bb_regime = compute_bb_width(bb_window)

    level_f = float(level)
    close_price = float(close_arr[idx])
    current_price = float(close_arr[-1])

    return SignalCore(
        sig_type=sig_type,
        direction=direction,
        level=level_f,
        idx_break=int(idx),
        close_price=close_price,
        current_price=current_price,
        has_sweep=bool(has_sweep),
        atr_val=float(atr_val),
        volatilite=atr_regime,
        trend=trend,
        force=force_label,
        dist_atr=float(dist_atr),
        score=int(score),
        bb_width_pct=bb_pct,
        bb_regime=bb_regime,
        signal_time_utc=candle_time,
        session=get_session(candle_time),
        statut=statut,
        candles_elapsed=(n - 1) - int(idx),
        distance_pct=calc_distance_pct(level_f, close_price),
        current_distance_pct=calc_distance_pct(level_f, current_price),
    )


def detect_choch(df: pd.DataFrame, tf: str, inst: str) -> Optional[SignalCore]:
    swings = detect_swing_points(df, tf)
    trend = get_structural_trend(swings)
    if trend == "Range":
        return None
    atr_val, atr_regime = calc_atr_bundle(df, inst)
    if not math.isfinite(atr_val) or atr_val <= 0:
        return None

    # r7: restore explicit detection window. BUG-B (r6) was based on the false
    # premise that _scan_one calls detect_choch() per-candle. It does not.
    # _scan_one calls detect_choch() exactly ONCE per (inst, tf). The loop is
    # therefore the ONLY mechanism covering signals formed on N-2..N-k.
    # The first (most recent) match wins, preserving signal_id determinism.
    n = len(df)
    lookback = SWING_LOOKBACK.get(tf, 5)
    window = DETECTION_LOOKBACK.get(tf, 5)

    for offset in range(window):
        idx = n - 1 - offset
        if idx < 3:
            break
        prev_swings = [s for s in swings if s["idx"] <= idx - (lookback + 1)]
        if not prev_swings:
            continue
        sig = _evaluate_candle(
            idx=idx, df=df, prev_swings=prev_swings,
            atr_val=atr_val, atr_regime=atr_regime, trend=trend, tf=tf,
        )
        if sig is not None:
            return sig
    return None


# ---- 4.7 row/payload projections (single source of truth) ----------------

def _signal_id(inst: str, tf: str, sig: SignalCore) -> str:
    """Deterministic id: same candle close + same rule => same id."""
    raw = (
        f"{inst}|{tf}|{sig.signal_time_utc.strftime('%Y%m%dT%H%MZ')}"
        f"|{sig.sig_type}|{sig.direction}|{RULE_VERSION}"
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{inst}__{tf}__{sig.signal_time_utc.strftime('%Y%m%dT%H%M')}__{digest}"


def signal_to_row(inst: str, tf: str, sig: SignalCore) -> dict[str, Any]:
    inst_disp = inst.replace("_", "/")
    return {
        "Instrument": inst_disp,
        "Paire": inst_disp,
        "_time_sort": sig.signal_time_utc,
        "Timeframe": tf,
        "Type": sig.sig_type,
        "Ordre": "Achat" if sig.direction == "Bullish" else "Vente",
        "Signal": f"{sig.direction} {sig.sig_type}",
        "Niveau": format_niveau(sig.level, inst),
        "Distance%": format_distance(sig.distance_pct),
        "Volatilité": sig.volatilite,
        "Force": sig.force,
        "BB_Width": format_bb_width((sig.bb_width_pct, sig.bb_regime)),
        "Statut": sig.statut,
        "Heure (UTC)": sig.signal_time_utc.strftime("%Y-%m-%d %H:%M"),
        "signal_id": _signal_id(inst, tf, sig),
    }


def signal_to_payload(
    inst: str, tf: str, sig: SignalCore, scan_time: datetime,
) -> dict[str, Any]:
    """JSON pipeline payload — bit-stable, sortable, rule-versioned."""
    inst_disp = inst.replace("_", "/")
    prec = instrument_precision(inst)
    return {
        "signal_id": _signal_id(inst, tf, sig),
        "scanner_version": SCANNER_VERSION,
        "rule_version": RULE_VERSION,
        "generated_at": scan_time.isoformat(),
        "pair": inst_disp,
        "pair_oanda": inst,
        "timeframe": tf,
        "type": sig.sig_type,
        "direction": sig.direction,
        "is_bullish": sig.direction == "Bullish",
        "order": "buy" if sig.direction == "Bullish" else "sell",
        "trend": sig.trend,
        "is_choch": sig.sig_type == "CHoCH",
        "status": sig.statut,
        "confluence_score": int(sig.score),
        "level": round(sig.level, prec),
        "close_price": round(sig.close_price, prec),
        "current_price": round(sig.current_price, prec),
        "distance_pct": (
            round(sig.distance_pct, 4) if sig.distance_pct is not None else None
        ),
        "current_distance_pct": (
            round(sig.current_distance_pct, 4)
            if sig.current_distance_pct is not None else None
        ),
        "distance_atr_multiple": round(sig.dist_atr, 2),
        "volatility": sig.volatilite,
        "force": sig.force,
        "bb_width_pct": (
            round(sig.bb_width_pct, 2)
            if sig.bb_width_pct is not None else None
        ),
        "bb_regime": sig.bb_regime,
        "session": sig.session,
        "signal_time": sig.signal_time_utc.isoformat(),
        "candles_elapsed": int(sig.candles_elapsed),
        "has_sweep": bool(sig.has_sweep),
        "atr": round(sig.atr_val, prec + 2),
    }


# =====================================================================
# SECTION 5 — I/O LAYER (OANDA)
# =====================================================================

@dataclass
class AuthState:
    """Per-session auth failure tracker (NOT module-level)."""
    failures: int = 0
    aborted: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_failure(self) -> int:
        with self.lock:
            self.failures += 1
            if self.failures >= MAX_AUTH_FAILURES:
                self.aborted = True
            return self.failures

    def is_aborted(self) -> bool:
        with self.lock:
            return self.aborted

    def reset(self) -> None:
        with self.lock:
            self.failures = 0
            self.aborted = False


def _fetch_candles_raw(inst: str, gran: str) -> Optional[list[dict]]:
    """Single OANDA REST call. Returns raw `complete` candles list, or None."""
    count = GRAN_COUNT.get(gran, 300)
    req = instruments.InstrumentsCandles(
        instrument=inst,
        params={"count": count, "granularity": gran, "price": "M"},
    )
    api = _get_oanda_api()
    api.request(req)
    return [c for c in req.response.get("candles", []) if c.get("complete")]


@st.cache_data(
    ttl=CANDLES_CACHE_TTL_SECONDS,
    show_spinner=False,
    max_entries=512,
)
def get_candles_cached(
    inst: str, gran: str, cache_bust: int,
) -> Optional[pd.DataFrame]:
    """
    Streamlit-cached candles fetch with explicit cache_bust key.
    Cache TTL is short (60s) — quotes refresh quickly during market hours.

    Errors are NOT cached: on any failure we return None and the next call
    will retry (st.cache_data caches the None too, but TTL is short).
    For auth/network errors we raise so the caller can track per-session.
    """
    for attempt in range(OANDA_MAX_RETRIES + 1):
        try:
            raw = _fetch_candles_raw(inst, gran)
            break
        except V20Error as exc:
            if exc.code == 401:
                # Bubble up so the caller can update AuthState
                raise
            if exc.code == 429 and attempt < OANDA_MAX_RETRIES:
                time.sleep(OANDA_BACKOFF_BASE * (2 ** attempt))
                continue
            _log(logging.WARNING, "oanda_v20_error",
                 instrument=inst, granularity=gran, code=exc.code,
                 err=str(exc))
            return None
        except requests.RequestException as exc:
            if attempt < OANDA_MAX_RETRIES:
                time.sleep(OANDA_BACKOFF_BASE * (2 ** attempt))
                continue
            _log(logging.WARNING, "oanda_network_error",
                 instrument=inst, granularity=gran, err=str(exc))
            return None
    else:
        return None  # pragma: no cover - unreachable

    if raw is None or len(raw) < 50:
        return None

    rows = [
        r for c in raw
        if (r := _parse_candle_row(c, inst, gran)) is not None
    ]
    if len(rows) < 50:
        return None

    df = pd.DataFrame(rows).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


# =====================================================================
# SECTION 6 — ORCHESTRATION
# =====================================================================

@dataclass
class ScanResult:
    rows: list[dict[str, Any]]
    payloads: list[dict[str, Any]]
    errors: list[str]
    timed_out: int
    scan_time: datetime
    aborted: bool = False


def _scan_one(
    inst: str, tf_name: str, tf_code: str, cache_bust: int,
    auth: AuthState,
) -> tuple[str, str, Optional[SignalCore], Optional[str]]:
    """Worker — fetches candles and runs detection. Returns errors as strings."""
    if auth.is_aborted():
        return inst, tf_name, None, "aborted"
    try:
        df = get_candles_cached(inst, tf_code, cache_bust)
    except V20Error as exc:
        if exc.code == 401:
            n = auth.record_failure()
            _log(logging.ERROR, "oanda_auth_failure",
                 instrument=inst, granularity=tf_name, count=n)
            return inst, tf_name, None, f"401#{n}"
        return inst, tf_name, None, f"v20:{exc.code}"
    except Exception as exc:  # noqa: BLE001 — defensive boundary
        _log(logging.ERROR, "scan_one_unexpected",
             instrument=inst, granularity=tf_name, err=str(exc))
        return inst, tf_name, None, f"unexpected:{type(exc).__name__}"

    if df is None:
        return inst, tf_name, None, None
    try:
        sig = detect_choch(df, tf_name, inst)
    except Exception as exc:  # noqa: BLE001
        _log(logging.ERROR, "detect_choch_failed",
             instrument=inst, granularity=tf_name, err=str(exc))
        return inst, tf_name, None, f"detect:{type(exc).__name__}"
    return inst, tf_name, sig, None


def run_scan(
    auth: AuthState, cache_bust: int,
    progress_callback: Optional[callable] = None,
) -> ScanResult:
    """
    Pure orchestration. Idempotent: same inputs (cache_bust unchanged)
    produce identical outputs thanks to st.cache_data on candles.
    """
    correlation_id = uuid.uuid4().hex[:12]
    scan_time = datetime.now(timezone.utc)
    auth.reset()
    _log(logging.INFO, "scan_start",
         correlation_id=correlation_id,
         instruments=len(INSTRUMENTS), timeframes=len(TIMEFRAMES))

    executor = _get_scan_executor()
    futures: dict[Future, tuple[str, str]] = {
        executor.submit(_scan_one, inst, tf_name, tf_code, cache_bust, auth):
            (inst, tf_name)
        for inst in INSTRUMENTS
        for tf_name, tf_code in TIMEFRAMES.items()
    }

    done, not_done = wait(futures.keys(), timeout=SCAN_GLOBAL_TIMEOUT)
    for f in not_done:
        f.cancel()

    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for fut in done:
        inst, tf_name = futures[fut]
        try:
            inst_r, tf_r, sig, err = fut.result()
        except CancelledError:
            continue
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{inst}/{tf_name}: {exc}")
            continue
        if err and err.startswith("401"):
            errors.append(f"{inst_r}/{tf_r}: auth {err}")
            continue
        if err:
            errors.append(f"{inst_r}/{tf_r}: {err}")
            continue
        if sig is None:
            if progress_callback is not None:
                progress_callback(inst_r, tf_r)
            continue
        row = signal_to_row(inst_r, tf_r, sig)
        sid = row["signal_id"]
        if sid in seen_ids:
            if progress_callback is not None:
                progress_callback(inst_r, tf_r)
            continue
        seen_ids.add(sid)
        rows.append(row)
        if sig.statut in ("Fresh", "Aged"):
            payloads.append(signal_to_payload(inst_r, tf_r, sig, scan_time))
        if progress_callback is not None:
            progress_callback(inst_r, tf_r)

    aborted = auth.is_aborted()
    _log(logging.INFO, "scan_end",
         correlation_id=correlation_id,
         signals=len(rows), pipeline=len(payloads),
         errors=len(errors), timed_out=len(not_done), aborted=aborted)
    return ScanResult(
        rows=rows, payloads=payloads, errors=errors,
        timed_out=len(not_done), scan_time=scan_time, aborted=aborted,
    )


# =====================================================================
# SECTION 7 — EXPORT (PDF / PNG / JSON)
# =====================================================================

def _json_default(obj: Any) -> Any:
    """Robust JSON converter — no information loss, never crashes."""
    if obj is None:
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    try:
        return str(obj)
    except Exception:  # noqa: BLE001 - last-resort barrier
        return "<unserializable>"


def serialize_pipeline(
    payloads: Sequence[Mapping[str, Any]], scan_time: datetime,
) -> bytes:
    doc = {
        "meta": {
            "scanner_version": SCANNER_VERSION,
            "rule_version": RULE_VERSION,
            "generated_at": scan_time.isoformat(),
            "signal_count": len(payloads),
        },
        "signals": list(payloads),
    }
    return json.dumps(
        doc, ensure_ascii=False, indent=2,
        default=_json_default, allow_nan=False,
    ).encode("utf-8")


def create_pdf(df_export: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        leftMargin=20, rightMargin=20, topMargin=40, bottomMargin=40,
    )
    elements: list[Any] = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph(
        f"Rapport des Signaux CHoCH v{SCANNER_VERSION} ({RULE_VERSION})",
        styles["Title"]))
    elements.append(Paragraph(
        f"Généré le {datetime.now(timezone.utc).strftime('%d/%m/%Y à %H:%M')} UTC",
        styles["Normal"]))
    elements.append(Spacer(1, 20))

    cols_present = [c for c in EXPORT_COLS if c in df_export.columns]
    widths_map = {c: 60 for c in cols_present}
    widths_map.update({
        "Instrument": 65, "Distance%": 52, "Statut": 45, "Heure (UTC)": 105,
    })
    col_widths = [widths_map.get(c, 60) for c in cols_present]
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
    return buffer.getvalue()


def generate_png(data: pd.DataFrame, display_cols: Sequence[str]) -> bytes:
    fig = Figure(figsize=(22, min(max(5, len(data) * 0.35), 30)))
    ax = fig.add_subplot(111)
    ax.axis("off")
    cols = [c for c in display_cols if c in data.columns]
    disp = data[cols]
    tbl = ax.table(
        cellText=disp.values, colLabels=disp.columns,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    return buf.getvalue()


# =====================================================================
# SECTION 8 — UI LAYER
# =====================================================================

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
    except (ValueError, TypeError):
        return "color:#90a4ae"
    if v <= 0.15:
        return "color:#00c853;font-weight:bold"
    if v <= 0.40:
        return "color:#ff9800;font-weight:bold"
    return "color:#ff5252;font-weight:bold"


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "scanning": False,
        "cache_bust": 0,
        "auth": AuthState(),
        "df": None,
        "pipeline_signals": [],
        "scan_time": None,
        "scan_errors": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _render_downloads(
    df_all: pd.DataFrame, df_export: pd.DataFrame,
    pipeline_signals: Sequence[Mapping[str, Any]], scan_time: datetime,
) -> None:
    ts = scan_time.strftime("%Y%m%d_%H%M")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cols = [c for c in EXPORT_COLS if c in df_export.columns]
        csv_bytes = df_export[cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "CSV", csv_bytes, f"choch_{ts}.csv", "text/csv",
            key=f"dl_csv_{ts}",
        )
    with c2:
        # Lazy generation, cached against the dataframe identity via session
        png_key = f"png_{ts}"
        if png_key not in st.session_state:
            st.session_state[png_key] = generate_png(df_all, DISPLAY_COLS)
        st.download_button(
            "PNG", st.session_state[png_key],
            f"choch_{ts}.png", "image/png",
            key=f"dl_png_{ts}",
        )
    with c3:
        pdf_key = f"pdf_{ts}"
        if pdf_key not in st.session_state:
            st.session_state[pdf_key] = create_pdf(df_export)
        st.download_button(
            "PDF", st.session_state[pdf_key],
            f"choch_signaux_{ts}.pdf", "application/pdf",
            key=f"dl_pdf_{ts}",
        )
    with c4:
        st.download_button(
            "JSON", serialize_pipeline(pipeline_signals, scan_time),
            f"choch_pipeline_{ts}.json", "application/json",
            key=f"dl_json_{ts}",
        )


def _render_dataframe(df_all: pd.DataFrame) -> None:
    cols_disp = [c for c in DISPLAY_COLS if c in df_all.columns]
    styled = (
        df_all[cols_disp].style
        .map(lambda x: ("color:#e879f9;font-weight:bold" if x == "CHoCH"
                        else "color:#94a3b8" if x == "BOS" else ""),
             subset=["Type"])
        .map(lambda x: ("color:#00c853;font-weight:bold" if x == "Achat"
                        else "color:#ff5252;font-weight:bold" if x == "Vente"
                        else ""), subset=["Ordre"])
        .map(lambda x: ("color:#00c853" if "Bull" in str(x)
                        else "color:#ff5252" if "Bear" in str(x) else ""),
             subset=["Signal"])
        .map(lambda x: ("color:#00c853;font-weight:bold" if x == "Fort"
                        else "color:#ff5252" if x == "Faible"
                        else "color:#ff9800"), subset=["Force"])
        .map(_style_bb, subset=["BB_Width"])
        .map(_style_distance, subset=["Distance%"])
        .map(lambda x: ("color:#00c853;font-weight:bold" if x == "Fresh"
                        else "color:#ff9800;font-weight:bold" if x == "Aged"
                        else "color:#ff5252;font-weight:bold" if x == "Stale"
                        else ""), subset=["Statut"])
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)


def _render_results() -> None:
    df_all = st.session_state.df
    if df_all is None or df_all.empty:
        return
    df_export = df_all[df_all["Statut"].isin(["Fresh", "Aged"])].copy()
    pipeline_signals = st.session_state.get("pipeline_signals", [])
    scan_time = st.session_state.get("scan_time") or datetime.now(timezone.utc)

    n_stale = int((df_all["Statut"] == "Stale").sum())
    if n_stale > 0:
        st.info(
            f"{n_stale} signal(s) Stale visible(s) dans le tableau "
            "— exclus des exports."
        )
    _render_downloads(df_all, df_export, pipeline_signals, scan_time)
    _render_dataframe(df_all)

    if pipeline_signals:
        with st.expander(
            f"Aperçu JSON Pipeline ({len(pipeline_signals)} signaux Fresh/Aged)"
        ):
            st.json(pipeline_signals[0])


def _trigger_scan() -> None:
    """Single-entry guarded scan trigger (avoids re-entry on Streamlit reruns)."""
    if st.session_state.scanning:
        return
    st.session_state.scanning = True
    try:
        # Secrets check
        if "OANDA_ACCESS_TOKEN" not in st.secrets:
            st.error("Clé API OANDA manquante dans les secrets.")
            return

        auth: AuthState = st.session_state.auth
        cache_bust: int = st.session_state.cache_bust

        # r7: animated progress bar replacing static st.spinner
        _pb = st.progress(0.0, text="Initialisation du scan…")
        _st = st.empty()
        _t0 = time.monotonic()
        _completed = 0
        _total = len(INSTRUMENTS) * len(TIMEFRAMES)

        def _tick(inst: str, tf: str) -> None:
            nonlocal _completed
            _completed += 1
            pct = _completed / _total
            elapsed = time.monotonic() - _t0
            eta = (elapsed / _completed) * (_total - _completed) if _completed else 0.0
            _pb.progress(
                pct,
                text=f"[{_completed}/{_total}] {inst.replace('_', '/')} ({tf}) — ETA {int(eta)}s"
            )
            _st.markdown(
                f"<div style='font-size:0.8rem;color:#64748b'>"
                f"Workers: {SCAN_MAX_WORKERS} | Timeout: {SCAN_GLOBAL_TIMEOUT}s | "
                f"Écoulé: {elapsed:.1f}s</div>",
                unsafe_allow_html=True,
            )

        try:
            result = run_scan(auth, cache_bust, progress_callback=_tick)
        except Exception as exc:  # noqa: BLE001 — final defensive barrier
            _log(logging.ERROR, "scan_fatal", err=str(exc))
            _pb.empty()
            _st.empty()
            st.error(f"Erreur critique du scan : {exc}")
            return
        finally:
            _pb.empty()
            _st.empty()

        if result.aborted:
            st.error(
                "Scan interrompu — trop d'erreurs d'authentification OANDA. "
                "Vérifiez le token."
            )
            return

        if result.timed_out:
            st.warning(
                f"Timeout global — {result.timed_out} requête(s) ignorée(s), "
                "résultats partiels."
            )
        if result.errors:
            st.warning(
                f"{len(result.errors)} erreur(s) : "
                f"{'; '.join(result.errors[:5])}"
            )

        if not result.rows:
            st.session_state.df = None
            st.session_state.pipeline_signals = []
            st.session_state.scan_time = result.scan_time
            st.info("Aucun signal CHoCH/BOS récent qualifié (Score ≥ 65).")
            return

        df = (
            pd.DataFrame(result.rows)
            .sort_values("_time_sort", ascending=False)
            .drop_duplicates(subset="signal_id", keep="first")
            .drop(columns=["_time_sort", "signal_id"])
            .reset_index(drop=True)
        )
        st.session_state.df = df
        st.session_state.pipeline_signals = result.payloads
        st.session_state.scan_time = result.scan_time
        st.session_state.scan_errors = result.errors
        st.success(
            f"Scan terminé — {len(df)} signaux | "
            f"{len(result.payloads)} dans le pipeline JSON | "
            f"rule={RULE_VERSION}"
        )
    finally:
        st.session_state.scanning = False


# =====================================================================
# SECTION 9 — STREAMLIT ENTRY POINT
# =====================================================================

st.set_page_config(
    page_title=f"CHoCH Scanner v{SCANNER_VERSION}",
    layout="wide",
)
st.title(
    f"Scanner Change of Character (CHoCH) — v{SCANNER_VERSION} "
    f"({RULE_VERSION})"
)

_init_session_state()

col_a, col_b = st.columns([3, 1])
with col_a:
    scan_clicked = st.button(
        "Lancer le Scan", type="primary",
        use_container_width=True,
        disabled=st.session_state.scanning,
        key="btn_scan",
    )
with col_b:
    force_refresh = st.button(
        "Force refresh (bust cache)",
        use_container_width=True,
        disabled=st.session_state.scanning,
        key="btn_force_refresh",
    )

if force_refresh:
    st.session_state.cache_bust += 1
    get_candles_cached.clear()
    st.toast("Cache des bougies vidé.", icon="🔄")

if scan_clicked:
    _trigger_scan()

if st.session_state.df is not None:
    _render_results()
