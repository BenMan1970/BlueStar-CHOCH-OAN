"""
CHoCH Scanner v5.9.1 — Change of Character & Break of Structure detector.

Surgical refactor of v5.9:
- Streamlit caching: API instance (@st.cache_resource), candles & exports (@st.cache_data)
- Vectorized swing detection (rolling max/min)
- Robust thread pool shutdown (cancel_futures=True)
- Fine-grained exception classification
- Strict typing, dead code removal
- Secrets-masking log filter
"""
from __future__ import annotations

import io
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure  # noqa: E402

import numpy as np
import pandas as pd
import streamlit as st
from oandapyV20 import API
from oandapyV20.endpoints import instruments
from oandapyV20.exceptions import V20Error
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

# ===================== LOGGING =====================
class _SecretsFilter(logging.Filter):
    """Mask common secret patterns in log messages (defense in depth)."""
    _PATTERNS = ("Bearer ", "access_token=")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for p in self._PATTERNS:
            if p in msg:
                record.msg = msg.replace(p, p[:6] + "***")
                record.args = ()
        return True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.addFilter(_SecretsFilter())

SCANNER_VERSION = "5.9.1"

# ===================== CONFIG =====================
INSTRUMENTS: list[str] = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "DE30_EUR", "XAU_USD", "SPX500_USD", "NAS100_USD", "US30_USD",
]

VOLATILITY_STATIC: dict[str, str] = {
    "EUR_USD": "Basse", "GBP_USD": "Basse", "USD_JPY": "Basse", "USD_CHF": "Basse",
    "USD_CAD": "Basse", "AUD_USD": "Moyenne", "NZD_USD": "Moyenne",
    "EUR_GBP": "Moyenne", "EUR_JPY": "Moyenne", "EUR_CHF": "Moyenne",
    "EUR_AUD": "Moyenne", "EUR_CAD": "Moyenne", "EUR_NZD": "Moyenne",
    "GBP_JPY": "Haute", "GBP_CHF": "Haute", "GBP_AUD": "Haute",
    "GBP_CAD": "Haute", "GBP_NZD": "Haute",
    "AUD_JPY": "Haute", "AUD_CAD": "Moyenne", "AUD_CHF": "Haute",
    "AUD_NZD": "Moyenne", "CAD_JPY": "Haute", "CAD_CHF": "Haute", "CHF_JPY": "Haute",
    "NZD_JPY": "Haute", "NZD_CAD": "Moyenne", "NZD_CHF": "Haute",
    "DE30_EUR": "Très Haute", "XAU_USD": "Très Haute", "SPX500_USD": "Très Haute",
    "NAS100_USD": "Très Haute", "US30_USD": "Très Haute",
}

TIMEFRAMES: dict[str, str] = {"H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"}
SWING_LOOKBACK: dict[str, int] = {"H1": 5, "H4": 5, "D1": 4, "Weekly": 3}
SWING_HISTORY: dict[str, int]  = {"H1": 120, "H4": 90, "D1": 60, "Weekly": 26}
ATR_DIST_MULT: float = 1.8
MIN_SCORE: int = 65

SCAN_GLOBAL_TIMEOUT: int = 180

TF_STATUT: dict[str, dict[str, int]] = {
    "H1":     {"Fresh": 4, "Aged": 12},
    "H4":     {"Fresh": 3, "Aged": 8},
    "D1":     {"Fresh": 2, "Aged": 5},
    "Weekly": {"Fresh": 2, "Aged": 4},
}

DISPLAY_COLS: list[str] = [
    "Instrument", "Timeframe", "Type", "Ordre", "Signal",
    "Niveau", "Distance%", "Volatilité", "Force", "BB_Width", "Statut", "Heure (UTC)",
]
EXPORT_COLS: list[str] = list(DISPLAY_COLS)

GRAN_COUNT: dict[str, int] = {"H1": 400, "H4": 300, "D": 200, "W": 120}

# Cache TTL per granularity (seconds): aligned to candle close cadence
_TTL_BY_GRAN: dict[str, int] = {"H1": 60, "H4": 300, "D": 1800, "W": 3600}

# ===================== TYPE ALIASES =====================
Pivot = tuple[int, float, str]
Swing = dict[str, float | int | str]
SigResult = tuple[Optional[str], Optional[str], Optional[float]]
_NONE_SIG: SigResult = (None, None, None)

# ===================== API (cached resource) =====================
@st.cache_resource(show_spinner=False)
def _get_api() -> API:
    """Singleton OANDA API client, thread-safe via Streamlit's cache_resource."""
    token = st.secrets["OANDA_ACCESS_TOKEN"]
    return API(access_token=token, request_params={"timeout": 12})

def _reset_api_cache() -> None:
    """Invalidate cached API on auth failure (e.g. token rotation)."""
    try:
        _get_api.clear()
    except Exception:  # pragma: no cover - defensive
        pass

try:
    _ = st.secrets["OANDA_ACCESS_TOKEN"]
except KeyError:
    st.error("Clé API OANDA manquante dans les secrets.")
    st.stop()

# ===================== UTILITAIRES =====================
def _compute_true_range(data: pd.DataFrame) -> np.ndarray:
    high = data["high"].values
    low = data["low"].values
    close = data["close"].values
    return np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )


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


def calc_distance_pct(niveau: Optional[float], close_actuel: Optional[float]) -> Optional[float]:
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


def get_session(dt: datetime) -> str:
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
    """Parse and validate one OANDA candle dict, returns None on any anomaly."""
    try:
        mid = c["mid"]
        open_v = float(mid["o"])
        high_v = float(mid["h"])
        low_v  = float(mid["l"])
        close_v = float(mid["c"])
        t = pd.to_datetime(c["time"], utc=True, errors="raise")
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Bougie malformée [%s/%s] %s: %s",
                       inst, gran, type(exc).__name__, exc)
        return None
    if not all(np.isfinite(v) for v in (open_v, high_v, low_v, close_v)):
        logger.warning("Prix non-fini ignoré [%s/%s] t=%s", inst, gran, c.get("time"))
        return None
    if not (low_v <= min(open_v, close_v) <= max(open_v, close_v) <= high_v):
        logger.warning("OHLC incohérent [%s/%s] t=%s", inst, gran, c.get("time"))
        return None
    return {"time": t, "open": open_v, "high": high_v, "low": low_v, "close": close_v}


# ----- API fetch: cached + raw -----
def _fetch_candles_uncached(inst: str, gran: str) -> Optional[pd.DataFrame]:
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
            _reset_api_cache()
        logger.warning("V20Error [%s/%s] code=%s", inst, gran, exc.code)
        return None
    except (ConnectionError, TimeoutError, OSError) as exc:
        logger.warning("Réseau [%s/%s] %s: %s", inst, gran, type(exc).__name__, exc)
        return None
    except (KeyError, ValueError, TypeError) as exc:
        logger.error("Parsing [%s/%s] %s: %s", inst, gran, type(exc).__name__, exc)
        return None


@st.cache_data(ttl=60, max_entries=512, show_spinner=False)
def _fetch_candles_cached(inst: str, gran: str, bucket: int) -> Optional[pd.DataFrame]:
    """`bucket` discretizes wall-clock into TTL-sized windows for cache invalidation."""
    return _fetch_candles_uncached(inst, gran)


def get_candles(inst: str, gran: str) -> Optional[pd.DataFrame]:
    """Public entry point with TTL-aware caching by granularity."""
    ttl = _TTL_BY_GRAN.get(gran, 300)
    bucket = int(datetime.now(timezone.utc).timestamp() // ttl)
    return _fetch_candles_cached(inst, gran, bucket)


def compute_bb_width(data: pd.DataFrame, length: int = 20, std: int = 2) -> str:
    close = data["close"]
    if len(close) < length * 2:
        return "N/A"
    sma = close.rolling(length).mean()
    std_dev = close.rolling(length).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    bb_w = (upper - lower) / sma
    bb_avg = bb_w.rolling(length).mean()
    avg_last = bb_avg.iloc[-1]
    if pd.isna(avg_last) or np.isclose(avg_last, 0, atol=1e-10):
        return "N/A"
    bb_avg_safe = bb_avg.replace(0, np.nan)
    pct = ((bb_w - bb_avg) / bb_avg_safe * 100).iloc[-1]
    if pd.isna(pct):
        return "N/A"
    sign = "+" if pct >= 0 else ""
    if pct <= -25:
        return f"{sign}{pct:.0f}%_Squeeze"
    if pct >= 25:
        return f"{sign}{pct:.0f}%_Expansion"
    return f"{sign}{pct:.0f}%_Normal"


def compute_statut(idx_sig: Optional[int], len_df: int, tf: str) -> str:
    if idx_sig is None:
        return "N/A"
    candles_elapsed = (len_df - 1) - idx_sig
    thresholds = TF_STATUT.get(tf, {"Fresh": 2, "Aged": 5})
    if candles_elapsed <= thresholds["Fresh"]:
        return "Fresh"
    if candles_elapsed <= thresholds["Aged"]:
        return "Aged"
    return "Stale"


# ===================== CORE V5.9 =====================
def _classify_swings(pivots: list[Pivot]) -> list[Swing]:
    swings: list[Swing] = []
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


def detect_swing_points(data: pd.DataFrame, tf: str) -> list[Swing]:
    """Vectorized pivot detection via centered rolling max/min."""
    lookback = SWING_LOOKBACK.get(tf, 5)
    history = SWING_HISTORY.get(tf, 60)
    high_arr = data["high"].values
    low_arr = data["low"].values
    n = len(high_arr)
    start = max(lookback, n - history - lookback)
    end = n - lookback - 1
    if end <= start:
        return []

    window = 2 * lookback + 1
    roll_max = pd.Series(high_arr).rolling(window, center=True).max().to_numpy()
    roll_min = pd.Series(low_arr).rolling(window, center=True).min().to_numpy()
    idx_range = np.arange(start, end)
    is_high = high_arr[idx_range] == roll_max[idx_range]
    is_low  = low_arr[idx_range]  == roll_min[idx_range]

    pivots: list[Pivot] = (
        [(int(i), float(high_arr[i]), "H") for i in idx_range[is_high]] +
        [(int(i), float(low_arr[i]),  "L") for i in idx_range[is_low]]
    )
    pivots.sort(key=lambda x: x[0])

    # FIX C: deduplicate identical-price pivots (equal highs/lows on illiquid markets)
    seen: dict[tuple, int] = {}
    for pos, (i, price, k) in enumerate(pivots):
        seen[(price, k)] = pos
    dedup_positions = set(seen.values())
    pivots = [p for pos, p in enumerate(pivots) if pos in dedup_positions]
    pivots.sort(key=lambda x: x[0])
    return _classify_swings(pivots)


def get_structural_trend(swings: list[Swing]) -> str:
    if len(swings) < 4:
        return "Range"
    rec = swings[-6:]
    highs = [s for s in rec if s["kind"] in ("HH", "LH")]
    lows = [s for s in rec if s["kind"] in ("HL", "LL")]
    if not highs or not lows:
        return "Range"
    if highs[-1]["kind"] == "HH" and lows[-1]["kind"] == "HL":
        return "Bullish"
    if highs[-1]["kind"] == "LH" and lows[-1]["kind"] == "LL":
        return "Bearish"
    return "Range"


# ---- detect_choch_v58 helpers (unchanged logic) ------------------------------
def _resolve_bullish_trend(close_arr: np.ndarray, idx: int,
                            prev_swings: list[Swing]) -> SigResult:
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


def _resolve_bearish_trend(close_arr: np.ndarray, idx: int,
                            prev_swings: list[Swing]) -> SigResult:
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


def _resolve_signal(trend: str, close_arr: np.ndarray, idx: int,
                     prev_swings: list[Swing]) -> SigResult:
    if trend == "Bullish":
        return _resolve_bullish_trend(close_arr, idx, prev_swings)
    if trend == "Bearish":
        return _resolve_bearish_trend(close_arr, idx, prev_swings)
    return _NONE_SIG


def _has_valid_body(high_arr: np.ndarray, low_arr: np.ndarray,
                     close_arr: np.ndarray, open_arr: np.ndarray,
                     idx: int, min_body_ratio: float = 0.40) -> bool:
    rng = high_arr[idx] - low_arr[idx]
    if rng <= 0:
        return False
    body = abs(close_arr[idx] - open_arr[idx])
    return (body / rng) >= min_body_ratio


def _detect_liquidity_sweep(high_arr: np.ndarray, low_arr: np.ndarray, idx: int,
                             prev_swings: list[Swing], atr_val: float,
                             direction: str) -> bool:
    if direction == "Bearish":
        candidates = [s for s in prev_swings if s["kind"] in ("HH", "LH")]
        return any(
            high_arr[idx] > s["price"] and (high_arr[idx] - s["price"]) > (atr_val * 0.25)
            for s in candidates
        )
    candidates = [s for s in prev_swings if s["kind"] in ("HL", "LL")]
    return any(
        low_arr[idx] < s["price"] and (s["price"] - low_arr[idx]) > (atr_val * 0.25)
        for s in candidates
    )


def _compute_confluence_score(dist_atr: float, idx: int, df_index: pd.DatetimeIndex,
                               has_sweep: bool, sig_type: str) -> int:
    score = 25
    if dist_atr <= 1.0:
        score += 15
    if is_premium_session(get_session(df_index[idx])):
        score += 20
    if has_sweep:
        score += 15
    if sig_type == "CHoCH":
        score += 10
    return min(score, 100)


def detect_choch_v58(df: pd.DataFrame, tf: str, inst: str) -> Optional[dict]:
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
        if not _has_valid_body(high_arr, low_arr, close_arr, open_arr, idx):
            continue

        has_sweep = (
            sig_type == "CHoCH"
            and _detect_liquidity_sweep(high_arr, low_arr, idx, prev_swings,
                                         atr_val, direction)
        )

        dist_atr = abs(close_arr[idx] - level) / atr_val
        if dist_atr > ATR_DIST_MULT:
            continue

        score = _compute_confluence_score(dist_atr, idx, df.index, has_sweep, sig_type)
        if score < MIN_SCORE:
            continue

        rng_v = high_arr[idx] - low_arr[idx]
        body_v = abs(close_arr[idx] - open_arr[idx])
        if rng_v > 0 and body_v / rng_v >= 0.6:
            force_label = "Fort"
        elif rng_v > 0 and body_v / rng_v >= 0.4:
            force_label = "Moyen"
        else:
            force_label = "Faible"

        return {
            "sig_type": sig_type, "direction": direction, "level": level,
            "idx_break": idx, "close_price": close_arr[idx],
            "current_price": close_arr[-1], "has_sweep": has_sweep,
            "atr_val": atr_val, "volatilite": volatilite_label,
            "trend": trend, "force": force_label,
            "dist_atr": dist_atr, "score": score,
        }
    return None


def build_pipeline_payload_v58(df: pd.DataFrame, inst: str, inst_disp: str,
                                tf_name: str, sig: dict, trend: str,
                                scan_time: datetime, len_df: int, bb_str: str,
                                volatilite: str, force: str) -> dict:
    time_sig = df.index[sig["idx_break"]]
    session = get_session(time_sig)
    dist_pct = calc_distance_pct(sig["level"], sig["close_price"])
    current_dist_pct = calc_distance_pct(sig["level"], sig["current_price"])
    candles_since = (len_df - 1) - sig["idx_break"]
    statut = compute_statut(sig["idx_break"], len_df, tf_name)
    prec = instrument_precision(inst)

    _bb_raw = bb_str.split("%")[0].replace("+", "").strip() if "%" in bb_str else None
    try:
        bb_pct: Optional[float] = float(_bb_raw) if _bb_raw is not None else None
    except ValueError:
        bb_pct = None
    bb_regime = bb_str.split("%")[-1].strip().lstrip("_") if "%" in bb_str else "N/A"

    return {
        "signal_id": (
            f"{inst}__{tf_name}__{time_sig.strftime('%Y%m%dT%H%M')}"
            f"__scan{scan_time.strftime('%Y%m%dT%H%M')}"
        ),
        "scanner_version": SCANNER_VERSION,
        "generated_at": scan_time.isoformat(),
        "pair": inst_disp, "pair_oanda": inst, "timeframe": tf_name,
        "type": sig["sig_type"], "direction": sig["direction"],
        "is_bullish": sig["direction"] == "Bullish",
        "order": "buy" if sig["direction"] == "Bullish" else "sell",
        "trend": trend, "is_choch": sig["sig_type"] == "CHoCH",
        "status": statut, "confluence_score": sig["score"],
        "level": round(float(sig["level"]), prec),
        "close_price": round(float(sig["close_price"]), prec),
        "current_price": round(float(sig["current_price"]), prec),
        "distance_pct": round(dist_pct, 4) if dist_pct is not None else None,
        "current_distance_pct": round(current_dist_pct, 4) if current_dist_pct is not None else None,
        "distance_atr_multiple": round(sig["dist_atr"], 2),
        "volatility": volatilite, "force": force,
        "bb_width_pct": bb_pct, "bb_regime": bb_regime,
        "session": session, "signal_time": time_sig.isoformat(),
        "candles_elapsed": candles_since,
    }


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
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4),
                             leftMargin=20, rightMargin=20,
                             topMargin=40, bottomMargin=40)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph(f"Rapport des Signaux CHoCH v{SCANNER_VERSION}", styles["Title"]))
    elements.append(Paragraph(
        f"Généré le {datetime.now(timezone.utc).strftime('%d/%m/%Y à %H:%M')} UTC",
        styles["Normal"],
    ))
    elements.append(Spacer(1, 20))

    cols_present = [c for c in EXPORT_COLS if c in df_export.columns]
    col_widths_map = {c: 60 for c in cols_present}
    col_widths_map.update({"Instrument": 65, "Distance%": 52, "Statut": 45,
                            "Heure (UTC)": 105})
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
    fig = Figure(figsize=(22, min(max(5, len(data) * 0.35), 50)))
    ax = fig.add_subplot(111)
    ax.axis("off")
    disp = data[[c for c in display_cols if c in data.columns]]
    tbl = ax.table(cellText=disp.values, colLabels=disp.columns,
                    cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    return buf


# Cached export wrappers: payload is the DataFrame serialized to JSON (stable hash)
@st.cache_data(ttl=900, max_entries=8, show_spinner=False)
def _generate_png_bytes(payload_json: str) -> bytes:
    df = pd.read_json(io.StringIO(payload_json), orient="split")
    return generate_png(df, DISPLAY_COLS).getvalue()


@st.cache_data(ttl=900, max_entries=8, show_spinner=False)
def _generate_pdf_bytes(payload_json: str) -> bytes:
    df = pd.read_json(io.StringIO(payload_json), orient="split")
    return create_pdf(df).getvalue()


# ===================== UI =====================
st.set_page_config(page_title="CHoCH Scanner v5.9.1", layout="wide")
st.title("Scanner Change of Character (CHoCH) — v5.9.1 Intraday")

if "scanning" not in st.session_state:
    st.session_state.scanning = False


def _reset_scan_state() -> None:
    for key in ("df", "pipeline_signals"):
        st.session_state.pop(key, None)


if st.button("Lancer le Scan", type="primary", use_container_width=True,
              disabled=st.session_state.scanning):
    _reset_scan_state()
    st.session_state.scanning = True
    n_combos = len(INSTRUMENTS) * len(TIMEFRAMES)
    st.session_state.scan_time = datetime.now(timezone.utc)
    scan_time = st.session_state.scan_time

    try:
        with st.spinner(f"Scan en cours sur {n_combos} combinaisons…"):
            results: list[dict] = []
            pipeline_signals: list[dict] = []
            errors: list[str] = []

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
                        f"Timeout global atteint — {len(not_done)} requête(s) "
                        "ignorée(s), résultats partiels."
                    )

                for future in done:
                    inst, tf_name = futures[future]
                    try:
                        df = future.result()
                    except Exception as exc:
                        errors.append(f"{inst}/{tf_name}: {type(exc).__name__}: {exc}")
                        continue
                    if df is None:
                        continue

                    sig = detect_choch_v58(df, tf_name, inst)
                    if not sig:
                        continue

                    trend = sig["trend"]
                    volatilite = sig["volatilite"]
                    force = sig["force"]

                    inst_display = inst.replace("_", "/")
                    statut = compute_statut(sig["idx_break"], len(df), tf_name)
                    df_sub = df.iloc[: sig["idx_break"] + 1]
                    bb_str = compute_bb_width(df_sub)
                    dist_pct = calc_distance_pct(sig["level"], sig["close_price"])

                    results.append({
                        "Instrument": inst_display,
                        "Paire": inst_display,
                        "_time_sort": df.index[sig["idx_break"]],
                        "Timeframe": tf_name,
                        "Type": sig["sig_type"],
                        "Ordre": "Achat" if sig["direction"] == "Bullish" else "Vente",
                        "Signal": f"{sig['direction']} {sig['sig_type']}",
                        "Niveau": format_niveau(sig["level"], inst),
                        "Distance%": format_distance(dist_pct),
                        "Volatilité": volatilite,
                        "Force": force,
                        "BB_Width": bb_str,
                        "Statut": statut,
                        "Heure (UTC)": df.index[sig["idx_break"]].strftime("%Y-%m-%d %H:%M"),
                    })

                    if statut in ("Fresh", "Aged"):
                        pipeline_signals.append(build_pipeline_payload_v58(
                            df, inst, inst_display, tf_name, sig, trend,
                            scan_time, len(df), bb_str, volatilite, force,
                        ))
            finally:
                # robust shutdown: do not block UI on slow workers
                executor.shutdown(wait=False, cancel_futures=True)

            if errors:
                st.warning(f"{len(errors)} erreur(s) : {'; '.join(errors[:5])}")
            if results:
                df_sorted = pd.DataFrame(results).sort_values("_time_sort", ascending=False)
                before_dedup = len(df_sorted)
                df_result = (
                    df_sorted
                    .drop_duplicates(subset=["Instrument", "Timeframe", "Type", "Ordre"],
                                     keep="first")
                    .drop(columns=["_time_sort"])
                    .reset_index(drop=True)
                )
                if before_dedup > len(df_result):
                    logger.warning("%d doublon(s) éliminé(s)", before_dedup - len(df_result))
                st.session_state.df = df_result
                st.session_state.pipeline_signals = pipeline_signals
                st.success(
                    f"Scan terminé – {len(df_result)} signaux sur {n_combos} "
                    f"combinaisons | {len(pipeline_signals)} dans le pipeline JSON"
                )
            else:
                st.info("Aucun signal CHoCH/BOS récent qualifié (Score ≥ 65)")
    except Exception as exc:
        st.error(f"Erreur critique : {exc}")
        logger.exception("Erreur scan: %s", exc)
    finally:
        st.session_state.scanning = False


if "df" in st.session_state:
    df_all = st.session_state.df.copy()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    df_export = df_all[df_all["Statut"].isin(["Fresh", "Aged"])].copy()
    pipeline_signals = st.session_state.get("pipeline_signals", [])
    scan_time_meta = st.session_state.get("scan_time", datetime.now(timezone.utc))

    n_stale = len(df_all[df_all["Statut"] == "Stale"])
    if n_stale > 0:
        st.info(f"{n_stale} signal(s) Stale visible(s) dans le tableau — exclus des exports.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        csv_cols = [c for c in EXPORT_COLS if c in df_export.columns]
        st.download_button(
            "CSV",
            df_export[csv_cols].to_csv(index=False).encode(),
            f"choch_{ts}.csv",
            "text/csv",
        )
    with col2:
        png_payload = df_all.to_json(orient="split", date_format="iso")
        st.download_button(
            "PNG",
            _generate_png_bytes(png_payload),
            f"choch_{ts}.png",
            "image/png",
        )
    with col3:
        pdf_payload = df_export.to_json(orient="split", date_format="iso")
        st.download_button(
            "PDF",
            _generate_pdf_bytes(pdf_payload),
            f"choch_signaux_{ts}.pdf",
            "application/pdf",
        )
    with col4:
        pipeline_json = json.dumps(
            {
                "meta": {
                    "scanner_version": SCANNER_VERSION,
                    "generated_at": scan_time_meta.isoformat(),
                    "signal_count": len(pipeline_signals),
                },
                "signals": pipeline_signals,
            },
            ensure_ascii=False, indent=2, default=_json_default,
        ).encode("utf-8")
        st.download_button(
            "JSON",
            pipeline_json,
            f"choch_pipeline_{ts}.json",
            "application/json",
        )

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

    cols_display = [c for c in DISPLAY_COLS if c in df_all.columns]
    st.dataframe(
        df_all[cols_display]
        .style
        .map(lambda x: ("color:#e879f9;font-weight:bold" if x == "CHoCH"
                         else "color:#94a3b8" if x == "BOS" else ""),
              subset=["Type"])
        .map(lambda x: ("color:#00c853;font-weight:bold" if x == "Achat"
                         else "color:#ff5252;font-weight:bold" if x == "Vente" else ""),
              subset=["Ordre"])
        .map(lambda x: ("color:#00c853" if "Bull" in str(x)
                         else "color:#ff5252" if "Bear" in str(x) else ""),
              subset=["Signal"])
        .map(lambda x: ("color:#00c853;font-weight:bold" if x == "Fort"
                         else "color:#ff5252" if x == "Faible"
                         else "color:#ff9800"),
              subset=["Force"])
        .map(_style_bb, subset=["BB_Width"])
        .map(_style_distance, subset=["Distance%"])
        .map(lambda x: ("color:#00c853;font-weight:bold" if x == "Fresh"
                         else "color:#ff9800;font-weight:bold" if x == "Aged"
                         else "color:#ff5252;font-weight:bold" if x == "Stale"
                         else ""),
              subset=["Statut"]),
        hide_index=True,
        use_container_width=True,
    )

    if pipeline_signals:
        with st.expander(f"Aperçu JSON Pipeline ({len(pipeline_signals)} signaux Fresh/Aged)"):
            st.json(pipeline_signals[0])
