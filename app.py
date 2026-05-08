# ============================================================
# SCANNER CHOCH OANDA — v5.6 PIPELINE
# Nouveautés v5.6 (sur base v5.5 patchée) :
#   P14 - JSON pipeline-grade : types natifs (float, int, bool)
#   P14 - signal_id unique par signal (pair + tf + timestamp)
#   P14 - signal_time en ISO 8601 strict avec timezone
#   P14 - bb_width_pct (float) et bb_regime séparés
#   P14 - distance_pct float sans symbole %
#   P14 - level et close_price en float natif
#   P14 - champ trend exposé dans le payload
#   P14 - champ session (London/NewYork/Asia/Off)
#   P14 - scanner_version et generated_at dans chaque signal
#   P14 - export JSON pipeline séparé de l'export UI
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import logging
import threading

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.exceptions import V20Error

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

SCANNER_VERSION = "5.6"

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
    "EUR_USD": "Basse",   "GBP_USD": "Basse",   "USD_JPY": "Basse",
    "USD_CHF": "Basse",   "USD_CAD": "Basse",
    "AUD_USD": "Moyenne", "NZD_USD": "Moyenne",  "EUR_GBP": "Moyenne",
    "EUR_JPY": "Moyenne", "EUR_CHF": "Moyenne",  "EUR_AUD": "Moyenne",
    "EUR_CAD": "Moyenne", "EUR_NZD": "Moyenne",
    "GBP_JPY": "Haute",   "GBP_CHF": "Haute",    "GBP_AUD": "Haute",
    "GBP_CAD": "Haute",   "GBP_NZD": "Haute",
    "AUD_JPY": "Haute",   "AUD_CAD": "Moyenne",  "AUD_CHF": "Haute",
    "AUD_NZD": "Moyenne", "CAD_JPY": "Haute",    "CAD_CHF": "Haute",
    "CHF_JPY": "Haute",   "NZD_JPY": "Haute",
    "NZD_CAD": "Moyenne", "NZD_CHF": "Haute",
    "DE30_EUR":   "Très Haute", "XAU_USD":    "Très Haute",
    "SPX500_USD": "Très Haute", "NAS100_USD": "Très Haute",
    "US30_USD":   "Très Haute",
}

TIMEFRAMES     = {"H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"}
FRACTAL_LEN    = {"H1": 5, "H4": 5, "D1": 7, "Weekly": 7}
FRACTAL_WINDOW = {"H1": 120, "H4": 90, "D1": 60, "Weekly": 26}

TF_STATUT = {
    "H1":     {"Fresh": 4,  "Aged": 12},
    "H4":     {"Fresh": 3,  "Aged": 8},
    "D1":     {"Fresh": 2,  "Aged": 5},
    "Weekly": {"Fresh": 2,  "Aged": 4},
}

TF_LOOKBACK     = {"H1": 5, "H4": 5, "D1": 4, "Weekly": 3}
TF_EMA_LOOKBACK = {"H1": 5, "H4": 5, "D1": 10, "Weekly": 15}
GRAN_COUNT      = {"H1": 400, "H4": 300, "D": 200, "W": 120}

# Colonnes pour l'affichage UI (inchangé)
DISPLAY_COLS = [
    "Instrument", "Timeframe", "Type", "Ordre", "Signal",
    "Niveau", "Distance%", "Volatilité", "Force", "BB_Width", "Statut", "Heure (UTC)"
]
# Colonnes pour les exports UI (PDF / CSV / PNG)
EXPORT_COLS = [
    "Paire", "Timeframe", "Type", "Ordre", "Signal",
    "Niveau", "Distance%", "Volatilité", "Force", "BB_Width", "Statut", "Heure (UTC)"
]

# ===================== API THREAD-SAFE =====================
_thread_local = threading.local()

def _get_api() -> API:
    if not hasattr(_thread_local, "api"):
        try:
            _thread_local.api = API(
                access_token=st.secrets["OANDA_ACCESS_TOKEN"],
                request_params={"timeout": 12}
            )
        except Exception as e:
            logger.critical(f"Impossible d'initialiser l'API OANDA : {e}")
            raise
    return _thread_local.api

try:
    _ = st.secrets["OANDA_ACCESS_TOKEN"]
except Exception as e:
    st.error(f"Token OANDA manquant dans les secrets Streamlit : {e}")
    st.stop()


# ===================== UTILITAIRES =====================

def _compute_true_range(df: pd.DataFrame) -> np.ndarray:
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    return np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    )


def get_session(dt: datetime) -> str:
    """
    P14: Retourne la session de trading active au moment du signal.
    Basé sur l'heure UTC. Utilisé dans le payload JSON pipeline.
    Sessions : Sydney (21-06), Tokyo (00-09), London (07-16), NewYork (13-22).
    On retient la session dominante (London > NewYork > Tokyo > Off).
    """
    h = dt.hour
    if 7 <= h < 16:
        return "London"
    if 13 <= h < 22:
        return "NewYork"
    if 0 <= h < 9:
        return "Tokyo"
    return "Off"


def _parse_bb_components(bb_str: str) -> tuple[float | None, str]:
    """
    P14: Décompose la string BB_Width en (pct_float, regime_str).
    Entrée : "+23%_Normal" | "-32%_Squeeze" | "+28%_Expansion" | "N/A"
    Sortie : (23.0, "Normal") | (-32.0, "Squeeze") | (28.0, "Expansion") | (None, "N/A")
    """
    if bb_str == "N/A":
        return None, "N/A"
    try:
        pct_part, regime = bb_str.split("%_")
        return round(float(pct_part), 2), regime
    except Exception:
        return None, "N/A"


# ===================== FONCTIONS CORE =====================

def get_candles(inst: str, gran: str) -> pd.DataFrame | None:
    count = GRAN_COUNT.get(gran, 300)
    try:
        r = instruments.InstrumentsCandles(
            instrument=inst,
            params={"count": count, "granularity": gran}
        )
        _get_api().request(r)
        candles = [c for c in r.response.get("candles", []) if c.get("complete")]
        if len(candles) < 50:
            return None
        df = pd.DataFrame([{
            "time":  pd.to_datetime(c["time"], utc=True),
            "open":  float(c["mid"]["o"]),
            "high":  float(c["mid"]["h"]),
            "low":   float(c["mid"]["l"]),
            "close": float(c["mid"]["c"])
        } for c in candles])
        df.set_index("time", inplace=True)
        return df
    except V20Error as e:
        if e.code in (401, 403):
            logger.error(f"Auth OANDA échouée [{inst}/{gran}]: {e}")
        elif e.code == 429:
            logger.warning(f"Rate limit [{inst}/{gran}]")
        else:
            logger.warning(f"V20Error [{inst}/{gran}] code={e.code}: {e}")
        return None
    except Exception as e:
        logger.error(f"Erreur get_candles [{inst}/{gran}]: {type(e).__name__}: {e}")
        return None


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    tr = _compute_true_range(df)
    if len(tr) < period * 3:
        return np.nan
    return float(pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


def atr_to_volatility(atr_val: float, inst: str, df: pd.DataFrame) -> str:
    if np.isnan(atr_val) or len(df) < 28:
        return VOLATILITY_STATIC.get(inst, "Moyenne")
    tr = _compute_true_range(df)
    window = tr[-100:] if len(tr) >= 100 else tr
    median_tr = float(np.median(window))
    if np.isclose(median_tr, 0, atol=1e-10):
        return VOLATILITY_STATIC.get(inst, "Moyenne")
    ratio = atr_val / median_tr
    if ratio >= 1.8: return "Très Haute"
    if ratio >= 1.2: return "Haute"
    if ratio >= 0.7: return "Moyenne"
    return "Basse"


def compute_bb_width(df: pd.DataFrame, length: int = 20, std: int = 2) -> str:
    """Retourne la string UI. La décomposition pour pipeline est faite via _parse_bb_components."""
    close = df["close"]
    if len(close) < length * 2:
        return "N/A"
    sma      = close.rolling(length).mean()
    std_dev  = close.rolling(length).std()
    upper    = sma + std * std_dev
    lower    = sma - std * std_dev
    bb_w     = (upper - lower) / sma
    bb_avg   = bb_w.rolling(length).mean()
    avg_last = bb_avg.iloc[-1]
    if pd.isna(avg_last) or np.isclose(avg_last, 0, atol=1e-10):
        return "N/A"
    bb_avg_safe = bb_avg.replace(0, np.nan)
    pct = ((bb_w - bb_avg) / bb_avg_safe * 100).iloc[-1]
    if pd.isna(pct):
        return "N/A"
    sign = "+" if pct >= 0 else ""
    if pct < -25: return f"{sign}{pct:.0f}%_Squeeze"
    if pct > 25:  return f"{sign}{pct:.0f}%_Expansion"
    return f"{sign}{pct:.0f}%_Normal"


def compute_statut(idx_sig: int | None, len_df: int, tf: str) -> str:
    if idx_sig is None:
        return "N/A"
    candles_elapsed = (len_df - 1) - idx_sig
    thresholds = TF_STATUT.get(tf, {"Fresh": 2, "Aged": 5})
    if candles_elapsed <= thresholds["Fresh"]: return "Fresh"
    if candles_elapsed <= thresholds["Aged"]:  return "Aged"
    return "Stale"


def get_trend_context(df: pd.DataFrame, tf: str) -> str:
    close = df["close"]
    if len(close) < 55:
        return "Unknown"
    ema20    = close.ewm(span=20, adjust=False).mean()
    ema50    = close.ewm(span=50, adjust=False).mean()
    e20_last = ema20.iloc[-1]
    e50_last = ema50.iloc[-1]
    lookback = TF_EMA_LOOKBACK.get(tf, 5)
    e20_prev = ema20.iloc[-(lookback + 1)] if len(ema20) > lookback else ema20.iloc[0]
    if e20_last > e50_last and e20_last > e20_prev: return "Uptrend"
    if e20_last < e50_last and e20_last < e20_prev: return "Downtrend"
    return "Range"


def detect_choch(df: pd.DataFrame, tf: str):
    length   = FRACTAL_LEN.get(tf, 5)
    p        = length // 2
    lookback = TF_LOOKBACK.get(tf, 3)
    window   = FRACTAL_WINDOW.get(tf, 60)

    h_all = df["high"].values
    l_all = df["low"].values
    c_all = df["close"].values
    o_all = df["open"].values

    min_len = p * 2 + 1 + lookback + 1
    if len(c_all) < min_len:
        return None, None, None, None, None, None, None

    recent_ranges = h_all[-50:] - l_all[-50:]
    p25 = float(np.percentile(recent_ranges, 25)) if len(recent_ranges) >= 10 else None
    p75 = float(np.percentile(recent_ranges, 75)) if len(recent_ranges) >= 10 else None

    def get_force(rng: float) -> str:
        if p25 is None or p75 is None: return "Moyen"
        if rng >= p75: return "Fort"
        if rng <= p25: return "Faible"
        return "Moyen"

    def is_valid_breakout_candle(idx: int) -> bool:
        rng = h_all[idx] - l_all[idx]
        if np.isclose(rng, 0, atol=1e-10):
            return False
        body = abs(c_all[idx] - o_all[idx])
        return (body / rng) >= 0.3

    for offset in range(lookback):
        idx_cur  = len(df) - 1 - offset
        idx_prev = idx_cur - 1
        if idx_prev < p:
            break

        h = h_all[:idx_cur + 1]
        l = l_all[:idx_cur + 1]
        c = c_all[:idx_cur + 1]

        window_start      = max(p, len(h) - window)
        fractal_search_end = idx_cur - p

        if fractal_search_end <= window_start:
            continue

        last_high_idx = None
        last_low_idx  = None

        for i in range(window_start, fractal_search_end):
            local_max = max(h[i - p:i + p + 1])
            local_min = min(l[i - p:i + p + 1])
            if abs(h[i] - local_max) < 1e-9:
                last_high_idx = i
            if abs(l[i] - local_min) < 1e-9:
                last_low_idx = i

        last_high = float(h[last_high_idx]) if last_high_idx is not None else None
        last_low  = float(l[last_low_idx])  if last_low_idx  is not None else None

        if last_high is None and last_low is None:
            continue

        if not is_valid_breakout_candle(idx_cur):
            continue

        breakout_range = h_all[idx_cur] - l_all[idx_cur]
        sig_raw = time_sig = force = niveau_casse = None

        if last_high is not None and c[idx_cur] > last_high and c[idx_prev] <= last_high:
            sig_raw      = "Bullish"
            time_sig     = df.index[idx_cur]
            force        = get_force(breakout_range)
            niveau_casse = last_high

        elif last_low is not None and c[idx_cur] < last_low and c[idx_prev] >= last_low:
            sig_raw      = "Bearish"
            time_sig     = df.index[idx_cur]
            force        = get_force(breakout_range)
            niveau_casse = last_low

        if sig_raw is None:
            continue

        trend = get_trend_context(df.iloc[:idx_cur + 1], tf)

        if trend == "Range" and force in ("Faible", "Moyen"):
            continue

        return sig_raw, time_sig, force, idx_cur, trend, niveau_casse, float(c_all[-1])

    return None, None, None, None, None, None, None


def format_niveau(niveau: float | None, inst: str) -> str:
    if niveau is None:
        return "N/A"
    if any(k in inst for k in ["SPX500", "NAS100", "US30", "DE30", "XAU", "XAG"]):
        return f"{niveau:.2f}"
    if "JPY" in inst:
        return f"{niveau:.3f}"
    return f"{niveau:.5f}"


def format_distance(niveau: float | None, close_actuel: float | None, inst: str) -> str:
    if niveau is None or close_actuel is None or np.isclose(niveau, 0, atol=1e-8):
        return "N/A"
    dist = abs(close_actuel - niveau) / abs(niveau) * 100
    if dist > 100:
        return "N/A"
    return f"{dist:.3f}%"


def _precision_for(inst: str) -> int:
    """Retourne le nombre de décimales selon l'instrument — pour les floats pipeline."""
    if any(k in inst for k in ["SPX500", "NAS100", "US30", "DE30", "XAU", "XAG"]):
        return 2
    if "JPY" in inst:
        return 3
    return 5


# ===================== PAYLOAD PIPELINE =====================

def build_pipeline_payload(
    inst: str,
    inst_display: str,
    tf_name: str,
    sig_raw: str,
    time_sig,
    strength: str,
    idx_sig: int,
    trend: str,
    niveau_casse: float,
    close_actuel: float,
    type_label: str,
    statut: str,
    volatilite: str,
    bb_str: str,
    scan_time: datetime,
    len_df: int,
) -> dict:
    """
    P14: Construit le payload JSON pipeline-grade.
    Contrat strict :
      - Tous les nombres sont des types natifs Python (float/int), jamais des strings
      - signal_time en ISO 8601 avec timezone explicite (+00:00)
      - signal_id unique et déterministe
      - bb_width_pct et bb_regime séparés
      - distance_pct sans symbole %
      - champ trend exposé
      - champ session (London/NewYork/Tokyo/Off)
      - scanner_version et generated_at dans chaque signal
    """
    prec = _precision_for(inst)
    bb_pct, bb_regime = _parse_bb_components(bb_str)

    # distance_pct : float pur, None si non calculable
    dist_pct: float | None = None
    if not np.isclose(niveau_casse, 0, atol=1e-8):
        raw_dist = abs(close_actuel - niveau_casse) / abs(niveau_casse) * 100
        if raw_dist <= 100:
            dist_pct = round(raw_dist, 4)

    candles_since_signal = (len_df - 1) - idx_sig

    return {
        # ── Identité du signal ──────────────────────────────────────────
        "signal_id":       f"{inst}__{tf_name}__{time_sig.strftime('%Y%m%dT%H%M')}",
        "scanner_version": SCANNER_VERSION,
        "generated_at":    scan_time.isoformat(),          # ISO 8601 + timezone

        # ── Instrument ──────────────────────────────────────────────────
        "pair":            inst_display,                   # "EUR/USD"
        "pair_oanda":      inst,                           # "EUR_USD" (lookup interne)
        "timeframe":       tf_name,                        # "H1" | "H4" | "D1" | "Weekly"

        # ── Classification du signal ────────────────────────────────────
        "type":            type_label,                     # "CHoCH" | "BOS"
        "direction":       sig_raw,                        # "Bullish" | "Bearish"
        "is_bullish":      sig_raw == "Bullish",           # bool — filtrage facile pipeline
        "order":           "buy" if sig_raw == "Bullish" else "sell",
        "trend":           trend,                          # "Uptrend"|"Downtrend"|"Range"|"Unknown"
        "is_choch":        type_label == "CHoCH",          # bool
        "status":          statut,                         # "Fresh" | "Aged" | "Stale"

        # ── Niveaux — floats natifs, jamais strings ─────────────────────
        "level":           round(float(niveau_casse), prec),
        "close_price":     round(float(close_actuel), prec),
        "distance_pct":    dist_pct,                       # float | None

        # ── Contexte marché ─────────────────────────────────────────────
        "volatility":      volatilite,                     # "Basse"|"Moyenne"|"Haute"|"Très Haute"
        "force":           strength or "Moyen",            # "Fort" | "Moyen" | "Faible"
        "bb_width_pct":    bb_pct,                         # float | None
        "bb_regime":       bb_regime,                      # "Squeeze"|"Expansion"|"Normal"|"N/A"

        # ── Horodatage ──────────────────────────────────────────────────
        "signal_time":     time_sig.isoformat(),           # ISO 8601 strict "2026-05-07T17:00:00+00:00"
        "session":         get_session(time_sig),          # "London"|"NewYork"|"Tokyo"|"Off"
        "candles_elapsed": candles_since_signal,           # int — âge en bougies
    }


# ===================== PDF =====================
def create_pdf(df_export: pd.DataFrame) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        leftMargin=20, rightMargin=20, topMargin=40, bottomMargin=40
    )
    elements = []
    styles   = getSampleStyleSheet()
    elements.append(Paragraph("Rapport des Signaux CHoCH", styles["Title"]))
    elements.append(Paragraph(
        f"Généré le {datetime.now(timezone.utc).strftime('%d/%m/%Y à %H:%M')} UTC"
        " — Fresh & Aged uniquement",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 20))
    cols_present = [c for c in EXPORT_COLS if c in df_export.columns]
    data         = [cols_present] + df_export[cols_present].values.tolist()
    col_widths   = [65, 48, 42, 42, 82, 68, 52, 58, 45, 90, 45, 105]
    col_widths   = col_widths[:len(cols_present)]
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 9),
        ('FONTSIZE',      (0, 1), (-1, -1), 8),
        ('GRID',          (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, colors.beige]),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 5),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 5),
        ('TOPPADDING',    (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ===================== PNG =====================
def generate_png(df: pd.DataFrame, display_cols: list) -> io.BytesIO:
    fig = Figure(figsize=(22, max(5, len(df) * 0.35)))
    ax  = fig.add_subplot(111)
    ax.axis('off')
    disp = df[[c for c in display_cols if c in df.columns]]
    tbl  = ax.table(cellText=disp.values, colLabels=disp.columns,
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.8)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)
    return buf


# ===================== UI =====================
st.set_page_config(page_title="CHoCH Scanner", layout="wide")
st.markdown(
    "<h1 style='text-align:center;color:#1e40af;margin-bottom:30px;'>"
    "Scanner Change of Character (CHoCH) — v5.6</h1>",
    unsafe_allow_html=True
)

if "scanning" not in st.session_state:
    st.session_state.scanning = False

if st.button(
    "Lancer le Scan",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.scanning
):
    st.session_state.scanning = True
    n_combos  = len(INSTRUMENTS) * len(TIMEFRAMES)
    scan_time = datetime.now(timezone.utc)  # timestamp unique pour ce scan

    try:
        with st.spinner(f"Scan en cours sur {n_combos} combinaisons…"):
            results:          list[dict] = []   # données UI
            pipeline_signals: list[dict] = []   # payloads JSON pipeline
            errors:           list[str]  = []

            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {
                    executor.submit(get_candles, inst, tf_code): (inst, tf_name)
                    for inst in INSTRUMENTS
                    for tf_name, tf_code in TIMEFRAMES.items()
                }
                for future in as_completed(futures):
                    inst, tf_name = futures[future]
                    try:
                        df = future.result()
                    except Exception as e:
                        errors.append(f"{inst}/{tf_name}: {e}")
                        continue

                    if df is None:
                        continue

                    sig_raw, time_sig, strength, idx_sig, trend, niveau_casse, close_actuel = \
                        detect_choch(df, tf_name)

                    if not sig_raw:
                        continue

                    atr_val    = calc_atr(df)
                    volatilite = atr_to_volatility(atr_val, inst, df)
                    df_sig     = df.iloc[:idx_sig + 1] if idx_sig is not None else df
                    bb_str     = compute_bb_width(df_sig)

                    # Convention SMC : CHoCH = contre tendance, BOS = dans la tendance
                    is_choch = (
                        (trend == "Uptrend"   and sig_raw == "Bearish") or
                        (trend == "Downtrend" and sig_raw == "Bullish")
                    )
                    type_label   = "CHoCH" if is_choch else "BOS"
                    signal_label = f"{sig_raw} {type_label}"
                    inst_display = inst.replace("_", "/")
                    statut       = compute_statut(idx_sig, len(df), tf_name)

                    # ── Payload UI (affichage + CSV/PDF/PNG) ──────────────
                    results.append({
                        "Instrument":  inst_display,
                        "Paire":       inst_display,
                        "_time_sort":  time_sig,
                        "Timeframe":   tf_name,
                        "Type":        type_label,
                        "Ordre":       "Achat" if sig_raw == "Bullish" else "Vente",
                        "Signal":      signal_label,
                        "Niveau":      format_niveau(niveau_casse, inst),
                        "Distance%":   format_distance(niveau_casse, close_actuel, inst),
                        "Volatilité":  volatilite,
                        "Force":       strength or "Moyen",
                        "BB_Width":    bb_str,
                        "Statut":      statut,
                        "Heure (UTC)": time_sig.strftime("%Y-%m-%d %H:%M"),
                    })

                    # ── Payload pipeline JSON (P14) ───────────────────────
                    # Construit uniquement pour Fresh et Aged
                    if statut in ("Fresh", "Aged"):
                        pipeline_signals.append(build_pipeline_payload(
                            inst=inst,
                            inst_display=inst_display,
                            tf_name=tf_name,
                            sig_raw=sig_raw,
                            time_sig=time_sig,
                            strength=strength,
                            idx_sig=idx_sig,
                            trend=trend,
                            niveau_casse=niveau_casse,
                            close_actuel=close_actuel,
                            type_label=type_label,
                            statut=statut,
                            volatilite=volatilite,
                            bb_str=bb_str,
                            scan_time=scan_time,
                            len_df=len(df),
                        ))

            if errors:
                st.warning(f"{len(errors)} combinaison(s) en erreur.")

            if results:
                df_result = (
                    pd.DataFrame(results)
                    .sort_values("_time_sort", ascending=False)
                    .drop_duplicates(subset=["Instrument", "Timeframe"], keep="first")
                    .drop(columns=["_time_sort"])
                    .reset_index(drop=True)
                )
                st.session_state.df               = df_result
                st.session_state.pipeline_signals = pipeline_signals
                st.session_state.png_buf          = None
                st.success(
                    f"Scan terminé – {len(df_result)} signaux sur {n_combos} combinaisons "
                    f"| {len(pipeline_signals)} signal(s) Fresh/Aged dans le pipeline JSON"
                )
            else:
                st.info("Aucun signal CHoCH/BOS récent détecté")

    except Exception as e:
        st.error(f"Erreur critique inattendue lors du scan : {e}")
        logger.exception("Erreur critique scan")
    finally:
        st.session_state.scanning = False


# ===================== AFFICHAGE =====================
if "df" in st.session_state:
    df_all    = st.session_state.df.copy()
    ts        = datetime.now().strftime("%Y%m%d_%H%M")
    df_export = df_all[df_all["Statut"].isin(["Fresh", "Aged"])].copy()

    pipeline_signals = st.session_state.get("pipeline_signals", [])

    if st.session_state.get("png_buf") is None:
        st.session_state.png_buf = generate_png(df_all, DISPLAY_COLS)

    n_stale = len(df_all[df_all["Statut"] == "Stale"])
    if n_stale > 0:
        st.info(
            f"{n_stale} signal(s) Stale visible(s) dans le tableau — "
            "exclus des exports."
        )

    # ── Boutons d'export ──────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.download_button(
            "CSV",
            df_export[[c for c in EXPORT_COLS if c in df_export.columns]]
                .to_csv(index=False).encode(),
            f"choch_{ts}.csv", "text/csv"
        )
    with col2:
        st.session_state.png_buf.seek(0)
        st.download_button(
            "PNG",
            st.session_state.png_buf,
            f"choch_{ts}.png", "image/png"
        )
    with col3:
        st.download_button(
            "PDF",
            create_pdf(df_export),
            f"choch_signaux_{ts}.pdf", "application/pdf"
        )
    with col4:
        # JSON UI — format lisible, colonnes display
        st.download_button(
            "JSON (UI)",
            df_export[[c for c in EXPORT_COLS if c in df_export.columns]]
                .to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
            f"choch_ui_{ts}.json", "application/json",
            help="Export lisible pour consultation — colonnes affichage"
        )
    with col5:
        # JSON PIPELINE — types natifs, contrat strict pour le pipeline aval
        pipeline_json = json.dumps(
            {
                "meta": {
                    "scanner_version": SCANNER_VERSION,
                    "generated_at":    scan_time.isoformat() if "scan_time" in dir() else datetime.now(timezone.utc).isoformat(),
                    "signal_count":    len(pipeline_signals),
                },
                "signals": pipeline_signals,
            },
            ensure_ascii=False,
            indent=2,
            default=str   # fallback sécurisé pour tout type non sérialisable
        ).encode("utf-8")

        st.download_button(
            "JSON (Pipeline)",
            pipeline_json,
            f"choch_pipeline_{ts}.json", "application/json",
            help="Export pipeline — types natifs, ISO 8601, signal_id unique"
        )

    # ── Tableau UI ────────────────────────────────────────────────────
    def style_bb(val: str) -> str:
        if "Squeeze"   in str(val): return "color:#ff9800;font-weight:bold"
        if "Expansion" in str(val): return "color:#ab47bc;font-weight:bold"
        return "color:#90a4ae"

    def style_distance(val: str) -> str:
        try:
            v = float(str(val).replace("%", ""))
            if v <= 0.15: return "color:#00c853;font-weight:bold"
            if v <= 0.40: return "color:#ff9800;font-weight:bold"
            return "color:#ff5252;font-weight:bold"
        except Exception:
            return "color:#90a4ae"

    cols_display = [c for c in DISPLAY_COLS if c in df_all.columns]
    st.dataframe(
        df_all[cols_display].style
        .map(
            lambda x: "color:#e879f9;font-weight:bold" if x == "CHoCH"
            else "color:#94a3b8" if x == "BOS" else "",
            subset=["Type"]
        )
        .map(
            lambda x: "color:#00c853;font-weight:bold" if x == "Achat"
            else "color:#ff5252;font-weight:bold" if x == "Vente" else "",
            subset=["Ordre"]
        )
        .map(
            lambda x: "color:#00c853" if "Bull" in str(x)
            else "color:#ff5252" if "Bear" in str(x) else "",
            subset=["Signal"]
        )
        .map(
            lambda x: "color:#00c853;font-weight:bold" if x == "Fort"
            else "color:#ff5252" if x == "Faible"
            else "color:#ff9800" if x == "Moyen" else "",
            subset=["Force"]
        )
        .map(style_bb, subset=["BB_Width"])
        .map(style_distance, subset=["Distance%"])
        .map(
            lambda x: "color:#00c853;font-weight:bold" if x == "Fresh"
            else "color:#ff9800;font-weight:bold" if x == "Aged"
            else "color:#ff5252;font-weight:bold" if x == "Stale" else "",
            subset=["Statut"]
        ),
        hide_index=True,
        use_container_width=True
    )

    # ── Aperçu pipeline JSON dans l'UI ───────────────────────────────
    if pipeline_signals:
        with st.expander(f"Aperçu JSON Pipeline ({len(pipeline_signals)} signaux Fresh/Aged)"):
            st.json(pipeline_signals[0])   # affiche le premier signal comme exemple
