# app.py → v4.8 — Lookback 5 bougies + filtre EMA TF-aware + ATR dynamique
# v4.8 : strict breakout filter · BB_Width au signal · colonne Type CHoCH/BOS
import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ===================== CONFIG =====================
INSTRUMENTS = [
    # 28 paires Forex
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    # 6 indices et métaux
    "DE30_EUR", "XAU_USD", "XAG_USD", "SPX500_USD", "NAS100_USD", "US30_USD",
]

# Volatilité statique — fallback si ATR non disponible
VOLATILITY_STATIC = {
    "EUR_USD": "Basse",  "GBP_USD": "Basse",  "USD_JPY": "Basse",
    "USD_CHF": "Basse",  "USD_CAD": "Basse",
    "AUD_USD": "Moyenne","NZD_USD": "Moyenne", "EUR_GBP": "Moyenne",
    "EUR_JPY": "Moyenne","EUR_CHF": "Moyenne", "EUR_AUD": "Moyenne",
    "EUR_CAD": "Moyenne","EUR_NZD": "Moyenne",
    "GBP_JPY": "Haute",  "GBP_CHF": "Haute",  "GBP_AUD": "Haute",
    "GBP_CAD": "Haute",  "GBP_NZD": "Haute",
    "AUD_JPY": "Haute",  "AUD_CAD": "Moyenne", "AUD_CHF": "Haute",
    "AUD_NZD": "Moyenne","CAD_JPY": "Haute",   "CAD_CHF": "Haute",
    "CHF_JPY": "Haute",  "NZD_JPY": "Haute",
    "NZD_CAD": "Moyenne","NZD_CHF": "Haute",
    "DE30_EUR":  "Très Haute", "XAU_USD":    "Très Haute",
    "XAG_USD":  "Très Haute", "SPX500_USD": "Très Haute",
    "NAS100_USD":"Très Haute", "US30_USD":   "Très Haute",
}

TIMEFRAMES  = {"H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W", "Monthly": "M"}
FRACTAL_LEN = {"H1": 5, "H4": 6, "D1": 7, "Weekly": 8, "Monthly": 9}

TF_HOURS = {"H1": 1, "H4": 4, "D1": 24, "Weekly": 168, "Monthly": 720}

TF_STATUT = {
    "H1":      {"Fresh": 3,  "Aged": 8},
    "H4":      {"Fresh": 2,  "Aged": 5},
    "D1":      {"Fresh": 2,  "Aged": 5},
    "Weekly":  {"Fresh": 2,  "Aged": 4},
    "Monthly": {"Fresh": 1,  "Aged": 2},
}

TF_LOOKBACK = {"H1": 5, "H4": 5, "D1": 4, "Weekly": 3, "Monthly": 2}

TF_EMA_LOOKBACK = {"H1": 5, "H4": 5, "D1": 10, "Weekly": 15, "Monthly": 20}

# ===================== API =====================
try:
    api = API(
        access_token=st.secrets["OANDA_ACCESS_TOKEN"],
        request_params={"timeout": 10}
    )
except Exception as e:
    st.error(f"Token OANDA manquant dans les secrets Streamlit : {e}")
    st.stop()

# ===================== FONCTIONS =====================
def get_candles(inst, gran):
    try:
        r = instruments.InstrumentsCandles(
            instrument=inst,
            params={"count": 500, "granularity": gran}
        )
        api.request(r)
        candles = [c for c in r.response.get("candles", []) if c.get("complete")]
        if len(candles) < 50:
            return None
        df = pd.DataFrame([{
            "time":  pd.to_datetime(c["time"]),
            "open":  float(c["mid"]["o"]),
            "high":  float(c["mid"]["h"]),
            "low":   float(c["mid"]["l"]),
            "close": float(c["mid"]["c"])
        } for c in candles])
        df.set_index("time", inplace=True)
        return df
    except Exception:
        return None


def calc_atr(df, period=14):
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1])))
    if len(tr) < period:
        return np.nan
    return float(np.mean(tr[-period:]))


def atr_to_volatility(atr_val, inst, df):
    if np.isnan(atr_val) or len(df) < 28:
        return VOLATILITY_STATIC.get(inst, "Moyenne")
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1])))
    window = tr[-100:] if len(tr) >= 100 else tr
    median_tr = float(np.median(window))
    if median_tr == 0:
        return VOLATILITY_STATIC.get(inst, "Moyenne")
    ratio = atr_val / median_tr
    if ratio >= 1.8:
        return "Très Haute"
    if ratio >= 1.2:
        return "Haute"
    if ratio >= 0.7:
        return "Moyenne"
    return "Basse"


def compute_bb_width(df, length=20, std=2):
    close = df["close"]
    if len(close) < length * 2:
        return "N/A"
    sma     = close.rolling(length).mean()
    std_dev = close.rolling(length).std()
    upper   = sma + std * std_dev
    lower   = sma - std * std_dev
    bb_w    = (upper - lower) / sma
    bb_avg  = bb_w.rolling(length).mean()
    avg_last = bb_avg.iloc[-1]
    if pd.isna(avg_last) or avg_last == 0:
        return "N/A"
    pct = ((bb_w - bb_avg) / bb_avg * 100).iloc[-1]
    if pd.isna(pct):
        return "N/A"
    sign = "+" if pct >= 0 else ""
    if pct < -25:
        return f"{sign}{pct:.0f}%_Squeeze"
    if pct > 25:
        return f"{sign}{pct:.0f}%_Expansion"
    return f"{sign}{pct:.0f}%_Normal"


def compute_statut(time_sig, tf):
    try:
        now     = datetime.now(timezone.utc)
        sig_utc = time_sig.replace(tzinfo=timezone.utc) if time_sig.tzinfo is None else time_sig
        elapsed_h       = (now - sig_utc).total_seconds() / 3600
        candles_elapsed = elapsed_h / TF_HOURS.get(tf, 1)
        thresholds      = TF_STATUT.get(tf, {"Fresh": 2, "Aged": 5})
        if candles_elapsed <= thresholds["Fresh"]:
            return "Fresh"
        if candles_elapsed <= thresholds["Aged"]:
            return "Aged"
        return "Stale"
    except Exception:
        return "N/A"


def get_trend_context(df, tf):
    close = df["close"]
    if len(close) < 55:
        return "Unknown"
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    e20_last = ema20.iloc[-1]
    e50_last = ema50.iloc[-1]
    lookback = TF_EMA_LOOKBACK.get(tf, 5)
    e20_prev = ema20.iloc[-(lookback + 1)] if len(ema20) > lookback else ema20.iloc[0]
    if e20_last > e50_last and e20_last > e20_prev:
        return "Uptrend"
    if e20_last < e50_last and e20_last < e20_prev:
        return "Downtrend"
    return "Range"


def detect_choch(df, tf):
    """
    [v4.8] Retourne un 5-tuple : (sig, time_sig, force, idx_cur, trend)
    Corrections vs v4.7 :
      - c[idx_prev] < last_high  (strict, était <=)
      - c[idx_prev] > last_low   (strict, était >=)
      - idx_cur et trend exposés pour le caller
    """
    length   = FRACTAL_LEN.get(tf, 5)
    p        = length // 2
    lookback = TF_LOOKBACK.get(tf, 3)

    h_all = df["high"].values
    l_all = df["low"].values
    c_all = df["close"].values
    o_all = df["open"].values

    min_len = p * 2 + 1 + lookback + 1
    if len(c_all) < min_len:
        return None, None, None, None, None

    candle_ranges = h_all[1:] - l_all[1:]
    p25 = float(np.percentile(candle_ranges, 25)) if len(candle_ranges) >= 10 else None
    p75 = float(np.percentile(candle_ranges, 75)) if len(candle_ranges) >= 10 else None

    def get_force(rng):
        if p25 is None or p75 is None:
            return "Moyen"
        if rng >= p75:
            return "Fort"
        if rng <= p25:
            return "Faible"
        return "Moyen"

    def is_valid_breakout_candle(idx):
        rng = h_all[idx] - l_all[idx]
        if rng == 0:
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

        last_high = None
        last_low  = None
        for i in range(p, len(h) - p):
            if h[i] == max(h[i - p:i + p + 1]):
                last_high = h[i]
            if l[i] == min(l[i - p:i + p + 1]):
                last_low = l[i]

        if last_high is None and last_low is None:
            continue

        breakout_range = h_all[idx_cur] - l_all[idx_cur]

        if not is_valid_breakout_candle(idx_cur):
            continue

        sig      = None
        time_sig = None
        force    = None

        # [v4.8] Conditions strictes : < et > (était <= et >=)
        if last_high is not None and c[idx_cur] > last_high and c[idx_prev] < last_high:
            sig      = "Bullish CHoCH"
            time_sig = df.index[idx_cur]
            force    = get_force(breakout_range)

        elif last_low is not None and c[idx_cur] < last_low and c[idx_prev] > last_low:
            sig      = "Bearish CHoCH"
            time_sig = df.index[idx_cur]
            force    = get_force(breakout_range)

        if sig is None:
            continue

        trend = get_trend_context(df.iloc[:idx_cur + 1], tf)
        if trend == "Range" and force == "Faible":
            continue

        # [v4.8] Retourne idx_cur et trend pour le caller
        return sig, time_sig, force, idx_cur, trend

    return None, None, None, None, None


# ===================== PDF =====================
def create_pdf(df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        leftMargin=20, rightMargin=20, topMargin=40, bottomMargin=40
    )
    elements = []
    styles   = getSampleStyleSheet()

    elements.append(Paragraph("Rapport des Signaux CHoCH", styles["Title"]))
    elements.append(Paragraph(
        f"Généré le {datetime.now(timezone.utc).strftime('%d/%m/%Y à %H:%M')} UTC",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 20))

    data       = [df.columns.tolist()] + df.values.tolist()
    col_widths = [75, 55, 48, 48, 95, 65, 55, 100, 55, 115]

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 10),
        ('FONTSIZE',      (0, 1), (-1, -1), 9),
        ('GRID',          (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, colors.beige]),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 6),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 6),
        ('TOPPADDING',    (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ===================== PNG =====================
def generate_png(df, display_cols):
    fig, ax = plt.subplots(figsize=(20, max(5, len(df) * 0.35)))
    ax.axis('off')
    disp = df[[c for c in display_cols if c in df.columns]]
    tbl  = ax.table(cellText=disp.values, colLabels=disp.columns,
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.2, 1.8)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    buf.seek(0)
    return buf


# ===================== UI =====================
st.set_page_config(page_title="CHoCH Scanner", layout="wide")
st.markdown(
    "<h1 style='text-align:center;color:#1e40af;margin-bottom:30px;'>"
    "Scanner Change of Character (CHoCH) — v4.8</h1>",
    unsafe_allow_html=True
)

DISPLAY_COLS = [
    "Instrument", "Timeframe", "Type", "Ordre", "Signal",
    "Volatilité", "Force", "BB_Width", "Statut", "Heure (UTC)"
]

if st.button("Lancer le Scan", type="primary", use_container_width=True):
    n_combos = len(INSTRUMENTS) * len(TIMEFRAMES)
    with st.spinner(f"Scan en cours sur {n_combos} combinaisons…"):
        results = []
        errors  = []

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {
                executor.submit(get_candles, inst, code): (inst, name)
                for inst in INSTRUMENTS
                for name, code in TIMEFRAMES.items()
            }
            for future in as_completed(futures):
                inst, tf = futures[future]
                try:
                    df = future.result()
                except Exception as e:
                    errors.append(f"{inst}/{tf}: {e}")
                    continue

                if df is not None:
                    sig, time_sig, strength, idx_sig, trend = detect_choch(df, tf)
                    if sig:
                        atr_val    = calc_atr(df)
                        volatilite = atr_to_volatility(atr_val, inst, df)

                        # [v4.8] BB Width au moment du signal
                        df_sig = df.iloc[:idx_sig + 1] if idx_sig is not None else df

                        # [v4.8] CHoCH = contre-tendance / BOS = dans le sens
                        is_bos = (
                            (trend == "Uptrend"   and "Bull" in sig) or
                            (trend == "Downtrend" and "Bear" in sig)
                        )
                        type_label = "BOS" if is_bos else "CHoCH"

                        results.append({
                            "Instrument":  inst.replace("_", "/"),
                            "Timeframe":   tf,
                            "Type":        type_label,
                            "Ordre":       "Achat" if "Bull" in sig else "Vente",
                            "Signal":      sig,
                            "Volatilité":  volatilite,
                            "Force":       strength or "Moyen",
                            "BB_Width":    compute_bb_width(df_sig),
                            "Statut":      compute_statut(time_sig, tf),
                            "Heure (UTC)": time_sig.strftime("%Y-%m-%d %H:%M"),
                        })

        if errors:
            st.warning(f"{len(errors)} erreur(s) silencieuse(s) : {'; '.join(errors[:5])}")

        if results:
            df_result = pd.DataFrame(results).sort_values("Heure (UTC)", ascending=False)
            st.session_state.df      = df_result
            st.session_state.png_buf = None
            st.success(f"Scan terminé – {len(df_result)} signaux sur {n_combos} combinaisons !")
        else:
            st.info("Aucun signal CHoCH/BOS récent détecté")

# ===================== AFFICHAGE =====================
if "df" in st.session_state:
    df = st.session_state.df.copy()
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    if st.session_state.get("png_buf") is None:
        st.session_state.png_buf = generate_png(df, DISPLAY_COLS)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "CSV",
            df[[c for c in DISPLAY_COLS if c in df.columns]].to_csv(index=False).encode(),
            f"choch_{ts}.csv", "text/csv"
        )
    with col2:
        st.download_button(
            "PNG",
            st.session_state.png_buf,
            f"choch_{ts}.png", "image/png"
        )
    with col3:
        st.download_button(
            "PDF",
            create_pdf(df[[c for c in DISPLAY_COLS if c in df.columns]]),
            f"choch_signaux_{ts}.pdf", "application/pdf"
        )

    def style_bb(val):
        if "Squeeze"   in str(val): return "color:#ff9800;font-weight:bold"
        if "Expansion" in str(val): return "color:#ab47bc;font-weight:bold"
        return "color:#90a4ae"

    st.dataframe(
        df[[c for c in DISPLAY_COLS if c in df.columns]].style
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
        .map(
            lambda x: "color:#00c853;font-weight:bold" if x == "Fresh"
            else "color:#ff9800;font-weight:bold" if x == "Aged"
            else "color:#ff5252;font-weight:bold" if x == "Stale" else "",
            subset=["Statut"]
        ),
        hide_index=True,
        use_container_width=True
    )
