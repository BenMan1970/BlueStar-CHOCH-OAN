# app.py → v4.4 — Audit complet (9 findings patched)
# F1 Force breakout fix | F2 bare except | F3 future.result() guard
# F4 is not None | F5 bb_width div/0 | F6 PNG cache | F7 utcnow | F8 timeout | F9 len guard
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
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","USD_CAD","AUD_USD","NZD_USD",
    "EUR_GBP","EUR_JPY","EUR_CHF","EUR_AUD","EUR_CAD","EUR_NZD",
    "GBP_JPY","GBP_CHF","GBP_AUD","GBP_CAD","GBP_NZD",
    "AUD_JPY","AUD_CAD","AUD_CHF","AUD_NZD","CAD_JPY","CAD_CHF","CHF_JPY",
    "NZD_JPY","NZD_CAD","NZD_CHF",
    "XAU_USD","XAG_USD","XPT_USD",
    "US30_USD","NAS100_USD","SPX500_USD","DE30_EUR",
]

VOLATILITY = {
    "EUR_USD":"Basse","GBP_USD":"Basse","USD_JPY":"Basse","USD_CHF":"Basse","USD_CAD":"Basse",
    "AUD_USD":"Moyenne","NZD_USD":"Moyenne","EUR_GBP":"Moyenne",
    "EUR_JPY":"Moyenne","EUR_CHF":"Moyenne","EUR_AUD":"Moyenne","EUR_CAD":"Moyenne","EUR_NZD":"Moyenne",
    "GBP_JPY":"Haute","GBP_CHF":"Haute","GBP_AUD":"Haute","GBP_CAD":"Haute","GBP_NZD":"Haute",
    "AUD_JPY":"Haute","AUD_CAD":"Moyenne","AUD_CHF":"Haute","AUD_NZD":"Moyenne",
    "CAD_JPY":"Haute","CAD_CHF":"Haute","CHF_JPY":"Haute","NZD_JPY":"Haute",
    "NZD_CAD":"Moyenne","NZD_CHF":"Haute",
    "XAU_USD":"Très Haute","XAG_USD":"Très Haute","XPT_USD":"Très Haute",
    "US30_USD":"Très Haute","NAS100_USD":"Très Haute","SPX500_USD":"Très Haute","DE30_EUR":"Très Haute",
}

TIMEFRAMES  = {"H1":"H1", "H4":"H4", "D1":"D", "Weekly":"W", "Monthly":"M"}
FRACTAL_LEN = {"H1":5, "H4":6, "D1":7, "Weekly":8, "Monthly":9}

# Durée d'une bougie par TF (en heures)
TF_HOURS = {"H1": 1, "H4": 4, "D1": 24, "Weekly": 168, "Monthly": 720}

# Seuils Fresh/Aged/Stale en nombre de bougies
TF_STATUT = {
    "H1":      {"Fresh": 3,  "Aged": 8},
    "H4":      {"Fresh": 2,  "Aged": 5},
    "D1":      {"Fresh": 2,  "Aged": 5},
    "Weekly":  {"Fresh": 2,  "Aged": 4},
    "Monthly": {"Fresh": 1,  "Aged": 2},
}

# ===================== API — F8: timeout 10s =====================
try:
    api = API(
        access_token=st.secrets["OANDA_ACCESS_TOKEN"],
        request_params={"timeout": 10}   # F8: évite les threads bloqués indéfiniment
    )
except Exception as e:
    st.error(f"Token OANDA manquant dans les secrets Streamlit : {e}")
    st.stop()

# ===================== FONCTIONS =====================
def get_candles(inst, gran):
    try:  # F2: except Exception au lieu de bare except
        count = 500 if gran == "M" else 300
        r = instruments.InstrumentsCandles(
            instrument=inst,
            params={"count": count, "granularity": gran}
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
    except Exception:  # F2
        return None


def calc_atr(df, period=14):
    """ATR réel (True Range) — utilisé pour get_force dans detect_choch."""
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.maximum(h[1:] - l[1:],
         np.maximum(np.abs(h[1:] - c[:-1]),
                    np.abs(l[1:] - c[:-1])))
    if len(tr) < period:
        return np.nan
    return float(np.mean(tr[-period:]))


def compute_bb_width(df, length=20, std=2):
    """
    BB Width % vs sa propre moyenne sur le TF du signal.
    Tags : "-32%_Squeeze" | "+41%_Expansion" | "5%_Normal"
    """
    close = df["close"]
    if len(close) < length * 2:
        return "N/A"
    sma     = close.rolling(length).mean()
    std_dev = close.rolling(length).std()
    upper   = sma + std * std_dev
    lower   = sma - std * std_dev
    bb_w    = (upper - lower) / sma
    bb_avg  = bb_w.rolling(length).mean()

    # F5: guard division par zéro
    avg_last = bb_avg.iloc[-1]
    if pd.isna(avg_last) or avg_last == 0:
        return "N/A"

    pct  = ((bb_w - bb_avg) / bb_avg * 100).iloc[-1]
    if pd.isna(pct):
        return "N/A"
    sign = "+" if pct >= 0 else ""
    if pct < -25:
        return f"{sign}{pct:.0f}%_Squeeze"
    if pct > 25:
        return f"{sign}{pct:.0f}%_Expansion"
    return f"{sign}{pct:.0f}%_Normal"


def compute_statut(time_sig, tf):
    """Fresh / Aged / Stale selon le nb de bougies écoulées depuis le signal."""
    try:  # F2
        now     = datetime.now(timezone.utc)
        sig_utc = time_sig.replace(tzinfo=timezone.utc) if time_sig.tzinfo is None else time_sig
        elapsed_h      = (now - sig_utc).total_seconds() / 3600
        candles_elapsed = elapsed_h / TF_HOURS.get(tf, 1)
        thresholds      = TF_STATUT.get(tf, {"Fresh": 2, "Aged": 5})
        if candles_elapsed <= thresholds["Fresh"]:
            return "Fresh"
        if candles_elapsed <= thresholds["Aged"]:
            return "Aged"
        return "Stale"
    except Exception:  # F2
        return "N/A"


def detect_choch(df, tf):
    length = FRACTAL_LEN.get(tf, 5)
    p      = length // 2
    h, l, c, o = (
        df["high"].values, df["low"].values,
        df["close"].values, df["open"].values
    )

    # F9: guard longueur minimale
    if len(c) < max(p * 2 + 1, 3):
        return None, None, None

    last_high = None
    last_low  = None
    for i in range(p, len(df) - p):
        if h[i] == max(h[i-p:i+p+1]):
            last_high = h[i]
        if l[i] == min(l[i-p:i+p+1]):
            last_low = l[i]

    atr = calc_atr(df)

    def get_force(candle_range):
        """
        F1 FIX: on mesure le range complet de la bougie de breakout (h-l)
        plutôt que la distance marginale close vs fractal.
        Un candle_range ≈ 1× ATR = normal → Moyen.
        """
        if np.isnan(atr) or atr == 0:
            return "Moyen"
        ratio = candle_range / atr
        if tf == "H1":
            return "Fort" if ratio > 1.2 else ("Faible" if ratio < 0.6 else "Moyen")
        elif tf in ("Weekly", "Monthly"):
            return "Fort" if ratio > 1.5 else ("Faible" if ratio < 0.8 else "Moyen")
        else:  # H4, D1
            return "Fort" if ratio > 1.3 else ("Faible" if ratio < 0.7 else "Moyen")

    # F1 FIX: breakout_range = range de la bougie de breakout
    breakout_range = h[-1] - l[-1]

    # F4 FIX: is not None au lieu de falsy check
    if last_high is not None and c[-1] > last_high and c[-2] <= last_high:
        return "Bullish CHoCH", df.index[-1], get_force(breakout_range)

    if last_low is not None and c[-1] < last_low and c[-2] >= last_low:
        return "Bearish CHoCH", df.index[-1], get_force(breakout_range)

    return None, None, None


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
    # F7 FIX: datetime.now(timezone.utc) au lieu de utcnow()
    elements.append(Paragraph(
        f"Généré le {datetime.now(timezone.utc).strftime('%d/%m/%Y à %H:%M')} UTC",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 20))

    data       = [df.columns.tolist()] + df.values.tolist()
    col_widths = [80, 60, 55, 100, 75, 60, 110, 60, 125]  # 9 colonnes

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


# ===================== PNG — F6: généré une seule fois =====================
def generate_png(df, display_cols):
    """Génère le PNG une fois et le met en cache dans session_state."""
    fig, ax = plt.subplots(figsize=(18, max(5, len(df) * 0.35)))
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
    "Scanner Change of Character (CHoCH)</h1>",
    unsafe_allow_html=True
)

DISPLAY_COLS = [
    "Instrument", "Timeframe", "Ordre", "Signal",
    "Volatilité", "Force", "BB_Width", "Statut", "Heure (UTC)"
]

if st.button("Lancer le Scan", type="primary", use_container_width=True):
    with st.spinner("Scan en cours sur 175 combinaisons…"):
        results  = []
        errors   = []

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {
                executor.submit(get_candles, inst, code): (inst, name)
                for inst in INSTRUMENTS
                for name, code in TIMEFRAMES.items()
            }
            for future in as_completed(futures):
                inst, tf = futures[future]
                try:  # F3: wrapper future.result()
                    df = future.result()
                except Exception as e:
                    errors.append(f"{inst}/{tf}: {e}")
                    continue

                if df is not None:
                    sig, time_sig, strength = detect_choch(df, tf)
                    if sig:
                        results.append({
                            "Instrument":  inst.replace("_", "/"),
                            "Timeframe":   tf,
                            "Ordre":       "Achat" if "Bull" in sig else "Vente",
                            "Signal":      sig,
                            "Volatilité":  VOLATILITY.get(inst, "Moyenne"),
                            "Force":       strength or "Moyen",
                            "BB_Width":    compute_bb_width(df),
                            "Statut":      compute_statut(time_sig, tf),
                            "Heure (UTC)": time_sig.strftime("%Y-%m-%d %H:%M"),
                        })

        if errors:
            st.warning(f"{len(errors)} erreur(s) silencieuse(s) : {'; '.join(errors[:5])}")

        if results:
            df_result = pd.DataFrame(results).sort_values("Heure (UTC)", ascending=False)
            st.session_state.df      = df_result
            st.session_state.png_buf = None  # F6: invalider le cache PNG au nouveau scan
            st.success(f"Scan terminé – {len(df_result)} signaux sur 175 combinaisons !")
        else:
            st.info("Aucun signal CHoCH récent détecté")

# ===================== AFFICHAGE =====================
if "df" in st.session_state:
    df = st.session_state.df.copy()
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    # F6: PNG généré une seule fois, mis en cache dans session_state
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

    # Tableau avec couleurs
    def style_bb(val):
        if "Squeeze"   in str(val): return "color:#ff9800;font-weight:bold"
        if "Expansion" in str(val): return "color:#ab47bc;font-weight:bold"
        return "color:#90a4ae"

    st.dataframe(
        df[[c for c in DISPLAY_COLS if c in df.columns]].style
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
