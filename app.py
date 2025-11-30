# app.py - Version finale 100% fonctionnelle + PDF lisible par toutes les IA
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import dataframe_image as dfi

# ReportLab pour PDF texte
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

# ===================== CONFIG =====================
INSTRUMENTS_TO_SCAN = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF", "XAU_USD", "US30_USD", "NAS100_USD", "SPX500_USD"
]

VOLATILITY_LEVELS = {
    "EUR_USD": "Basse", "GBP_USD": "Basse", "USD_JPY": "Basse", "USD_CHF": "Basse",
    "USD_CAD": "Basse", "AUD_USD": "Moyenne", "NZD_USD": "Moyenne",
    "EUR_GBP": "Moyenne", "EUR_JPY": "Moyenne", "EUR_CHF": "Moyenne",
    "EUR_AUD": "Moyenne", "EUR_CAD": "Moyenne", "EUR_NZD": "Moyenne",
    "GBP_JPY": "Haute", "GBP_CHF": "Haute", "GBP_AUD": "Haute",
    "GBP_CAD": "Haute", "GBP_NZD": "Haute", "AUD_JPY": "Haute",
    "AUD_CAD": "Moyenne", "AUD_CHF": "Haute", "AUD_NZD": "Moyenne",
    "CAD_JPY": "Haute", "CAD_CHF": "Haute", "CHF_JPY": "Haute",
    "NZD_JPY": "Haute", "NZD_CAD": "Moyenne", "NZD_CHF": "Haute",
    "XAU_USD": "Très Haute", "US30_USD": "Très Haute", "NAS100_USD": "Très Haute", "SPX500_USD": "Très Haute"
}

TIME_FRAMES = {"H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"}
FRACTAL_LENGTHS = {"H1": 5, "H4": 6, "D1": 7, "Weekly": 8}
RECENT_BARS_THRESHOLD = 12
MAX_WORKERS = 6

# ===================== FONCTIONS =====================
def calculate_atr(df, period=14):
    if len(df) < period:
        return 0
    high = df['high'].values
    low = df['low'].values          # ← CORRIGÉ ICI !
    close = df['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    return np.mean(tr[-period:])

def get_oanda_data(instrument, granularity, count=300):
    params = {"count": count, "granularity": granularity}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    for _ in range(3):
        try:
            api.request(r)
            candles = [c for c in r.response['candles'] if c['complete']]
            if not candles:
                return None
            df = pd.DataFrame([{
                "time": pd.to_datetime(c['time']),
                "open": float(c['mid']['o']),
                "high": float(c['mid']['h']),
                "low": float(c['mid']['l']),
                "close": float(c['mid']['c'])
            } for c in candles])
            return df
        except:
            time.sleep(1)
    return None

def detect_choch(df, tf_code):
    length = FRACTAL_LENGTHS.get(tf_code, 5)
    if len(df) < length*3:
        return None, None, None, None, None
    p = length // 2
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    is_bull = np.full(len(df), False)
    is_bear = np.full(len(df), False)
    for i in range(p, len(df)-p):
        if h[i] == max(h[i-p:i+p+1]):
            is_bull[i] = True
        if l[i] == min(l[i-p:i+p+1]):
            is_bear[i] = True

    atr = calculate_atr(df)
    os = 0
    upper = lower = None

    for i in range(len(df)):
        if is_bull[i]:
            upper = h[i]
        if is_bear[i]:
            lower = l[i]

        if upper and c[i] > upper >= c[i-1] and os <= 0:
            strength = "Fort" if atr and (c[i]-upper) > atr*0.5 else "Moyen"
            if len(df)-1-i < RECENT_BARS_THRESHOLD:
                trend = "Haussier" if c[-1] > c[-20] else "Baissier" if c[-1] < c[-20] else "Neutre"
                return "Bullish CHoCH", df['time'].iloc[i], strength, trend, (df['high'].tail(50).max(), df['low'].tail(50).min())
            os = 1

        if lower and c[i] < lower <= c[i-1] and os >= 0:
            strength = "Fort" if atr and (lower-c[i]) > atr*0.5 else "Moyen"
            if len(df)-1-i < RECENT_BARS_THRESHOLD:
                trend = "Haussier" if c[-1] > c[-20] else "Baissier" if c[-1] < c[-20] else "Neutre"
                return "Bearish CHoCH", df['time'].iloc[i], strength, trend, (df['high'].tail(50).max(), df['low'].tail(50).min())
            os = -1
    return None, None, None, None, None

# ===================== PDF TEXTE PUR =====================
def generate_pdf(df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=15, rightMargin=15, topMargin=30, bottomMargin=30)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("Rapport des Signaux CHoCH", styles['Title']))
    elements.append(Paragraph(f"Généré le {datetime.utcnow().strftime('%d/%m/%Y à %H:%M UTC')}", styles['Normal']))
    elements.append(Spacer(1, 20))

    data = [df.columns.tolist()] + df.values.tolist()
    col_widths = [50, 35, 30, 60, 40, 35, 50, 50, 50, 70]

    for i in range(0, len(data), 38):
        if i > 0:
            elements.append(PageBreak())
        table = Table(data[i:i+38], colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.lightgrey]),
        ]))
        elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ===================== STREAMLIT =====================
st.set_page_config(page_title="CHoCH Scanner", layout="wide")
st.markdown("<h1 style='text-align:center;color:#1e3a8a;'>Scanner Change of Character (CHoCH)</h1>", unsafe_allow_html=True)

try:
    api = API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
except:
    st.error("OANDA_ACCESS_TOKEN manquant dans Secrets")
    st.stop()

if st.button("Lancer le Scan", type="primary", use_container_width=True):
    with st.spinner("Scan en cours..."):
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(get_oanda_data, inst, code): (inst, name)
                       for inst in INSTRUMENTS_TO_SCAN for name, code in TIME_FRAMES.items()}
            for future in as_completed(futures):
                inst, tf_name = futures[future]
                df = future.result()
                if df is not None:
                    signal, time_sig, strength, trend, levels = detect_choch(df, TIME_FRAMES[tf_name])
                    if signal:
                        results.append({
                            "Instrument": inst.replace("_", "/"),
                            "Timeframe": tf_name,
                            "Ordre": "Achat" if "Bullish" in signal else "Vente",
                            "Signal": signal,
                            "Volatilité": VOLATILITY_LEVELS.get(inst, "Moyenne"),
                            "Force": strength,
                            "Tendance": trend,
                            "Résistance": round(levels[0], 5) if levels else "-",
                            "Support": round(levels[1], 5) if levels else "-",
                            "Heure (UTC)": time_sig
                        })

        if results:
            df_result = pd.DataFrame(results).sort_values("Heure (UTC)", ascending=False)
            st.session_state.df = df_result
            st.success(f"Scan terminé — {len(df_result)} signaux trouvés !")
        else:
            st.info("Aucun signal CHoCH récent")

if "df" in st.session_state:
    df = st.session_state.df.copy()
    df['Heure (UTC)'] = pd.to_datetime(df['Heure (UTC)'])
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("CSV", df.to_csv(index=False).encode(), f"choch_{ts}.csv", "text/csv")
    with col2:
        buf_img = io.BytesIO()
        dfi.export(df, buf_img)
        buf_img.seek(0)
        st.download_button("PNG", buf_img, f"choch_{ts}.png", "image/png")
    with col3:
        pdf_buffer = generate_pdf(df)
        st.download_button(
            label="PDF (texte 100% lisible par IA)",
            data=pdf_buffer,
            file_name=f"choch_signaux_{ts}.pdf",
            mime="application/pdf"
        )

    # Affichage stylé
    def style_row(row):
        color = "#089981" if row.Ordre == "Achat" else "#f23645"
        return [f'background-color: {color}; color: white' if col == "Ordre" else '' for col in row.index]

    st.dataframe(df.style.apply(style_row, axis=1), use_container_width=True, hide_index=True)
