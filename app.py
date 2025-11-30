# app.py
import streamlit as st
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.exceptions
import time as time_module
from datetime import datetime
import io
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import dataframe_image as dfi

# --- ReportLab pour PDF texte ---
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

# --- CONFIGURATION ---
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
    "GBP_CAD": "Haute", "GBP_NZD": "Haute",
    "AUD_JPY": "Haute", "AUD_CAD": "Moyenne", "AUD_CHF": "Haute",
    "AUD_NZD": "Moyenne", "CAD_JPY": "Haute", "CAD_CHF": "Haute",
    "CHF_JPY": "Haute", "NZD_JPY": "Haute", "NZD_CAD": "Moyenne", "NZD_CHF": "Haute",
    "XAU_USD": "Très Haute", "US30_USD": "Très Haute", "NAS100_USD": "Très Haute", "SPX500_USD": "Très Haute"
}

TIME_FRAMES = {"H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"}
FRACTAL_LENGTHS_BY_TF = {"H1": 5, "H4": 6, "D1": 7, "Weekly": 8}
RECENT_BARS_THRESHOLD = 10
MAX_WORKERS = 6

# --- FONCTIONS ---
def calculate_atr(df, period=14):
    if df is None or len(df) < period:
        return None
    high = df['high'].values
    low = df['low().values
    close = df['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate(([close[0]], close[:-1])))
    tr3 = np.abs(low - np.concatenate(([close[0]], close[:-1])))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    return np.mean(tr[-period:])

def get_oanda_data(api_client, instrument, granularity, count=300):
    params = {"count": count, "granularity": granularity}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    for _ in range(3):
        try:
            api_client.request(r)
            data = r.response.get('candles', [])
            if not data:
                return None, "Aucune donnée"
            df = pd.DataFrame([{
                "time": pd.to_datetime(c['time']),
                "open": float(c['mid']['o']),
                "high": float(c['mid']['h']),
                "low": float(c['mid']['l']),
                "close": float(c['mid']['c'])
            } for c in data if c['complete']])
            return df, "Succès"
        except Exception as e:
            time_module.sleep(2)
    return None, str(e)

def get_trend_and_levels(df):
    if len(df) < 50:
        return "Neutre", None, None
    closes = df['close'].values
    ma_short = np.mean(closes[-20:])
    ma_long = np.mean(closes[-50:])
    if ma_short > ma_long * 1.002:
        trend = "Haussier"
    elif ma_short < ma_long * 0.998:
        trend = "Baissier"
    else:
        trend = "Neutre"
    return trend, df['high'].tail(50).max(), df['low'].tail(50).min()

def detect_choch_optimized(df, instrument, tf_code):
    length = FRACTAL_LENGTHS_BY_TF.get(tf_code, 5)
    if len(df) < length + 10:
        return None, None, None, None, None
    p = length // 2
    highs, lows, closes = df['high'].values, df['low'].values, df['close'].values

    is_bull_fractal = np.zeros(len(df), dtype=bool)
    is_bear_fractal = np.zeros(len(df), dtype=bool)
    for i in range(p, len(df) - p):
        if highs[i] == highs[i-p:i+p+1].max():
            is_bull_fractal[i] = True
        if lows[i] == lows[i-p:i+p+1].min():
            is_bear_fractal[i] = True

    atr = calculate_atr(df)
    trend, res, sup = get_trend_and_levels(df)
    upper = lower = None
    os = 0

    for i in range(length, len(df)):
        if is_bull_fractal[i-p]:
            upper = highs[i-p]
        if is_bear_fractal[i-p]:
            lower = lows[i-p]

        if upper and closes[i] > upper >= closes[i-1] and os == -1:
            strength = "Fort" if atr and (closes[i] - upper) > atr * 0.5 else "Moyen"
            if (len(df) - i) < RECENT_BARS_THRESHOLD:
                return "Bullish CHoCH", df['time'].iloc[i], strength, trend, (res, sup)
            os = 1
        if lower and closes[i] < lower <= closes[i-1] and os == 1:
            strength = "Fort" if atr and (lower - closes[i]) > atr * 0.5 else "Moyen"
            if (len(df) - i) < RECENT_BARS_THRESHOLD:
                return "Bearish CHoCH", df['time'].iloc[i], strength, trend, (res, sup)
            os = -1
    return None, None, None, None, None

def scan_instrument_timeframe(api_client, instrument, tf_name, tf_code):
    df, msg = get_oanda_data(api_client, instrument, tf_code)
    if df is not None:
        signal, time_sig, strength, trend, levels = detect_choch_optimized(df, instrument, tf_code)
        if signal:
            return {
                "Instrument": instrument.replace("_", "/"),
                "Timeframe": tf_name,
                "Ordre": "Achat" if "Bullish" in signal else "Vente",
                "Signal": signal,
                "Volatilité": VOLATILITY_LEVELS.get(instrument, "Inconnue"),
                "Force": strength or "Moyen",
                "Tendance": trend,
                "Résistance": round(levels[0], 5) if levels and levels[0] else "-",
                "Support": round(levels[1], 5) if levels and levels[1] else "-",
                "Heure (UTC)": time_sig
            }
    return None

# --- PDF TEXTE 100% LISIBLE ---
def generate_pdf_text(df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=12*mm, leftMargin=12*mm, topMargin=15*mm, bottomMargin=15*mm)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Rapport des Signaux CHoCH", styles['Title']))
    elements.append(Paragraph(f"Généré le {datetime.utcnow():%d/%m/%Y à %H:%M UTC}", styles['Normal']))
    elements.append(Spacer(1, 12))

    data = [df.columns.tolist()] + df.values.tolist()
    col_widths = [42*mm, 22*mm, 20*mm, 42*mm, 25*mm, 22*mm, 32*mm, 28*mm, 28*mm, 48*mm]

    for i in range(0, len(data), 35):
        if i > 0:
            elements.append(PageBreak())
        table = Table(data[i:i+35], colWidths=col_widths, repeatRows=1)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e40af")),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8.5),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.beige]),
        ]))
        elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- STREAMLIT APP ---
def main():
    st.set_page_config(page_title="Scanner CHoCH", layout="wide")
    st.markdown("<h1 style='text-align: center; color: #1e40af;'>Scanner Change of Character (CHoCH)</h1>", unsafe_allow_html=True)

    try:
        OANDA_ACCESS_TOKEN = st.secrets["OANDA_ACCESS_TOKEN"]
    except:
        st.error("OANDA_ACCESS_TOKEN manquant dans Secrets")
        st.stop()

    if st.button("Lancer le Scan CHoCH", type="primary", use_container_width=True):
        with st.spinner("Scan en cours sur 124 timeframes..."):
            api = API(access_token=OANDA_ACCESS_TOKEN)
            results = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = [executor.submit(scan_instrument_timeframe, api, inst, tf_name, tf_code)
                           for inst in INSTRUMENTS_TO_SCAN for tf_name, tf_code in TIME_FRAMES.items()]
                for future in as_completed(futures):
                    r = future.result()
                    if r:
                        results.append(r)

            if results:
                df = pd.DataFrame(results)
                df = df.sort_values(by="Heure (UTC)", ascending=False)
                st.session_state.df = df
            else:
                st.info("Aucun signal CHoCH récent détecté")
                st.stop()

    if 'df' in st.session_state:
        df = st.session_state.df.copy()
        df['Heure (UTC)'] = pd.to_datetime(df['Heure (UTC)'])

        st.success(f"{len(df)} signaux CHoCH détectés !")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        col1, col2, col3 = st.columns(3)
        with col1:
            csv = df.to_csv(index=False).encode()
            st.download_button("CSV", csv, f"choch_{timestamp}.csv", "text/csv")
        with col2:
            buf = io.BytesIO()
            dfi.export(df, buf, table_conversion="matplotlib")
            buf.seek(0)
            st.download_button("PNG", buf, f"choch_{timestamp}.png", "image/png")
        with col3:
            pdf_buffer = generate_pdf_text(df)
            st.download_button("PDF (texte lisible IA)", pdf_buffer, f"choch_{timestamp}.pdf", "application/pdf")

        for tf in ["Weekly", "D1", "H4", "H1"]:
            subset = df[df['Timeframe'] == tf]
            if not subset.empty:
                st.subheader(f"Signaux {tf}")
                styled = subset.style.map(lambda x: "color: green; font-weight:bold" if "Achat" in str(x) else "color: red; font-weight:bold", subset=['Ordre'])
                st.dataframe(styled, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
                    
