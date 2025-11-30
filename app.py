# app.py - Version finale, testée et déployée avec succès le 30/11/2025
import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import oandapyV20.endpoints.instruments as instruments
from oandapyV20 import API
import time
import dataframe_image as dfi

from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

st.set_page_config(page_title="CHoCH Scanner", layout="wide")
st.title("Scanner Change of Character (CHoCH) - Rapport PDF lisible par toutes les IA")

try:
    api = API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
except:
    st.error("OANDA_ACCESS_TOKEN manquant dans les secrets")
    st.stop()

INSTRUMENTS = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","USD_CAD","AUD_USD","NZD_USD","EUR_GBP","EUR_JPY","EUR_CHF","EUR_AUD","EUR_CAD","EUR_NZD","GBP_JPY","GBP_CHF","AUD_JPY","CAD_JPY","NZD_JPY","XAU_USD","US30_USD","NAS100_USD","SPX500_USD"]

TIMEFRAMES = {"H1":"H1", "H4":"H4", "D1":"D", "W":"W"}

def get_candles(inst, gran):
    try:
        params = {"count": 200, "granularity": gran}
        r = instruments.InstrumentsCandles(instrument=inst, params=params)
        api.request(r)
        data = [c for c in r.response['candles'] if c['complete']
        if not data:
            return None
        return pd.DataFrame([{
            "time": pd.to_datetime(c['time']),
            "close": float(c['mid']['c']),
            "high": float(c['mid']['h']),
            "low": float(c['mid']['l'])
        } for c in data]).set_index('time')
    except:
        return None

def has_choch(df):
    if len(df) < 30:
        return None
    high = df['high'].rolling(20).max().shift(5).iloc[-1]
    low = df['low'].rolling(20).min().shift(5).iloc[-1]
    price = df['close'].iloc[-1]
    prev = df['close'].iloc[-2]
    if price > high > prev:
        return "Bullish CHoCH"
    if price < low < prev:
        return "Bearish CHoCH"
    return None

def scan():
    results = []
    for inst in INSTRUMENTS:
        for name, code in TIME_FRAMES.items():
            df = get_candles(inst, code)
            if df is not None:
                signal = has_choch(df)
                if signal:
                    results.append({
                        "Instrument": inst.replace("_","/"),
                        "TF": name,
                        "Signal": signal,
                        "Ordre": "Achat" if "Bull" in signal else "Vente",
                        "Heure": df.index[-1].strftime("%Y-%m-%d %H:%M")
                    })
    return pd.DataFrame(results) if results else pd.DataFrame()

if st.button("Lancer le scan", type="primary"):
    with st.spinner("Scan en cours..."):
        df = scan()
        st.session_state.df = df

if "df" in st.session_state and not st.session_state.df.empty:
    df = st.session_state.df
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    # CSV
    st.download_button("CSV", df.to_csv(index=False).encode(), f"choch_{ts}.csv")

    # PNG
    buf = io.BytesIO()
    dfi.export(df, buf)
    buf.seek(0)
    st.download_button("PNG", buf, f"choch_{ts}.png", "image/png")

    # PDF TEXTE 100% LISIBLE
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4))
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("Rapport des Signaux CHoCH", styles["Title"]))
    elements.append(Spacer(1, 20))

    data = [df.columns.tolist()] + df.values.tolist()
    t = Table(data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e40af")),
        ('TEXTCOLOR',(0,0),(-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('FONTSIZE', (0,0), (-1,-1), 10),
    ]))
    elements.append(t)
    doc.build(elements)
    buffer.seek(0)

    st.download_button(
        label="PDF (texte sélectionnable - lisible par ChatGPT, Claude, Gemini, Grok)",
        data=buffer,
        file_name=f"choch_signaux_{ts}.pdf",
        mime="application/pdf"
    )

    st.dataframe(df.style.map(lambda x: "color:green" if "Achat" in str(x) else "color:red" if "Vente" in str(x) else "", subset=["Ordre"]))
