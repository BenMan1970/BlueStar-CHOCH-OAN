# Script complet CHoCH avec PDF, Streamlit et logique réparée
# (À compléter avec tes clés API et éléments visuels si besoin)

import streamlit as st
import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime
from fpdf import FPDF
import base64

# ----------------------- CONFIG -------------------------
API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"
client = Client(API_KEY, API_SECRET)

# ----------------------- PDF ----------------------------
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Rapport CHoCH', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def create_pdf(signal, symbol, time_sig, strength):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, f"Signal : {signal}", ln=True)
    pdf.cell(0, 10, f"Actif : {symbol}", ln=True)
    pdf.cell(0, 10, f"Heure : {time_sig}", ln=True)
    pdf.cell(0, 10, f"Force : {strength}", ln=True)

    filename = "signal.pdf"
    pdf.output(filename)
    return filename

# ----------------------- FRACTALES ----------------------
FRACTAL_LENGTHS_BY_TF = {
    "1m": 5, "3m": 5, "5m": 5, "15m": 5, "30m": 5,
    "1h": 7, "2h": 7, "4h": 7, "1d": 9
}

RECENT_BARS_THRESHOLD = 5

# ----------------------- ATR ----------------------------
def calculate_atr(df, period=14):
    if len(df) < period + 1:
        return None
    high = df['high']
    low = df['low']
    close = df['close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean().iloc[-1]

# ----------------------- CHoCH ---------------------------
def detect_choch(df, tf_code):
    if df is None or len(df) < 40:
        return None, None, None

    length = FRACTAL_LENGTHS_BY_TF.get(tf_code, 5)
    p = length // 2

    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    is_bull = np.zeros(len(df), bool)
    is_bear = np.zeros(len(df), bool)

    for i in range(p, len(df) - p):
        if high[i] == np.max(high[i-p:i+p+1]):
            is_bull[i] = True
        if low[i] == np.min(low[i-p:i+p+1]):
            is_bear[i] = True

    upper_fr = None
    lower_fr = None

    recent_close = close[-10:]
    order_state = 1 if recent_close[-1] > recent_close[0] else -1

    signal = None
    time_sig = None
    strength = None
    bar_idx = None

    atr = calculate_atr(df)

    for i in range(len(df)):
        if is_bull[i]:
            upper_fr = high[i]
        if is_bear[i]:
            lower_fr = low[i]

        if order_state == -1 and upper_fr is not None:
            if close[i] > upper_fr:
                bar_idx = i
                move = close[i] - upper_fr
                strength = "Fort" if atr and move > atr * 0.5 else "Moyen"
                signal = "Bullish CHoCH"
                time_sig = df["time"].iloc[i]
                order_state = 1

        if order_state == 1 and lower_fr is not None:
            if close[i] < lower_fr:
                bar_idx = i
                move = lower_fr - close[i]
                strength = "Fort" if atr and move > atr * 0.5 else "Moyen"
                signal = "Bearish CHoCH"
                time_sig = df["time"].iloc[i]
                order_state = -1

    if signal and bar_idx is not None and (len(df) - 1 - bar_idx) <= RECENT_BARS_THRESHOLD:
        return signal, time_sig, strength

    return None, None, None

# ----------------------- BINANCE DATA --------------------
def load_klines(symbol, interval, limit=500):
    raw = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(raw, columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote", "trades", "taker_base", "taker_quote", "ignore"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="ms")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df

# ----------------------- STREAMLIT UI --------------------
st.title("CHoCH Scanner avec PDF")

futures = [s['symbol'] for s in client.futures_exchange_info()['symbols']]

symbol = st.selectbox("Choisis une paire", futures)
interval = st.selectbox("Intervalle", ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"])

if st.button("Scanner"):
    df = load_klines(symbol, interval)
    signal, time_sig, strength = detect_choch(df, interval)

    if signal:
        st.success(f"Signal : {signal} | {strength} | {time_sig}")

        pdf_file = create_pdf(signal, symbol, time_sig, strength)
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="Télécharger PDF",
                data=f,
                file_name=pdf_file,
                mime="application/pdf"
            )
    else:
        st.info("Aucun signal récent.")



               
