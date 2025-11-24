# app.py → VERSION FINALE (LOGIQUE CHOCH RÉPARÉE)

import streamlit as st
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import time as time_module
from datetime import datetime
import io
from fpdf import FPDF
import dataframe_image as dfi
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import threading
import warnings

# ─── SUPPRESSION DES WARNINGS JAUNES ────────────────────────────────────
warnings.filterwarnings("ignore", category=SyntaxWarning, module="oandapyV20")

# ─── CONFIGURATION ──────────────────────────────────────────────────────
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
    "XAU_USD": "Très Haute", "US30_USD": "Très Haute",
    "NAS100_USD": "Très Haute", "SPX500_USD": "Très Haute"
}

TIME_FRAMES = {"H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"}
FRACTAL_LENGTHS_BY_TF = {"H1": 5, "H4": 6, "D": 7, "W": 8}

RECENT_BARS_THRESHOLD = 10
MAX_WORKERS = 5
REQUEST_DELAY = 0.09

lock = threading.Lock()

def delay():
    with lock:
        time_module.sleep(REQUEST_DELAY)

# ─── CHARGEMENT TOKEN ──────────────────────────────────────────────────
try:
    OANDA_ACCESS_TOKEN = st.secrets["OANDA_ACCESS_TOKEN"]
except:
    st.error("OANDA_ACCESS_TOKEN manquant dans Secrets")
    st.stop()

api_client = API(access_token=OANDA_ACCESS_TOKEN)

# ─── ATR ────────────────────────────────────────────────────────────────
def calculate_atr(df, period=14):
    if len(df) < period + 5:
        return None
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    tr = np.maximum(h-l, np.maximum(np.abs(h - np.roll(c,1)), np.abs(l - np.roll(c,1))))[1:]
    return np.mean(tr[-period:]) if len(tr) >= period else None

# ─── DONNÉES OANDA ─────────────────────────────────────────────────────
def get_oanda_data(instrument, granularity):
    params = {"count": 260, "granularity": granularity}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)

    for _ in range(3):
        try:
            delay()
            api_client.request(r)
            candles = r.response.get('candles', [])
            data = []
            for c in candles:
                if c['complete']:
                    m = c['mid']
                    data.append([
                        pd.to_datetime(c['time']),
                        float(m['o']), float(m['h']),
                        float(m['l']), float(m['c'])
                    ])
            if data:
                return pd.DataFrame(data, columns=['time','open','high','low','close'])
        except:
            time_module.sleep(1)

    return None

# ─── LOGIQUE CHOCH RÉPARÉE ─────────────────────────────────────────────
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

    # -------- Structure initiale --------
    if close[-1] > close[-10]:
        order_state = 1   # haussier
    else:
        order_state = -1  # baissier

    upper_fr = None
    lower_fr = None
    signal = None
    bar_idx = None
    time_sig = None

    atr = calculate_atr(df)

    for i in range(len(df)):
        if is_bull[i]:
            upper_fr = high[i]
        if is_bear[i]:
            lower_fr = low[i]

        # BULLISH CHoCH
        if order_state == -1 and upper_fr and close[i] > upper_fr:
            bar_idx = i
            move = close[i] - upper_fr
            strength = "Fort" if atr and move > atr * 0.5 else "Moyen"
            signal = "Bullish CHoCH"
            time_sig = df["time"].iloc[i]
            order_state = 1

        # BEARISH CHoCH
        if order_state == 1 and lower_fr and close[i] < lower_fr:
            bar_idx = i
            move = lower_fr - close[i]
            strength = "Fort" if atr and move > atr * 0.5 else "Moyen"
            signal = "Bearish CHoCH"
            time_sig = df["time"].iloc[i]
            order_state = -1

    if signal and bar_idx is not None and (len(df) - 1 - bar_idx) <= RECENT_BARS_THRESHOLD:
        return signal, time_sig, strength

    return None, None, None

# ─── SCAN PAR PAIRE ────────────────────────────────────────────────────
def scan_pair_tf(instrument, tf_name, tf_code):
    df = get_oanda_data(instrument, tf_code)
    if df is None:
        return {"error": f"Pas de données pour {instrument} {tf_name}"}

    sig, t, force = detect_choch(df, tf_code)

    if sig:
        return {
            "Instrument": instrument.replace("_", "/"),
            "Timeframe": tf_name,
            "Ordre": "Achat" if "Bullish" in sig else "Vente",
            "Signal": sig,
            "Volatilité": VOLATILITY_LEVELS.get(instrument, "Inconnue"),
            "Force": force or "Moyen",
            "Heure (UTC)": t
        }

    return None

# ─── STREAMLIT UI ──────────────────────────────────────────────────────
st.set_page_config(page_title="Scanner CHoCH", layout="wide")
st.markdown("<h1 style='text-align:center;'>Scanner Change of Character (CHoCH)</h1>", unsafe_allow_html=True)

if st.button("Lancer un nouveau Scan", type="primary"):
    st.session_state.clear()
    with st.spinner("Scan en cours..."):

        results = []
        failed = []
        total = len(INSTRUMENTS_TO_SCAN) * len(TIME_FRAMES)
        bar = st.progress(0)
        status = st.empty()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures


               
