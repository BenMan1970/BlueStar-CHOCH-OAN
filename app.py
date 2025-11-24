# app.py → VERSION FINALE 100% FONCTIONNELLE (24/11/2025)
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

# ─── SUPPRESSION DES WARNINGS JAUNES (Python 3.13+) ─────────────────────
warnings.filterwarnings("ignore", category=SyntaxWarning, module="oandapyV20")

# ─── CONFIGURATION (exactement comme ton premier code) ──────────────────
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
REQUEST_DELAY = 0.09  # safe OANDA

lock = threading.Lock()

def delay():
    with lock:
        time_module.sleep(REQUEST_DELAY)

# ─── ATR + DONNÉES ─────────────────────────────────────────────────────
def calculate_atr(df, period=14):
    if len(df) < period + 5: return None
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    tr = np.maximum(h-l, np.maximum(np.abs(h - np.roll(c,1)), np.abs(l - np.roll(c,1))))[1:]
    return np.mean(tr[-period:]) if len(tr) >= period else None

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
                    data.append([pd.to_datetime(c['time']), float(m['o']), float(m['h']), float(m['l']), float(m['c'])])
            if data:
                df = pd.DataFrame(data, columns=['time','open','high','low','close'])
                return df
        except:
            time_module.sleep(1)
    return None

# ─── DÉTECTION CHOCH (corrigée à 100%) ──────────────────────────────────
def detect_choch(df, tf_code):
    if df is None or len(df) < 30:
        return None, None, None

    length = FRACTAL_LENGTHS_BY_TF.get(tf_code, 5)
    p = length // 2
    if len(df) <= length + p:
        return None, None, None

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    atr = calculate_atr(df)

    # Fractales
    is_bull = np.zeros(len(df), bool)
    is_bear = np.zeros(len(df), bool)
    for i in range(p, len(df)-p):
        if high[i] == np.max(high[i-p:i+p+1]): is_bull[i] = True
        if low[i]  == np.min(low[i-p:i+p+1]):  is_bear[i] = True

    upper_fr = lower_fr = None
    order_state = 0
    signal = time_sig = strength = bar_idx = None

    for i in range(length, len(df)):
        if is_bull[i-p]: upper_fr = high[i-p]
        if is_bear[i-p]: lower_fr = low[i-p]

        # Bullish CHoCH
        if order_state == -1 and upper_fr is not None and close[i] > upper_fr >= close[i-1]:
            bar_idx = i
            move = close[i] - upper_fr
            strength = "Fort" if atr and move > atr*0.5 else "Moyen"
            signal = "Bullish CHoCH"
            time_sig = df['time'].iloc[i]
            order_state = 1

        # Bearish CHoCH
        if order_state == 1 and lower_fr is not None and close[i] < lower_fr <= close[i-1]:
            bar_idx = i
            move = lower_fr - close[i]
            strength = "Fort" if atr and move > atr*0.5 else "Moyen"
            signal = "Bearish CHoCH"
            time_sig = df['time'].iloc[i]
            order_state = -1

    if signal and bar_idx is not None and (len(df) - 1 - bar_idx) < RECENT_BARS_THRESHOLD:
        return signal, time_sig, strength

    return None, None, None

# ─── SCAN PAR PAIRE/TF ────────────────────────────────────────────────
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

# ─── STREAMLIT ────────────────────────────────────────────────────────
st.set_page_config(page_title="Scanner CHoCH", layout="wide")
st.markdown("<h1 style='text-align:center;'>Scanner Change of Character (CHoCH)</h1>", unsafe_allow_html=True)

# ─── CHARGEMENT TOKEN (exactement comme avant) ───────────────────────
try:
    OANDA_ACCESS_TOKEN = st.secrets["OANDA_ACCESS_TOKEN"]
except:
    st.error("OANDA_ACCESS_TOKEN manquant dans Secrets")
    st.stop()

api_client = API(access_token=OANDA_ACCESS_TOKEN)

if st.button("Lancer un nouveau Scan", type="primary"):
    st.session_state.clear()
    with st.spinner("Scan en cours..."):
        results = []
        failed = []
        total = len(INSTRUMENTS_TO_SCAN) * len(TIME_FRAMES)
        bar = st.progress(0)
        status = st.empty()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(scan_pair_tf, i, n, c): (i,n) 
                      for i in INSTRUMENTS_TO_SCAN 
                      for n,c in TIME_FRAMES.items()}

            done = 0
            for f in as_completed(futures):
                done += 1
                bar.progress(done/total)
                status.text(f"{done}/{total}")
                res = f.result()
                if isinstance(res, dict) and "error" in res:
                    failed.append(res["error"])
                elif res:
                    results.append(res)

        df_results = pd.DataFrame(results) if results else pd.DataFrame()
        st.session_state.results = df_results
        st.session_state.failed = failed

    st.rerun()

# ─── AFFICHAGE ───────────────────────────────────────────────────────
if "results" in st.session_state:
    df = st.session_state.results
    if df.empty:
        st.success("Aucun signal CHoCH récent détecté")
    else:
        df["Heure (UTC)"] = pd.to_datetime(df["Heure (UTC)"]).dt.strftime("%d/%m %H:%M")
        csv = df.to_csv(index=False).encode()
        st.download_button("Télécharger CSV", csv, "choch_signaux.csv", "text/csv")

        for tf in TIME_FRAMES:
            sub = df[df["Timeframe"] == tf]
            if sub.empty: continue
            sub = sub[["Instrument","Ordre","Signal","Volatilité","Force","Heure (UTC)"]].copy()
            st.subheader(f"--- {tf} ---")
            st.dataframe(sub.style
                .map(lambda x: "color:#089981;font-weight:bold" if "Bullish" in str(x) else "color:#f23645;font-weight:bold", subset=["Signal"])
                .map(lambda x: "background:#089981;color:white" if x=="Achat" else "background:#f23645;color:white", subset=["Ordre"]),
                hide_index=True, use_container_width=True)

    if st.session_state.get("failed"):
        with st.expander("Erreurs (voir détails)"):
            for e in st.session_state.failed:
                st.write(f"- {e}")
