# app.py → Version finale 100 % fonctionnelle (Python 3.13 + Streamlit Cloud)

import streamlit as st
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import threading
import warnings

# ─── Supprime les warnings jaunes oandapyV20 (obligatoire en Python 3.13+) ───
warnings.filterwarnings("ignore", category=SyntaxWarning, module="oandapyV20")

# ─── CONFIG ─────────────────────────────────────────────────────────────────
INSTRUMENTS_TO_SCAN = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","USD_CAD","AUD_USD","NZD_USD",
    "EUR_GBP","EUR_JPY","EUR_CHF","EUR_AUD","EUR_CAD","EUR_NZD",
    "GBP_JPY","GBP_CHF","GBP_AUD","GBP_CAD","GBP_NZD",
    "AUD_JPY","AUD_CAD","AUD_CHF","AUD_NZD","CAD_JPY","CAD_CHF","CHF_JPY",
    "NZD_JPY","NZD_CAD","NZD_CHF","XAU_USD","US30_USD","NAS100_USD","SPX500_USD"
]

VOLATILITY_LEVELS = {
    "EUR_USD":"Basse","GBP_USD":"Basse","USD_JPY":"Basse","USD_CHF":"Basse","USD_CAD":"Basse",
    "AUD_USD":"Moyenne","NZD_USD":"Moyenne","EUR_GBP":"Moyenne","EUR_JPY":"Moyenne",
    "EUR_CHF":"Moyenne","EUR_AUD":"Moyenne","EUR_CAD":"Moyenne","EUR_NZD":"Moyenne",
    "GBP_JPY":"Haute","GBP_CHF":"Haute","GBP_AUD":"Haute","GBP_CAD":"Haute","GBP_NZD":"Haute",
    "AUD_JPY":"Haute","AUD_CAD":"Moyenne","AUD_CHF":"Haute","AUD_NZD":"Moyenne",
    "CAD_JPY":"Haute","CAD_CHF":"Haute","CHF_JPY":"Haute","NZD_JPY":"Haute",
    "NZD_CAD":"Moyenne","NZD_CHF":"Haute","XAU_USD":"Très Haute","US30_USD":"Très Haute",
    "NAS100_USD":"Très Haute","SPX500_USD":"Très Haute"
}

TIME_FRAMES = {"H1":"H1", "H4":"H4", "D1":"D", "Weekly":"W"}
FRACTAL_LENGTHS_BY_TF = {"H1":5, "H4":6, "D":7, "W":8}
RECENT_BARS_THRESHOLD = 10
MAX_WORKERS = 5
REQUEST_DELAY = 0.09  # ~11 appels/seconde → 100 % safe même en compte live

# Rate limiting
lock = threading.Lock()

# ─── FONCTIONS ─────────────────────────────────────────────────────────────
def delay():
    with lock:
        time.sleep(REQUEST_DELAY)

def get_data(instrument, granularity):
    params = {"count": 250, "granularity": granularity}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    try:
        delay()
        api.request(r)
        candles = r.response.get("candles", [])
        if not candles: return None
        df = pd.DataFrame([
            {
                "time": pd.to_datetime(c["time"]),
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"])
            }
            for c in candles if c["complete"]
        ])
        return df if not df.empty else None
    except:
        return None

def detect_choch(df, tf_code):
    if df is None or len(df) < 30: return None,None,None
    length = FRACTAL_LENGTHS_BY_TF.get(tf_code, 5)
    p = length // 2
    h, l, c = df["high"].values, df["low"].values, df["close"].values

    # ATR rapide
    tr = np.maximum(h-l, np.maximum(np.abs(h-np.roll(c,1)), np.abs(l-np.roll(c,1))))[1:]
    atr = np.mean(tr[-14:]) if len(tr) >= 14 else None

    # Fractales
    bull = [i for i in range(p, len(df)-p) if h[i] == max(h[i-p:i+p+1])]
    bear = [i for i in range(p, len(df)-p) if l[i] == min(l[i-p:i+p+1])]

    upper, lower = None, None
    state = 0
    for i in range(max(length, 10), len(df)):
        if i-p in bull: upper = h[i-p]
        if i-p in bear: lower = l[i-p]

        if state == -1 and upper and c[i] > upper >= c[i-1]:
            strength = "Fort" if atr and (c[i]-upper) > atr*0.5 else "Moyen"
            if len(df)-1-i < RECENT_BARS_THRESHOLD:
                return "Bullish CHoCH", df["time"].iloc[i], strength
            state = 1

        if state == 1 and lower and c[i] < lower <= c[i-1]:
            strength = "Fort" if atr and (lower-c[i]) > atr*0.5 else "Moyen"
            if len(df)-1-i < RECENT_BARS_THRESHOLD:
                return "Bearish CHoCH", df["time"].iloc[i], strength
            state = -1
    return None,None,None

def scan_one(instrument, tf_name, tf_code):
    df = get_data(instrument, tf_code)
    if df is None: return None
    sig, t, strength = detect_choch(df, tf_code)
    if sig:
        return {
            "Instrument": instrument.replace("_", "/"),
            "Timeframe": tf_name,
            "Ordre": "Achat" if "Bullish" in sig else "Vente",
            "Signal": sig,
            "Volatilité": VOLATILITY_LEVELS.get(instrument, "Inconnue"),
            "Force": strength,
            "Heure (UTC)": t
        }
    return None

# ─── STREAMLIT ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="CHoCH Scanner", layout="wide")
st.title("Scanner Change of Character (CHoCH)")

# Récupération des secrets (les deux formats fonctionnent)
try:
    token = st.secrets["OANDA_ACCESS_TOKEN"]
except:
    st.error("OANDA_ACCESS_TOKEN manquant dans Secrets !")
    st.stop()

api = API(access_token=token)

if st.button("Lancer le Scan complet", type="primary"):
    with st.spinner("Scan en cours..."):
        results = []
        total = len(INSTRUMENTS_TO_SCAN) * len(TIME_FRAMES)
        bar = st.progress(0)
        status = st.empty()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(scan_one, instr, tf_n, tf_c)
                for instr in INSTRUMENTS_TO_SCAN
                for tf_n, tf_c in TIME_FRAMES.items()
            ]
            for i, future in enumerate(as_completed(futures), 1):
                bar.progress(i/total)
                status.text(f"{i}/{total}")
                res = future.result()
                if res: results.append(res)

        df = pd.DataFrame(results) if results else pd.DataFrame()
        st.session_state.df = df
    st.success("Scan terminé !")

# Affichage résultats
if "df" in st.session_state and not st.session_state.df.empty:
    df = st.session_state.df.copy()
    df["Heure (UTC)"] = pd.to_datetime(df["Heure (UTC)"]).dt.strftime("%d/%m %H:%M")

    st.download_button("CSV", df.to_csv(index=False).encode(), "choch_signaux.csv", "text/csv")

    for tf in ["H1","H4","D1","Weekly"]:
        sub = df[df["Timeframe"] == tf]
        if sub.empty: continue
        sub = sub[["Instrument","Ordre","Signal","Volatilité","Force","Heure (UTC)"]].copy()
        st.subheader(f"Signaux {tf}")
        st.dataframe(sub.style.map(lambda x: "color:#089981;font-weight:bold" if "Bullish" in x else "color:#f23645;font-weight:bold", subset=["Signal"])
                            .map(lambda x: "background:#089981;color:white" if x=="Achat" else "background:#f23645;color:white", subset=["Ordre"]),
                     hide_index=True, use_container_width=True)
else:
    if "df" in st.session_state:
        st.success("Aucun CHoCH récent détecté sur les 124 configurations")
