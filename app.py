# app.py
import streamlit as st
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import time
import time as time_module
from datetime import datetime
import io
import dataframe_image as dfi
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import threading
import warnings

# ─── Supprime les warnings jaunes oandapyV20 (Python 3.13+) ─────────────────────
warnings.filterwarnings("ignore", category=SyntaxWarning, module="oandapyV20")

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
INSTRUMENTS_TO_SCAN = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","USD_CAD","AUD_USD","NZD_USD",
    "EUR_GBP","EUR_JPY","EUR_CHF","EUR_AUD","EUR_CAD","EUR_NZD",
    "GBP_JPY","GBP_CHF","GBP_AUD","GBP_CAD","GBP_NZD",
    "AUD_JPY","AUD_CAD","AUD_CHF","AUD_NZD","CAD_JPY","CAD_CHF","CHF_JPY",
    "NZD_JPY","NZD_CAD","NZD_CHF",
    "XAU_USD","US30_USD","NAS100_USD","SPX500_USD"
]

VOLATILITY_LEVELS = {
    "EUR_USD":"Basse","GBP_USD":"Basse","USD_JPY":"Basse","USD_CHF":"Basse",
    "USD_CAD":"Basse","AUD_USD":"Moyenne","NZD_USD":"Moyenne",
    "EUR_GBP":"Moyenne","EUR_JPY":"Moyenne","EUR_CHF":"Moyenne",
    "EUR_AUD":"Moyenne","EUR_CAD":"Moyenne","EUR_NZD":"Moyenne",
    "GBP_JPY":"Haute","GBP_CHF":"Haute","GBP_AUD":"Haute",
    "GBP_CAD":"Haute","GBP_NZD":"Haute",
    "AUD_JPY":"Haute","AUD_CAD":"Moyenne","AUD_CHF":"Haute",
    "AUD_NZD":"Moyenne","CAD_JPY":"Haute","CAD_CHF":"Haute",
    "CHF_JPY":"Haute","NZD_JPY":"Haute","NZD_CAD":"Moyenne","NZD_CHF":"Haute",
    "XAU_USD":"Très Haute","US30_USD":"Très Haute",
    "NAS100_USD":"Très Haute","SPX500_USD":"Très Haute"
}

TIME_FRAMES = {"H1":"H1", "H4":"H4", "D1":"D", "Weekly":"W"}
FRACTAL_LENGTHS_BY_TF = {"H1":5, "H4":6, "D":7, "W":8}
RECENT_BARS_THRESHOLD = 10
MAX_WORKERS = 5

# Rate limiting safe
RATE_LIMIT_LOCK = threading.Lock()
REQUEST_DELAY = 0.085

# ─── FONCTIONS ─────────────────────────────────────────────────────────────────
def rate_limited_sleep():
    with RATE_LIMIT_LOCK:
        time.sleep(REQUEST_DELAY)

def calculate_atr(df, period=14):
    if df is None or len(df) < period + 5:
        return None
    high = df['high'].values
    low  = df['low'].values
    close = df['close'].values
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close,1)),
                               np.abs(low  - np.roll(close,1))))[1:]
    return np.mean(tr[-period:]) if len(tr) >= period else None

def get_oanda_data(api_client, instrument, granularity, count=250):
    params = {"count": count, "granularity": granularity}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    for _ in range(3):
        try:
            rate_limited_sleep()
            api_client.request(r)
            candles = r.response.get('candles', [])
            if not candles:
                return None, "Aucune donnée"

            times, o, h, l, c = [], [], [], [], []
            for candle in candles:
                if candle['complete']:
                    times.append(pd.to_datetime(candle['time']))
                    mid = candle['mid']
                    o.append(float(mid['o']))
                    h.append(float(mid['h']))
                    l.append(float(mid['l']))
                    c.append(float(mid['c']))
            if not times:
                return None, "Aucune bougie complète"
            df = pd.DataFrame({"time":times, "open":o, "high":h, "low":l, "close":c})
            return df, "OK"
        except Exception as e:
            time_module.sleep(1)
    return None, f"Erreur API : {e}"

def detect_choch_optimized(df, tf_code):
    if df is None or len(df) < 30:
        return None, None, None

    length = FRACTAL_LENGTHS_BY_TF.get(tf_code, 5)
    p = length // 2
    if len(df) <= length + p:
        return None, None, None

    high = df['high'].values
    low  = df['low'].values
    close = df['close'].values
    atr = calculate_atr(df)

    # Détection fractales
    bull_fr = np.zeros(len(df), bool)
    bear_fr = np.zeros(len(df), bool)
    for i in range(p, len(df)-p):
        if high[i] == np.max(high[i-p:i+p+1]): bull_fr[i] = True
        if low[i]  == np.min(low[i-p:i+p+1]):  bear_fr[i] = True

    upper = {'value':None, 'crossed':True}
    lower = {'value':None, 'crossed':True}
    order_state = 0
    signal = time_sig = strength = bar_idx = None

    for i in range(length, len(df)):
        if bull_fr[i-p]: upper = {'value':high[i-p], 'crossed':False}
        if bear_fr[i-p]: lower = {'value':low[i-p],  'crossed':False}

        if (upper['value'] is not None and not upper['crossed'] and
            close[i] > upper['value'] and close[i-1] <= upper['value'] and order_state == -1):
            signal = "Bullish CHoCH"
            time_sig = df['time'].iloc[i]
            bar_idx = i
            strength = "Fort" if atr and (close[i]-upper['value']) > atr*0.5 else "Moyen"
            order_state = 1
            upper['crossed'] = True

        if (lower['value'] is not None and not lower['crossed'] and
            close[i] < lower['value'] and close[i-1] >= lower['value'] and order_state == 1):
            signal = "Bearish CHoCH"
            time_sig = df['time'].iloc[i]
            bar_idx = i
            strength = "Fort" if atr and (lower['value']-close[i]) > atr*0.5 else "Moyen"
            order_state = -1
            lower['crossed'] = True

    if signal and bar_idx is not None and (len(df)-1-bar_idx) < RECENT_BARS_THRESHOLD:
        return signal, time_sig, strength

    return None, None, None

def scan_pair_tf(api_client, instrument, tf_name, tf_code):
    df, msg = get_oanda_data(api_client, instrument, tf_code)
    if df is None:
        return {"error":True, "instrument":instrument, "tf":tf_name, "message":msg}

    sig, t, strength = detect_choch_optimized(df, tf_code)
    if sig:
        return {
            "Instrument": instrument.replace("_","/"),
            "Timeframe": tf_name,
            "Ordre": "Achat" if "Bullish" in sig else "Vente",
            "Signal": sig,
            "Volatilité": VOLATILITY_LEVELS.get(instrument, "Inconnue"),
            "Force": strength or "Moyen",
            "Heure (UTC)": t
        }
    return {"error":False}

# ─── STREAMLIT APP ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Scanner CHoCH Pro", layout="wide")
    st.markdown("""
    <div style="display:flex;align-items:center;">
        <h1 style="margin:0;">Scanner Change of Character (CHoCH)</h1>
    </div>
    """, unsafe_allow_html=True)

    try:
        token = st.secrets["OANDA_ACCESS_TOKEN"]
    except:
        st.error("⚠️ OANDA_ACCESS_TOKEN manquant dans Secrets")
        st.stop()

    if st.button("Lancer le Scan", type="primary"):
        st.session_state.clear()
        api = API(access_token=token)

        with st.spinner("Scan en cours…"):
            results, failed = [], []
            total = len(INSTRUMENTS_TO_SCAN) * len(TIME_FRAMES)
            bar = st.progress(0)
            status = st.empty()

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exec:
                futures = {exec.submit(scan_pair_tf, api, instr, tf_n, tf_c): (instr, tf_n)
                           for instr in INSTRUMENTS_TO_SCAN
                           for tf_n, tf_c in TIME_FRAMES.items()}

                done = 0
                for f in as_completed(futures):
                    done += 1
                    bar.progress(done/total)
                    status.text(f"{done}/{total}")
                    res = f.result()
                    if res.get("error") == True:
                        failed.append(f"**{res['instrument']} {res['tf']}** → {res['message']}")
                    elif "Instrument" in res:
                        results.append(res)

            st.session_state.results = pd.DataFrame(results) if results else pd.DataFrame()
            st.session_state.failed = failed
        st.success("Scan terminé !")
        st.rerun()

    # ─── AFFICHAGE RÉSULTATS ───────────────────────────────────────────────────
    if "results" in st.session_state and not st.session_state.results.empty:
        df = st.session_state.results.copy()
        df["Heure (UTC)"] = pd.to_datetime(df["Heure (UTC)"]).dt.strftime("%Y-%m-%d %H:%M")

        st.markdown("### Export")
        csv = df.to_csv(index=False).encode()
        st.download_button("CSV", csv, "choch_signaux.csv", "text/csv")

        for tf in ["H1","H4","D1","Weekly"]:
            sub = df[df["Timeframe"] == tf]
            if sub.empty: continue
            sub = sub[["Instrument","Ordre","Signal","Volatilité","Force","Heure (UTC)"]].copy()
            sub.insert(0,"", ["New"] + [""]*(len(sub)-1))
            st.subheader(f"Signaux {tf}")

            def style_row(row):
                color = "#089981" if "Bullish" in row["Signal"] else "#f23645"
                return [f"color:{color};font-weight:bold" if col=="Signal" else
                        f"background:#089981;color:white" if row["Ordre"]=="Achat" else
                        f"background:#f23645;color:white" if row["Ordre"]=="Vente" else
                        "" for col in row.index]

            st.dataframe(sub.style.apply(style_row, axis=1), hide_index=True, use_container_width=True)

        if st.session_state.failed:
            with st.expander("Erreurs (clique pour voir)"):
                for e in st.session_state.failed:
                    st.write(e)
    elif "results" in st.session_state:
        st.success("Aucun CHoCH récent détecté")

if __name__ == "__main__":
    main()
            
