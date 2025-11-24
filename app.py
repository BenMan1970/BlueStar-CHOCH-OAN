# app.py
import streamlit as st
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import time
import time as time_module
from datetime import datetime
import io
from fpdf import FPDF
import dataframe_image as dfi
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import threading

# --- CONFIGURATION ---
INSTRUMENTS_TO_SCAN = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "XAU_USD", "US30_USD", "NAS100_USD", "SPX500_USD"
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

FRACTAL_LENGTH = 5
RECENT_BARS_THRESHOLD = 10
MAX_WORKERS = 5

# Fractales adaptatives par timeframe
FRACTAL_LENGTHS_BY_TF = {
    "H1": 5,
    "H4": 6,
    "D": 7,
    "W": 8
}

# Rate limiting doux (12 appels/seconde → safe même en compte réel)
RATE_LIMIT_LOCK = threading.Lock()
REQUEST_DELAY = 0.085  # 85 ms entre chaque appel

# --- FONCTIONS ---
def rate_limited_sleep():
    with RATE_LIMIT_LOCK:
        time.sleep(REQUEST_DELAY)

def calculate_atr(df, period=14):
    """Calcule l'ATR simple (14 périodes)"""
    if df is None or len(df) < period + 5:
        return None
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1[1:], np.maximum(tr2[1:], tr3[1:]))
    if len(tr) < period:
        return None
    return np.mean(tr[-period:])

def get_oanda_data(api_client, instrument, granularity, count=250, max_retries=3):
    params = {"count": count, "granularity": granularity}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    
    for attempt in range(max_retries):
        try:
            rate_limited_sleep()  # Respect du rate limit
            api_client.request(r)
            data = r.response.get('candles')
            if not data:
                return None, f"Aucune donnée pour {instrument}"

            times, opens, highs, lows, closes = [], [], [], [], []
            for c in data:
                if c['complete']:
                    times.append(pd.to_datetime(c['time']))
                    opens.append(float(c['mid']['o']))
                    highs.append(float(c['mid']['h']))
                    lows.append(float(c['mid']['l']))
                    closes.append(float(c['mid']['c']))

            if not times:
                return None, "Aucune bougie complète"

            df = pd.DataFrame({
                "time": times,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes
            })
            return df, "Succès"

        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"Erreur après {max_retries} tentatives : {str(e)}"
            time_module.sleep(1)

    return None, "Échec total"

def detect_choch_optimized(df, instrument, tf_code, length=None):
    if df is None or len(df) < 20:
        return None, None, None

    length = FRACTAL_LENGTHS_BY_TF.get(tf_code, FRACTAL_LENGTH)
    p = length // 2
    if len(df) <= length + p:
        return None, None, None

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atr = calculate_atr(df)

    # Détection des fractales
    is_bull_fractal = np.zeros(len(df), dtype=bool)
    is_bear_fractal = np.zeros(len(df), dtype=bool)
    for i in range(p, len(df) - p):
        if highs[i] == np.max(highs[i - p:i + p + 1]):
            is_bull_fractal[i] = True
        if lows[i] == np.min(lows[i - p:i + p + 1]):
            is_bear_fractal[i] = True

    upper_fractal = {'value': None, 'iscrossed': True}
    lower_fractal = {'value': None, 'iscrossed': True}
    os = 0
    choch_signal = choch_time = choch_bar_index = confirmation_strength = None

    for i in range(length, len(df)):
        if is_bull_fractal[i - p]:
            upper_fractal = {'value': highs[i - p], 'iscrossed': False}
        if is_bear_fractal[i - p]:
            lower_fractal = {'value': lows[i - p], 'iscrossed': False}

        curr, prev = closes[i], closes[i - 1]

        # Bullish CHoCH
        if (upper_fractal['value'] is not None and not upper_fractal['iscrossed'] and
                curr > upper_fractal['value'] and prev <= upper_fractal['value'] and os == -1):
            choch_signal = "Bullish CHoCH"
            choch_time = df['time'].iloc[i]
            choch_bar_index = i
            move = curr - upper_fractal['value']
            confirmation_strength = "Fort" if atr and move > atr * 0.5 else "Moyen"
            os = 1
            upper_fractal['iscrossed'] = True

        # Bearish CHoCH
        if (lower_fractal['value'] is not None and not lower_fractal['iscrossed'] and
                curr < lower_fractal['value'] and prev >= lower_fractal['value'] and os == 1):
            choch_signal = "Bearish CHoCH"
            choch_time = df['time'].iloc[i]
            choch_bar_index = i
            move = lower_fractal['value'] - curr
            confirmation_strength = "Fort" if atr and move > atr * 0.5 else "Moyen"
            os = -1
            lower_fractal['iscrossed'] = True

    # Vérification de récence du signal
    if choch_signal and choch_bar_index is not None:
        bars_since = len(df) - 1 - choch_bar_index
        if bars_since < RECENT_BARS_THRESHOLD:
            return choch_signal, choch_time, confirmation_strength

    return None, None, None

def scan_instrument_timeframe(api_client, instrument, tf_name, tf_code):
    df, msg = get_oanda_data(api_client, instrument, tf_code)
    if df is None:
        return {"error": True, "instrument": instrument, "tf": tf_name, "message": msg}

    signal, sig_time, strength = detect_choch_optimized(df, instrument, tf_code)
    if signal:
        return {
            "Instrument": instrument.replace("_", "/"),
            "Timeframe": tf_name,
            "Ordre": "Achat" if "Bullish" in signal else "Vente",
            "Signal": signal,
            "Volatilité": VOLATILITY_LEVELS.get(instrument, "Inconnue"),
            "Force": strength or "Moyen",
            "Heure (UTC)": sig_time
        }
    return {"error": False}  # Pas d'erreur, juste pas de signal

# --- STREAMLIT APP ---
def main():
    st.set_page_config(page_title="Scanner CHoCH Pro", layout="wide")
    st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 25px;">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 10px;">
                <path d="M4 4V8H8" stroke="#f23645" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M4 12V20H20V4H12" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M8 4L4 8" stroke="#089981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <h1 style="margin: 0;">Scanner de Change of Character (CHoCH)</h1>
        </div>
    """, unsafe_allow_html=True)

    try:
        OANDA_ACCESS_TOKEN = st.secrets["OANDA_ACCESS_TOKEN"]
    except:
        st.error("OANDA_ACCESS_TOKEN manquant dans les secrets Streamlit.")
        st.stop()

    if st.button("Lancer un nouveau Scan", type="primary"):
        st.session_state.clear()

        api_client = API(access_token=OANDA_ACCESS_TOKEN, environment="practice")
        with st.spinner("Scan en cours sur tous les instruments..."):
            results, failed = [], []
            total = len(INSTRUMENTS_TO_SCAN) * len(TIME_FRAMES)
            progress_bar = st.progress(0)
            status_text = st.empty()

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(scan_instrument_timeframe, api_client, instr, tf_name, tf_code): (instr, tf_name)
                    for instr in INSTRUMENTS_TO_SCAN
                    for tf_name, tf_code in TIME_FRAMES.items()
                }

                completed = 0
                for future in as_completed(futures):
                    completed += 1
                    instr, tf = futures[future]
                    progress_bar.progress(completed / total)
                    status_text.text(f"{completed}/{total} terminés")

                    try:
                        res = future.result()
                        if res.get("error") is True:
                            failed.append(f"- **{res['instrument']} ({res['tf']})** : {res['message']}")
                        elif "Instrument" in res:
                            results.append(res)
                    except Exception as e:
                        failed.append(f"- **{instr} ({tf})** : {str(e)}")

            st.session_state["results"] = pd.DataFrame(results) if results else pd.DataFrame()
            st.session_state["failed"] = failed
            progress_bar.progress(1.0)
            status_text.success("Scan terminé !")

        st.rerun()

    # --- AFFICHAGE DES RÉSULTATS ---
    if "results" in st.session_state:
        df = st.session_state["results"]
        if df.empty:
            st.success("Aucun signal CHoCH récent détecté sur les 124 configurations.")
        else:
            df['Heure (UTC)'] = pd.to_datetime(df['Heure (UTC)'])
            df = df.sort_values("Heure (UTC)", ascending=False)

            # Exports
            st.markdown("### Exporter les résultats")
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
            csv = df.to_csv(index=False).encode()
            st.download_button("Télécharger CSV", csv, f"choch_{timestamp}.csv", "text/csv")

            buf = io.BytesIO()
            try:
                dfi.export(df, buf, table_conversion="matplotlib")
                buf.seek(0)
                st.download_button("Télécharger PNG", buf, f"choch_{timestamp}.png", "image/png")
            except:
                st.warning("Export PNG indisponible (dataframe_image)")

            # Affichage par timeframe
            for tf in ["H1", "H4", "D1", "Weekly"]:
                sub = df[df["Timeframe"] == tf].copy()
                if sub.empty:
                    continue
                sub = sub.drop(columns="Timeframe")
                sub.insert(0, " ", ["New"] + [""]*(len(sub)-1))
                sub["Heure (UTC)"] = sub["Heure (UTC)"].dt.strftime("%Y-%m-%d %H:%M")

                st.subheader(f"--- Signaux {tf} ---")

                def color_sig(val): return f"color: {'#089981' if 'Bullish' in val else '#f23645'}; font-weight:bold"
                def color_order(val): return f"background-color: {'#089981' if val=='Achat' else '#f23645'}; color:white; text-align:center; border-radius:5px; padding:2px 8px;"
                def color_vol(val):
                    c = {"Basse":"#089981", "Moyenne":"#FFA500", "Haute":"#FF6B6B", "Très Haute":"#f23645"}.get(val, "gray")
                    return f"background-color:{c}; color:white; text-align:center; border-radius:4px;"
                def color_force(val): return f"background-color: {'#089981' if val=='Fort' else '#FFA500'}; color:white; text-align:center; border-radius:4px;"

                styled = sub.style \
                    .applymap(color_sig, subset=["Signal"]) \
                    .applymap(color_order, subset=["Ordre"]) \
                    .applymap(color_vol, subset=["Volatilité"]) \
                    .applymap(color_force, subset=["Force"])

                st.dataframe(styled, hide_index=True, use_container_width=True)

        if st.session_state.get("failed"):
            with st.expander("Erreurs / instruments non chargés"):
                for f in st.session_state["failed"]:
                    st.write(f)

if __name__ == "__main__":
    main()
