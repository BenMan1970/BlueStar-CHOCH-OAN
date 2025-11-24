# APP.PY — VERSION RÉPARÉE 100% OANDA (PDF inclus)
# Correction de la syntaxe et cohérence générale

import streamlit as st
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from datetime import datetime
import numpy as np
import time as time_module
import threading
import warnings
from fpdf import FPDF
import io

# ───────────────────────────────────────────
# SUPPRESSION WARNINGS PYTHON
warnings.filterwarnings("ignore", category=SyntaxWarning, module="oandapyV20")

# ───────────────────────────────────────────
# CONFIG
INSTRUMENTS_TO_SCAN = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF", "XAU_USD", "US30_USD", "NAS100_USD", "SPX500_USD"
]

VOLATILITY_LEVELS = {
    "EUR_USD": "Basse",
    "GBP_USD": "Basse",
    "USD_JPY": "Basse",
    "USD_CHF": "Basse",
    "USD_CAD": "Basse",
    "AUD_USD": "Moyenne",
    "NZD_USD": "Moyenne",
    "EUR_GBP": "Moyenne",
    "EUR_JPY": "Moyenne",
    "EUR_CHF": "Moyenne",
    "EUR_AUD": "Moyenne",
    "EUR_CAD": "Moyenne",
    "EUR_NZD": "Moyenne",
    "GBP_JPY": "Haute",
    "GBP_CHF": "Haute",
    "GBP_AUD": "Haute",
    "GBP_CAD": "Haute",
    "GBP_NZD": "Haute",
    "AUD_JPY": "Haute",
    "AUD_CAD": "Moyenne",
    "AUD_CHF": "Haute",
    "AUD_NZD": "Moyenne",
    "CAD_JPY": "Haute",
    "CAD_CHF": "Haute",
    "CHF_JPY": "Haute",
    "NZD_JPY": "Haute",
    "NZD_CAD": "Moyenne",
    "NZD_CHF": "Haute",
    "XAU_USD": "Très Haute",
    "US30_USD": "Très Haute",
    "NAS100_USD": "Très Haute",
    "SPX500_USD": "Très Haute"
}

TIME_FRAMES = {"H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"}
FRACTAL_LENGTHS_BY_TF = {"H1": 5, "H4": 6, "D": 7, "W": 8}
RECENT_BARS_THRESHOLD = 10

REQUEST_DELAY = 0.1
lock = threading.Lock()

def delay():
    with lock:
        time_module.sleep(REQUEST_DELAY)

# ───────────────────────────────────────────
# ATR

def calculate_atr(df, period=14):
    if len(df) < period + 5:
        return None
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    prev_c = np.roll(c, 1)
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))[1:]
    return tr[-period:].mean() if len(tr) >= period else None

# ───────────────────────────────────────────
# DATA OANDA

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
                if c.get('complete'):
                    m = c.get('mid', {})
                    if all(k in m for k in ('o','h','l','c')):
                        data.append([
                            pd.to_datetime(c['time']),
                            float(m['o']), float(m['h']), float(m['l']), float(m['c'])
                        ])
            if data:
                return pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close'])
        except Exception:
            time_module.sleep(1)
    return None

# ───────────────────────────────────────────
# DETECTION CHoCH REPARÉE

def detect_choch(df, tf_code):
    if df is None or len(df) < 40:
        return None, None, None

    length = FRACTAL_LENGTHS_BY_TF.get(tf_code, 5)
    p = length // 2

    high = df['high'].values
    low = df['low'].values
    close = df['close'].values

    is_bull = np.zeros(len(df), dtype=bool)
    is_bear = np.zeros(len(df), dtype=bool)

    for i in range(p, len(df) - p):
        window_high = high[i-p:i+p+1]
        window_low = low[i-p:i+p+1]
        if high[i] == np.max(window_high):
            is_bull[i] = True
        if low[i] == np.min(window_low):
            is_bear[i] = True

    upper_fr = None
    lower_fr = None

    recent = close[-10:]
    order_state = 1 if recent[-1] > recent[0] else -1

    atr = calculate_atr(df)

    signal = None
    time_sig = None
    strength = None
    bar_idx = None

    for i in range(len(df)):
        if is_bull[i]:
            upper_fr = high[i]
        if is_bear[i]:
            lower_fr = low[i]

        if order_state == -1 and upper_fr is not None and close[i] > upper_fr:
            bar_idx = i
            move = close[i] - upper_fr
            strength = "Fort" if (atr is not None and move > atr * 0.5) else "Moyen"
            signal = "Bullish CHoCH"
            time_sig = df['time'].iloc[i]
            order_state = 1

        if order_state == 1 and lower_fr is not None and close[i] < lower_fr:
            bar_idx = i
            move = lower_fr - close[i]
            strength = "Fort" if (atr is not None and move > atr * 0.5) else "Moyen"
            signal = "Bearish CHoCH"
            time_sig = df['time'].iloc[i]
            order_state = -1

    if signal and bar_idx is not None and (len(df) - 1 - bar_idx) <= RECENT_BARS_THRESHOLD:
        return signal, time_sig, strength

    return None, None, None

# ───────────────────────────────────────────
# PDF helper
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Rapport CHoCH', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


def create_pdf(df):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)
    pdf.cell(0, 10, 'Signaux CHoCH', ln=True)
    pdf.ln(4)
    for idx, row in df.iterrows():
        pdf.cell(0, 8, f"{row['Instrument']} | {row['Timeframe']} | {row['Ordre']} | {row['Signal']} | {row['Force']} | {row['Heure (UTC)']}", ln=True)
    out = io.BytesIO()
    pdf.output(out)
    out.seek(0)
    return out

# ───────────────────────────────────────────
# STREAMLIT UI
st.set_page_config(page_title="Scanner CHoCH", layout="wide")
st.markdown("<h1 style='text-align:center;'>Scanner CHoCH (OANDA)</h1>", unsafe_allow_html=True)

try:
    OANDA_ACCESS_TOKEN = st.secrets['OANDA_ACCESS_TOKEN']
except Exception:
    st.error('OANDA_ACCESS_TOKEN manquant dans Secrets')
    st.stop()

api_client = API(access_token=OANDA_ACCESS_TOKEN)

if st.button('Lancer un Scan', type='primary'):
    st.session_state.clear()
    with st.spinner('Scan en cours...'):
        results = []
        for instrument in INSTRUMENTS_TO_SCAN:
            for tf_name, tf_code in TIME_FRAMES.items():
                df = get_oanda_data(instrument, tf_code)
                sig, t, f = detect_choch(df, tf_code)
                if sig:
                    results.append({
                        'Instrument': instrument.replace('_', '/'),
                        'Timeframe': tf_name,
                        'Ordre': 'Achat' if 'Bullish' in sig else 'Vente',
                        'Signal': sig,
                        'Volatilité': VOLATILITY_LEVELS.get(instrument, 'Inconnue'),
                        'Force': f or 'Moyen',
                        'Heure (UTC)': t
                    })
        st.session_state.results = pd.DataFrame(results)
    st.rerun()

# AFFICHAGE
if 'results' in st.session_state:
    df = st.session_state.results
    if df.empty:
        st.success('Aucun signal trouvé')
    else:
        df['Heure (UTC)'] = pd.to_datetime(df['Heure (UTC)']).dt.strftime('%d/%m %H:%M')
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode()
        st.download_button('Télécharger CSV', csv, 'signals.csv', 'text/csv')

        pdf_bytes = create_pdf(df)
        st.download_button('Télécharger PDF', pdf_bytes, 'signals.pdf', 'application/pdf')
