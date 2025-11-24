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
import warnings

# ──────────────────────────────────────────────────────────────
# SUPPRESSION DES WARNINGS JAUNES (Python 3.13+ + oandapyV20)
# ──────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=SyntaxWarning, module="oandapyV20")

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
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
FRACTAL_LENGTHS_BY_TF = {"H1": 5, "H4": 6, "D": 7, "W": 8}
RECENT_BARS_THRESHOLD = 10
MAX_WORKERS = 5

# Rate limiting safe (environ 11-12 requêtes/seconde)
RATE_LIMIT_LOCK = threading.Lock()
REQUEST_DELAY = 0.085

# ──────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ──────────────────────────────────────────────────────────────
def rate_limited_sleep():
    with RATE_LIMIT_LOCK:
        time.sleep(REQUEST_DELAY)

def calculate_atr(df, period=14):
    if df is None or len(df) < period + 5:
        return None
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1[1:], np.maximum(tr2[1:], tr3[1:]))
    return np.mean(tr[-period:]) if len(tr) >= period else None

def get_oanda_data(api_client, instrument, granularity, count=250, max_retries=3):
    params = {"count": count, "granularity": granularity}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    for attempt in range(max_retries):
        try:
            rate_limited_sleep()
            api_client.request(r)
            data = r.response.get('candles', [])
            if not data:
                return None, "Aucune donnée"

            times, opens, highs, lows, closes = [], [], [], [], []
            for c in data:
                if c.get('complete', False):
                    times.append(pd.to_datetime(c['time']))
                    mid = c['mid']
                    opens.append(float(mid['o']))
                    highs.append(float(mid['h']))
                    lows.append(float(mid['l']))
                    closes.append(float(mid['c']))

            if not times:
                return None, "Aucune bougie complète"

            df = pd.DataFrame({"time": times, "open": opens, "high": highs,
                               "low": lows, "close": closes})
            return df, "Succès"

        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"Erreur : {str(e)}"
            time_module.sleep(1)
    return None, "Échec total"

def detect_choch_optimized(df, instrument, tf_code):
    if df is None or len(df) < 30:
        return None, None, None

    length = FRACTAL_LENGTHS_BY_TF.get(tf_code, 5)
    p = length // 2
    if len(df) <= length + p:
        return None, None, None

    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    atr = calculate_atr(df)

    # Fractales
    is_bull_fractal = np.zeros(len(df), dtype=bool)
    is_bear_fractal = np.zeros(len(df), dtype=bool)
    for i in range(p, len(df) - p):
        if highs[i] == np.max(highs[i - p:i + p + 1]):
            is_b
