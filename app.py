import streamlit as st
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.exceptions
import time as time_module
from datetime import datetime
import io
from fpdf import FPDF
import dataframe_image as dfi
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# --- CONFIGURATION ---
INSTRUMENTS_TO_SCAN = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "XAU_USD", "US30_USD", "NAS100_USD", "SPX500_USD"
]

# Volatilité classée par instrument (basée sur ATR moyen historique)
VOLATILITY_LEVELS = {
    # Majors (faible volatilité)
    "EUR_USD": "Basse", "GBP_USD": "Basse", "USD_JPY": "Basse", "USD_CHF": "Basse", 
    "USD_CAD": "Basse", "AUD_USD": "Moyenne", "NZD_USD": "Moyenne",
    # Crosses (moyenne volatilité)
    "EUR_GBP": "Moyenne", "EUR_JPY": "Moyenne", "EUR_CHF": "Moyenne", 
    "EUR_AUD": "Moyenne", "EUR_CAD": "Moyenne", "EUR_NZD": "Moyenne",
    "GBP_JPY": "Haute", "GBP_CHF": "Haute", "GBP_AUD": "Haute", 
    "GBP_CAD": "Haute", "GBP_NZD": "Haute",
    "AUD_JPY": "Haute", "AUD_CAD": "Moyenne", "AUD_CHF": "Haute", 
    "AUD_NZD": "Moyenne", "CAD_JPY": "Haute", "CAD_CHF": "Haute", 
    "CHF_JPY": "Haute", "NZD_JPY": "Haute", "NZD_CAD": "Moyenne", "NZD_CHF": "Haute",
    # Commodités & Indices (très haute volatilité)
    "XAU_USD": "Très Haute", "US30_USD": "Très Haute", "NAS100_USD": "Très Haute", "SPX500_USD": "Très Haute"
}

TIME_FRAMES = {"H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"}
FRACTAL_LENGTH = 5
RECENT_BARS_THRESHOLD = 10
MAX_WORKERS = 5

# Fractales adaptatives par timeframe
FRACTAL_LENGTHS_BY_TF = {
    "H1": 5,      # Court terme, fractales serrées
    "H4": 6,      # Un peu plus large
    "D1": 7,      # Fractales plus importantes
    "Weekly": 8   # Long terme, fractales larges
}

# --- Fonctions optimisées ---
def calculate_atr(df, period=14):
    """Calcule l'ATR pour mesurer la volatilité actuelle"""
    if df is None or len(df) < period:
        return None
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - close[:-1])
    tr3 = np.abs(low - close[:-1])
    
    tr = np.maximum(tr1[1:], np.maximum(tr2, tr3))
    atr = np.mean(tr[-period:])
    
    return atr

def get_oanda_data(api_client, instrument, granularity, count=250, max_retries=3, retry_delay=2):
    params = {"count": count, "granularity": granularity}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    for attempt in range(max_retries):
        try:
            api_client.request(r)
            data = r.response.get('candles')
            if not data:
                return None, f"Aucune bougie pour {instrument} sur {granularity}."
            
            times = []
            opens, highs, lows, closes = [], [], [], []
            for c in data:
                if c['complete']:
                    times.append(pd.to_datetime(c['time']))
                    opens.append(float(c['mid']['o']))
                    highs.append(float(c['mid']['h']))
                    lows.append(float(c['mid']['l']))
                    closes.append(float(c['mid']['c']))
            
            if not times:
                return None, f"DataFrame vide pour {instrument} sur {granularity}."
            
            df = pd.DataFrame({
                "time": times,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes
            })
            return df, "Succès"
            
        except oandapyV20.exceptions.V20Error as e:
            error_message = f"Erreur API OANDA (tentative {attempt + 1}/{max_retries}): {e}"
            if attempt + 1 == max_retries:
                return None, error_message
            time_module.sleep(retry_delay)
        except Exception as e:
            error_message = f"Erreur inattendue (tentative {attempt + 1}/{max_retries}): {e}"
            if attempt + 1 == max_retries:
                return None, error_message
            time_module.sleep(retry_delay)
    
    return None, "Échec après plusieurs tentatives."

def detect_choch_optimized(df, instrument, tf_code, length=None):
    """Détection CHoCH optimisée avec fractales adaptatives"""
    if df is None or len(df) < length:
        return None, None, None
    
    # Adapter la longueur de fractale selon le timeframe
    if length is None:
        length = FRACTAL_LENGTHS_BY_TF.get(tf_code, FRACTAL_LENGTH)
    
    p = length // 2
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    # Calculer ATR pour qualifier la confirmation
    atr = calculate_atr(df)
    
    is_bull_fractal = np.zeros(len(df), dtype=bool)
    is_bear_fractal = np.zeros(len(df), dtype=bool)
    
    for i in range(p, len(df) - p):
        window_highs = highs[i - p:i + p + 1]
        window_lows = lows[i - p:i + p + 1]
        if highs[i] == np.max(window_highs):
            is_bull_fractal[i] = True
        if lows[i] == np.min(window_lows):
            is_bear_fractal[i] = True
    
    upper_fractal = {'value': None, 'iscrossed': True}
    lower_fractal = {'value': None, 'iscrossed': True}
    os = 0
    choch_signal, choch_time = None, None
    choch_bar_index = -1
    confirmation_strength = None
    
    for i in range(length, len(df)):
        if is_bull_fractal[i - p]:
            upper_fractal = {'value': highs[i - p], 'iscrossed': False}
        if is_bear_fractal[i - p]:
            lower_fractal = {'value': lows[i - p], 'iscrossed': False}
        
        current_close, previous_close = closes[i], closes[i - 1]
        
        if upper_fractal['value'] is not None and not upper_fractal['iscrossed']:
            if current_close > upper_fractal['value'] and previous_close <= upper_fractal['value']:
                if os == -1:
                    choch_signal = "Bullish CHoCH"
                    choch_time = df['time'].iloc[i]
                    choch_bar_index = i
                    # Force de confirmation basée sur la distance parcourue au-delà de la fractale
                    move_beyond = current_close - upper_fractal['value']
                    confirmation_strength = "Fort" if atr and move_beyond > atr * 0.5 else "Moyen"
                os, upper_fractal['iscrossed'] = 1, True
        
        if lower_fractal['value'] is not None and not lower_fractal['iscrossed']:
            if current_close < lower_fractal['value'] and previous_close >= lower_fractal['value']:
                if os == 1:
                    choch_signal = "Bearish CHoCH"
                    choch_time = df['time'].iloc[i]
                    choch_bar_index = i
                    move_beyond = lower_fractal['value'] - current_close
                    confirmation_strength = "Fort" if atr and move_beyond > atr * 0.5 else "Moyen"
                os, lower_fractal['iscrossed'] = -1, True
    
    if choch_signal and (len(df) - 1 - choch_bar_index) < RECENT_BARS_THRESHOLD:
        return choch_signal, choch_time, confirmation_strength
    
    return None, None, None

def scan_instrument_timeframe(api_client, instrument, tf_name, tf_code):
    """Fonction pour scanner un couple instrument/timeframe"""
    df, status_message = get_oanda_data(api_client, instrument, tf_code)
    
    if df is not None:
        signal, signal_time, confirmation = detect_choch_optimized(df, instrument, tf_code)
        if signal:
            action = "Achat" if "Bullish" in signal else "Vente"
            volatility = VOLATILITY_LEVELS.get(instrument, "Inconnue")
            return {
                "Instrument": instrument.replace("_", "/"),
                "Timeframe": tf_name,
                "Ordre": action,
                "Signal": signal,
                "Volatilité": volatility,
                "Force": confirmation,
                "Heure (UTC)": signal_time
            }
    
    return {"error": True, "instrument": instrument, "tf": tf_name, "message": status_message}

def main():
    st.set_page_config(page_title="Scanner de CHoCH", layout="wide")
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
    except (KeyError, FileNotFoundError):
        st.error("Erreur : Veuillez configurer OANDA_ACCESS_TOKEN dans les secrets de Streamlit.")
        st.stop()
    
    if st.button('Lancer un nouveau Scan'):
        if 'scan_results' in st.session_state:
            del st.session_state['scan_results']
        if 'failed_scans' in st.session_state:
            del st.session_state['failed_scans']
            
        try:
            api_client = API(access_token=OANDA_ACCESS_TOKEN)
        except Exception as e:
            st.error(f"Erreur d'initialisation de l'API Oanda: {e}")
            st.stop()

        with st.spinner('Scan en cours...'):
            results, failed = [], []
            total_scans = len(INSTRUMENTS_TO_SCAN) * len(TIME_FRAMES)
            progress_bar = st.progress(0)
            progress_status = st.empty()
            completed_scans = 0
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {}
                
                for instrument in INSTRUMENTS_TO_SCAN:
                    for tf_name, tf_code in TIME_FRAMES.items():
                        future = executor.submit(scan_instrument_timeframe, api_client, instrument, tf_name, tf_code)
                        futures[future] = (instrument, tf_name)
                
                for future in as_completed(futures):
                    completed_scans += 1
                    instrument, tf_name = futures[future]
                    
                    try:
                        result = future.result()
                        if "error" not in result or not result["error"]:
                            results.append(result)
                        else:
                            failed.append(f"- **{result['instrument']} ({result['tf']})**: {result['message']}")
                    except Exception as e:
                        failed.append(f"- **{instrument} ({tf_name})**: Erreur d'exécution: {e}")
                    
                    progress_value = completed_scans / total_scans
                    progress_bar.progress(progress_value)
                    progress_status.text(f"Progression: {completed_scans}/{total_scans} scans")
            
            progress_status.success("✅ Scan terminé !")
            st.session_state['scan_results'] = pd.DataFrame(results) if results else None
            st.session_state['failed_scans'] = failed if failed else None
            
            st.rerun()

    if 'scan_results' in st.session_state:
        full_df = st.session_state['scan_results']

        if full_df is None or full_df.empty:
            st.success("✅ Aucun signal de CHoCH récent détecté.")
        else:
            full_df['Heure (UTC)'] = pd.to_datetime(full_df['Heure (UTC)'])
            st.markdown("### 📤 Exporter les résultats")
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
            df_to_export = full_df.copy()

            csv = df_to_export.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger en CSV", csv, f"choch_signaux_{timestamp}.csv", "text/csv")

            try:
                image_buf = io.BytesIO()
                dfi.export(df_to_export, image_buf, table_conversion='matplotlib')
                image_buf.seek(0)
                st.download_button("🖼️ Télécharger en PNG", image_buf, f"choch_signaux_{timestamp}.png", "image/png")
            except Exception as e:
                st.warning(f"Erreur lors de l'export PNG : {e}")

            try:
                pdf = FPDF(orientation='L', unit='mm', format='A4')
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "Rapport des Signaux CHoCH", 0, 1, 'C')
                pdf.set_font("Arial", size=10)
                rows_per_page = 20
                for i in range(0, len(df_to_export), rows_per_page):
                    if i > 0:
                        pdf.add_page()
                    chunk = df_to_export.iloc[i:i+rows_per_page]
                    img_temp_path = f"temp_chunk_{i}.png"
                    dfi.export(chunk, img_temp_path, table_conversion='matplotlib')
                    pdf.image(img_temp_path, x=10, y=pdf.get_y(), w=277)
                    os.remove(img_temp_path)
                pdf_output = pdf.output(dest='S').encode('latin-1')
                st.download_button("📄 Télécharger en PDF", pdf_output, f"choch_signaux_{timestamp}.pdf", "application/pdf")
            except Exception as e:
                st.warning(f"Erreur lors de l'export PDF : {e}")

            for tf_name in TIME_FRAMES.keys():
                tf_df = full_df[full_df['Timeframe'] == tf_name].copy()
                if not tf_df.empty:
                    tf_df = tf_df.sort_values(by='Heure (UTC)', ascending=False)
                    tf_df.insert(0, ' ', ['⭐'] + [''] * (len(tf_df) - 1))
                    tf_df['Heure (UTC)'] = tf_df['Heure (UTC)'].dt.strftime('%Y-%m-%d %H:%M')
                    st.subheader(f"--- Signaux {tf_name} ---")
                    def color_signal(val):
                        return f'color: {"#089981" if "Bullish" in val else "#f23645"}; font-weight: bold;'
                    def style_order(val):
                        return f'background-color: {"#089981" if val == "Achat" else "#f23645"}; color: white; border-radius: 5px; text-align: center; font-weight: bold;'
                    def style_volatility(val):
                        colors = {"Basse": "#089981", "Moyenne": "#FFA500", "Haute": "#FF6B6B", "Très Haute": "#f23645"}
                        color = colors.get(val, "#FFFFFF")
                        return f'background-color: {color}; color: white; text-align: center; border-radius: 3px;'
                    def style_force(val):
                        return f'background-color: {"#089981" if val == "Fort" else "#FFA500"}; color: white; text-align: center; border-radius: 3px;'
                    
                    styled_df = tf_df.drop(columns=['Timeframe']).style \
                        .applymap(color_signal, subset=['Signal']) \
                        .applymap(style_order, subset=['Ordre']) \
                        .applymap(style_volatility, subset=['Volatilité']) \
                        .applymap(style_force, subset=['Force'])
                    st.dataframe(styled_df, hide_index=True, use_container_width=True)

        if 'failed_scans' in st.session_state and st.session_state['failed_scans']:
            with st.expander("⚠️ Voir le rapport des scans ayant échoué"):
                st.markdown("\n".join(st.session_state['failed_scans']))

if __name__ == "__main__":
    main()
