import streamlit as st 
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import time as time_module
from datetime import datetime
import io
from fpdf import FPDF
from PIL import Image
import dataframe_image as dfi

# --- CONFIGURATION ---
INSTRUMENTS_TO_SCAN = [
    # Paires Majeures
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    # Paires Mineures (Croisées EUR)
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    # Paires Mineures (Croisées GBP)
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    # Paires Mineures (Autres)
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CAD", "NZD_CHF",
    # Indices et Métaux
    "XAU_USD", "US30_USD", "NAS100_USD", "SPX500_USD"
]
TIME_FRAMES = {
    "H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"
}
FRACTAL_LENGTH = 5
RECENT_BARS_THRESHOLD = 10

# --- Fonctions ---
def get_oanda_data(api_client, instrument, granularity, count=250):
    params = {"count": count, "granularity": granularity}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    try:
        api_client.request(r)
        data = r.response.get('candles')
        if not data: return None
        df = pd.DataFrame([
            {"time": pd.to_datetime(c['time']), "open": float(c['mid']['o']), "high": float(c['mid']['h']), "low": float(c['mid']['l']), "close": float(c['mid']['c'])}
            for c in data if c['complete']
        ])
        df['time'] = pd.to_datetime(df['time'])
        return df
    except Exception: return None

def detect_choch(df, length=5):
    if df is None or len(df) < length: return None, None
    p = length // 2
    df['is_bull_fractal'] = (df['high'] == df['high'].rolling(window=length, center=True, min_periods=length).max())
    df['is_bear_fractal'] = (df['low'] == df['low'].rolling(window=length, center=True, min_periods=length).min())
    upper_fractal = {'value': None, 'iscrossed': True}
    lower_fractal = {'value': None, 'iscrossed': True}
    os = 0
    choch_signal, choch_time, choch_bar_index = None, None, -1
    for i in range(length, len(df)):
        if df['is_bull_fractal'].iloc[i - p]: upper_fractal = {'value': df['high'].iloc[i - p], 'iscrossed': False}
        if df['is_bear_fractal'].iloc[i - p]: lower_fractal = {'value': df['low'].iloc[i - p], 'iscrossed': False}
        current_close, previous_close = df['close'].iloc[i], df['close'].iloc[i - 1]
        if upper_fractal['value'] is not None and not upper_fractal['iscrossed']:
            if current_close > upper_fractal['value'] and previous_close <= upper_fractal['value']:
                if os == -1: choch_signal, choch_time, choch_bar_index = "Bullish CHoCH", df['time'].iloc[i], i
                os, upper_fractal['iscrossed'] = 1, True
        if lower_fractal['value'] is not None and not lower_fractal['iscrossed']:
            if current_close < lower_fractal['value'] and previous_close >= lower_fractal['value']:
                if os == 1: choch_signal, choch_time, choch_bar_index = "Bearish CHoCH", df['time'].iloc[i], i
                os, lower_fractal['iscrossed'] = -1, True
    if choch_signal and (len(df) - 1 - choch_bar_index) < RECENT_BARS_THRESHOLD:
        return choch_signal, choch_time
    return None, None

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
    except KeyError:
        st.error("Erreur : Veuillez configurer OANDA_ACCESS_TOKEN dans les secrets de Streamlit.")
        st.stop()

    results_placeholder = st.empty()

    if st.button('Lancer le Scan'):
        results_placeholder.empty()
        try:
            api_client = API(access_token=OANDA_ACCESS_TOKEN)
        except Exception as e:
            st.error(f"Erreur d'initialisation de l'API Oanda: {e}")
            st.stop()

        with st.spinner('Scan en cours...'):
            results = []
            total_scans = len(INSTRUMENTS_TO_SCAN) * len(TIME_FRAMES)
            progress_bar = st.progress(0)
            progress_status = st.empty()
            for i, instrument in enumerate(INSTRUMENTS_TO_SCAN):
                for tf_name, tf_code in TIME_FRAMES.items():
                    progress_value = (i * len(TIME_FRAMES) + list(TIME_FRAMES.keys()).index(tf_name) + 1) / total_scans
                    progress_bar.progress(progress_value)
                    progress_status.text(f"Scan de {instrument} sur {tf_name}...")
                    df = get_oanda_data(api_client, instrument, tf_code)
                    if df is not None:
                        signal, signal_time = detect_choch(df, length=FRACTAL_LENGTH)
                        if signal:
                            action = "Achat" if "Bullish" in signal else "Vente"
                            results.append({
                                "Instrument": instrument.replace("_", "/"), "Timeframe": tf_name, "Ordre": action,
                                "Signal": signal, "Heure (UTC)": signal_time
                            })
                    else:
                        st.warning(f"Données non disponibles pour {instrument} sur {tf_name}.")
                    time_module.sleep(0.25)
            progress_status.success("Scan terminé !")

            with results_placeholder.container():
                if not results:
                    st.success("✅ Aucun signal de CHoCH récent détecté.")
                else:
                    full_df = pd.DataFrame(results)
                    full_df['Heure (UTC)'] = pd.to_datetime(full_df['Heure (UTC)'])

                    # --- Export PDF, PNG, CSV ---
                    st.markdown("### 📤 Exporter les résultats")
                    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")

                    csv = full_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Télécharger en CSV", csv, f"choch_signaux_{timestamp}.csv", "text/csv")

                    try:
                        image_buf = io.BytesIO()
                        dfi.export(full_df, image_buf, table_conversion='matplotlib')
                        image_buf.seek(0)
                        st.download_button("🖼 Télécharger en PNG", image_buf, f"choch_signaux_{timestamp}.png", "image/png")
                    except Exception as e:
                        st.warning(f"Erreur lors de l'export PNG : {e}")

                    try:
                        pdf_buf = io.BytesIO()
                        img_temp = "temp_table_img.png"
                        dfi.export(full_df, img_temp, table_conversion='matplotlib')
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.image(img_temp, x=10, y=10, w=190)
                        pdf.output(pdf_buf)
                        pdf_buf.seek(0)
                        st.download_button("📄 Télécharger en PDF", pdf_buf, f"choch_signaux_{timestamp}.pdf", "application/pdf")
                    except Exception as e:
                        st.warning(f"Erreur lors de l'export PDF : {e}")

                    # --- Affichage par timeframe ---
                    for tf_name, tf_code in TIME_FRAMES.items():
                        tf_df = full_df[full_df['Timeframe'] == tf_name].copy()
                        if not tf_df.empty:
                            tf_df = tf_df.sort_values(by='Heure (UTC)', ascending=False)
                            tf_df.insert(0, ' ', ['⭐'] + [''] * (len(tf_df) - 1))
                            tf_df['Heure (UTC)'] = tf_df['Heure (UTC)'].dt.strftime('%Y-%m-%d %H:%M')

                            st.subheader(f"--- Signaux {tf_name} ---")

                            def color_signal(val): return f'color: {"#089981" if "Bullish" in val else "#f23645"}; font-weight: bold;'
                            def style_order(val): return f'background-color: {"#089981" if val == "Achat" else "#f23645"}; color: white; border-radius: 5px; text-align: center; font-weight: bold;'

                            styled_df = tf_df.drop(columns=['Timeframe']).style\
                                .applymap(color_signal, subset=['Signal'])\
                                .applymap(style_order, subset=['Ordre'])

                            st.dataframe(styled_df, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()
