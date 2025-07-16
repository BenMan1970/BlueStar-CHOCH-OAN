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

# --- CONFIGURATION (inchangée) ---
INSTRUMENTS_TO_SCAN = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "XAU_USD", "US30_USD", "NAS100_USD", "SPX500_USD"
]
TIME_FRAMES = {"H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"}
FRACTAL_LENGTH = 5
RECENT_BARS_THRESHOLD = 10

# --- Fonctions (inchangées) ---
def get_oanda_data(api_client, instrument, granularity, count=250, max_retries=3, retry_delay=5):
    params = {"count": count, "granularity": granularity}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    for attempt in range(max_retries):
        try:
            api_client.request(r)
            data = r.response.get('candles')
            if not data: return None, f"Aucune bougie retournée par l'API pour {instrument} sur {granularity}."
            df = pd.DataFrame([
                {"time": pd.to_datetime(c['time']), "open": float(c['mid']['o']), "high": float(c['mid']['h']), "low": float(c['mid']['l']), "close": float(c['mid']['c'])}
                for c in data if c['complete']
            ])
            if df.empty: return None, f"DataFrame vide après traitement pour {instrument} sur {granularity}."
            return df, "Succès"
        except oandapyV20.exceptions.V20Error as e:
            error_message = f"Erreur API OANDA (tentative {attempt + 1}/{max_retries}): {e}"
            if attempt + 1 == max_retries: return None, error_message
            time_module.sleep(retry_delay)
        except Exception as e:
            error_message = f"Erreur inattendue (tentative {attempt + 1}/{max_retries}): {e}"
            if attempt + 1 == max_retries: return None, error_message
            time_module.sleep(retry_delay)
    return None, "Échec de la récupération des données après plusieurs tentatives."

def detect_choch(df, length=5):
    if df is None or len(df) < length: return None, None
    p = length // 2
    df['is_bull_fractal'] = (df['high'] == df['high'].rolling(window=length, center=True, min_periods=length).max())
    df['is_bear_fractal'] = (df['low'] == df['low'].rolling(window=length, center=True, min_periods=length).min())
    upper_fractal, lower_fractal, os = {'value': None, 'iscrossed': True}, {'value': None, 'iscrossed': True}, 0
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
    except (KeyError, FileNotFoundError):
        st.error("Erreur : Veuillez configurer OANDA_ACCESS_TOKEN dans les secrets de Streamlit.")
        st.stop()
    
    # --- MODIFICATION CLÉ : Logique du bouton de scan ---
    # Le bouton va maintenant lancer le scan ET stocker les résultats dans st.session_state
    if st.button('Lancer un nouveau Scan'):
        # On nettoie les anciens résultats s'ils existent
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
            results = []
            failed = []
            total_scans = len(INSTRUMENTS_TO_SCAN) * len(TIME_FRAMES)
            progress_bar = st.progress(0)
            progress_status = st.empty()
            
            for i, instrument in enumerate(INSTRUMENTS_TO_SCAN):
                for j, (tf_name, tf_code) in enumerate(TIME_FRAMES.items()):
                    progress_value = (i * len(TIME_FRAMES) + j + 1) / total_scans
                    progress_bar.progress(progress_value)
                    progress_status.text(f"Scan de {instrument} sur {tf_name}...")
                    df, status_message = get_oanda_data(api_client, instrument, tf_code)
                    if df is not None:
                        signal, signal_time = detect_choch(df, length=FRACTAL_LENGTH)
                        if signal:
                            action = "Achat" if "Bullish" in signal else "Vente"
                            results.append({"Instrument": instrument.replace("_", "/"), "Timeframe": tf_name, "Ordre": action, "Signal": signal, "Heure (UTC)": signal_time})
                    else:
                        failed.append(f"- **{instrument} ({tf_name})**: {status_message}")
                    time_module.sleep(0.5)
            
            progress_status.success("Scan terminé !")
            
            # NOUVEAU : On sauvegarde les résultats dans la session
            st.session_state['scan_results'] = pd.DataFrame(results) if results else None
            st.session_state['failed_scans'] = failed if failed else None
            st.experimental_rerun() # On force la ré-exécution pour afficher les résultats proprement

    # --- MODIFICATION CLÉ : Logique d'affichage ---
    # On affiche les résultats s'ils sont présents dans st.session_state
    # Cela fonctionnera même après un téléchargement
    if 'scan_results' in st.session_state:
        full_df = st.session_state['scan_results']

        if full_df is None or full_df.empty:
            st.success("✅ Aucun signal de CHoCH récent détecté.")
        else:
            full_df['Heure (UTC)'] = pd.to_datetime(full_df['Heure (UTC)'])

            # --- EXPORT SECTION ---
            st.markdown("### 📤 Exporter les résultats")
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
            df_to_export = full_df.copy() # On utilise une copie pour l'export

            # CSV Export
            csv = df_to_export.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger en CSV", csv, f"choch_signaux_{timestamp}.csv", "text/csv")

            # PNG Export
            try:
                image_buf = io.BytesIO()
                dfi.export(df_to_export, image_buf, table_conversion='matplotlib')
                image_buf.seek(0)
                st.download_button("🖼️ Télécharger en PNG", image_buf, f"choch_signaux_{timestamp}.png", "image/png")
            except Exception as e:
                st.warning(f"Erreur lors de l'export PNG : {e}")

            # PDF Export (CORRIGÉ POUR GERER PLUSIEURS PAGES)
            try:
                pdf = FPDF(orientation='L', unit='mm', format='A4')
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()
                pdf.set_font("Arial", 'B', 16)
                pdf.cell(0, 10, "Rapport des Signaux CHoCH", 0, 1, 'C')
                pdf.set_font("Arial", size=10)
                
                rows_per_page = 20 # Ajustez ce nombre selon vos préférences
                for i in range(0, len(df_to_export), rows_per_page):
                    if i > 0: # Si ce n'est pas la première page, on en ajoute une nouvelle
                        pdf.add_page()
                    
                    chunk = df_to_export.iloc[i:i+rows_per_page]
                    
                    # Générer une image temporaire pour ce morceau de tableau
                    img_temp_path = f"temp_chunk_{i}.png"
                    dfi.export(chunk, img_temp_path, table_conversion='matplotlib')
                    
                    # Ajouter l'image au PDF
                    pdf.image(img_temp_path, x=10, y=pdf.get_y(), w=277) # w=297mm - 2*10mm de marge
                    
                    # Supprimer l'image temporaire
                    os.remove(img_temp_path)
                
                # Sauvegarder le PDF
                pdf_output = pdf.output(dest='S').encode('latin-1')
                st.download_button("📄 Télécharger en PDF", pdf_output, f"choch_signaux_{timestamp}.pdf", "application/pdf")

            except Exception as e:
                st.warning(f"Erreur lors de l'export PDF : {e}")

            # --- AFFICHAGE PAR TIMEFRAME ---
            for tf_name in TIME_FRAMES.keys():
                tf_df = full_df[full_df['Timeframe'] == tf_name].copy()
                if not tf_df.empty:
                    tf_df = tf_df.sort_values(by='Heure (UTC)', ascending=False)
                    tf_df.insert(0, ' ', ['⭐'] + [''] * (len(tf_df) - 1))
                    tf_df['Heure (UTC)'] = tf_df['Heure (UTC)'].dt.strftime('%Y-%m-%d %H:%M')
                    st.subheader(f"--- Signaux {tf_name} ---")
                    def color_signal(val): return f'color: {"#089981" if "Bullish" in val else "#f23645"}; font-weight: bold;'
                    def style_order(val): return f'background-color: {"#089981" if val == "Achat" else "#f23645"}; color: white; border-radius: 5px; text-align: center; font-weight: bold;'
                    styled_df = tf_df.drop(columns=['Timeframe']).style.applymap(color_signal, subset=['Signal']).applymap(style_order, subset=['Ordre'])
                    st.dataframe(styled_df, hide_index=True, use_container_width=True)

        # Affichage des scans échoués
        if 'failed_scans' in st.session_state and st.session_state['failed_scans']:
            with st.expander("⚠️ Voir le rapport des scans ayant échoué"):
                st.markdown("\n".join(st.session_state['failed_scans']))

if __name__ == "__main__":
    main()


