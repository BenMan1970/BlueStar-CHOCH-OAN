import streamlit as st
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import time as time_module

# --- CONFIGURATION DES PARAMETRES DU SCANNER ---
INSTRUMENTS_TO_SCAN = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
    "XAU_USD",    # Or
    "US30_USD",   # Dow Jones
    "NAS100_USD", # Nasdaq 100
    "SPX500_USD"  # S&P 500
]
TIME_FRAMES = {
    "H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"
}
FRACTAL_LENGTH = 5
RECENT_BARS_THRESHOLD = 3

# --- Fonctions de l'application (inchangées) ---

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
        return df
    except Exception:
        return None

def detect_choch(df, length=5):
    if df is None or len(df) < length: return None, None
    p = length // 2
    df['is_bull_fractal'] = (df['high'] == df['high'].rolling(window=length, center=True, min_periods=length).max())
    df['is_bear_fractal'] = (df['low'] == df['low'].rolling(window=length, center=True, min_periods=length).min())
    os, upper_fractal_value, lower_fractal_value = 0, None, None
    choch_signal, choch_time, choch_bar_index = None, None, -1
    for i in range(p, len(df)):
        if df['is_bull_fractal'].iloc[i-p]: upper_fractal_value = df['high'].iloc[i-p]
        if df['is_bear_fractal'].iloc[i-p]: lower_fractal_value = df['low'].iloc[i-p]
        current_close, previous_close = df['close'].iloc[i], df['close'].iloc[i-1]
        if upper_fractal_value is not None and current_close > upper_fractal_value and previous_close <= upper_fractal_value:
            if os == -1: choch_signal, choch_time, choch_bar_index = "Bullish CHoCH", df['time'].iloc[i], i
            os, upper_fractal_value = 1, None
        if lower_fractal_value is not None and current_close < lower_fractal_value and previous_close >= lower_fractal_value:
            if os == 1: choch_signal, choch_time, choch_bar_index = "Bearish CHoCH", df['time'].iloc[i], i
            os, lower_fractal_value = -1, None
    if choch_signal and (len(df) - 1 - choch_bar_index) < RECENT_BARS_THRESHOLD:
        return choch_signal, choch_time
    return None, None

def main():
    """Fonction principale de l'application Streamlit."""
    st.set_page_config(page_title="Scanner de CHoCH", layout="wide")

    st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 10px;">
                <path d="M4 4V8H8" stroke="#f23645" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M4 12V20H20V4H12" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                <path d="M8 4L4 8" stroke="#089981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <h1 style="margin: 0;">Scanner de Change of Character (CHoCH)</h1>
        </div>
    """, unsafe_allow_html=True)

    st.write(f"Ce scanner recherche des signaux de CHoCH survenus dans les **{RECENT_BARS_THRESHOLD}** dernières bougies.")
    st.info("Avertissement : Basé sur l'indicateur 'Market Structure' de LuxAlgo (Licence CC BY-NC-SA 4.0). Cet outil est à but éducatif.")
    
    try:
        OANDA_ACCESS_TOKEN = st.secrets["OANDA_ACCESS_TOKEN"]
    except KeyError:
        st.error("Erreur : Veuillez configurer OANDA_ACCESS_TOKEN dans les secrets de Streamlit.")
        st.stop()

    results_placeholder = st.empty()
    
    if st.button('Lancer le Scan'):
        # ... la logique de scan reste identique ...
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
                                "Signal": signal, "Heure (UTC)": signal_time.strftime('%Y-%m-%d %H:%M')
                            })
                    else: st.warning(f"Données non disponibles pour {instrument} sur {tf_name}.")
                    time_module.sleep(0.2)
            
            progress_status.success("Scan terminé !")
            
            # --- PARTIE CORRIGÉE POUR L'AFFICHAGE ---
            if results:
                column_order = ["Instrument", "Timeframe", "Ordre", "Signal", "Heure (UTC)"]
                results_df = pd.DataFrame(results)[column_order]

                def color_signal(val): return f'color: {"#089981" if "Bullish" in val else "#f23645"}; font-weight: bold;'
                def style_order(val): return f'background-color: {"#089981" if val == "Achat" else "#f23645"}; color: white; border-radius: 5px; text-align: center; font-weight: bold;'
                
                styled_df = results_df.style.applymap(color_signal, subset=['Signal'])\
                                            .applymap(style_order, subset=['Ordre'])

                # Étape 1 : Afficher SEULEMENT le tableau dans son conteneur.
                table_html = styled_df.to_html(index=False)
                results_placeholder.markdown(f'<div id="results-table-container">{table_html}</div>', unsafe_allow_html=True)
                
                # Étape 2 : Préparer et afficher SEPARÉMENT le bouton et son script.
                export_button_html = """
                <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
                <script>
                function exportTableToImage() {
                    const tableElement = document.getElementById('results-table-container');
                    html2canvas(tableElement, {scale: 2, useCORS: true, backgroundColor: '#0E1117'}).then(canvas => {
                        const link = document.createElement('a');
                        link.download = 'choch_scanner_results.png';
                        link.href = canvas.toDataURL('image/png');
                        link.click();
                    });
                }
                </script>
                <button onclick="exportTableToImage()" style="padding: 8px 12px; border-radius: 8px; border: 1px solid #4A4A4A; background-color: #0E1117; color: white; cursor: pointer; margin-top: 20px;">
                    Exporter en Image
                </button>
                """
                # On utilise st.markdown ici pour ajouter le bouton SOUS le tableau.
                st.markdown(export_button_html, unsafe_allow_html=True)

            else:
                results_placeholder.success("✅ Aucun signal de CHoCH récent détecté.")

if __name__ == "__main__":
    main()
