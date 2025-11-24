import streamlit as st
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.exceptions
import time as time_module
from datetime import datetime
from fpdf import FPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Scanner de CHoCH", layout="wide")

# --- CONSTANTES ---
INSTRUMENTS_TO_SCAN = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY", "NZD_JPY", "NZD_CAD", "NZD_CHF",
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
    "XAU_USD": "Très Haute", "US30_USD": "Très Haute", "NAS100_USD": "Très Haute", "SPX500_USD": "Très Haute"
}

TIME_FRAMES = {"H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"}
FRACTAL_LENGTH = 5
RECENT_BARS_THRESHOLD = 10
MAX_WORKERS = 5 # Réduire si vous rencontrez des erreurs de connexion

FRACTAL_LENGTHS_BY_TF = {
    "H1": 5, "H4": 6, "D1": 7, "Weekly": 8
}

# --- FONCTIONS METIER ---

def calculate_atr(df, period=14):
    """Calcule l'ATR pour mesurer la volatilité actuelle"""
    if df is None or len(df) < period:
        return None
    
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Simple Moving Average of TR
    atr = np.mean(tr[-period:])
    
    return atr

def get_oanda_data(api_client, instrument, granularity, count=250, max_retries=3):
    """Récupération robuste des données Oanda"""
    params = {"count": count, "granularity": granularity}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    
    for attempt in range(max_retries):
        try:
            api_client.request(r)
            data = r.response.get('candles')
            if not data:
                return None, f"Aucune bougie reçue."
            
            records = []
            for c in data:
                if c['complete']:
                    records.append({
                        "time": pd.to_datetime(c['time']),
                        "open": float(c['mid']['o']),
                        "high": float(c['mid']['h']),
                        "low": float(c['mid']['l']),
                        "close": float(c['mid']['c'])
                    })
            
            if not records:
                return None, f"Données vides."
            
            return pd.DataFrame(records), "Succès"
            
        except Exception as e:
            if attempt + 1 == max_retries:
                return None, str(e)
            time_module.sleep(1) # Petit délai avant retry
    
    return None, "Erreur inconnue"

def detect_choch_optimized(df, instrument, tf_code, length=None):
    """Détection CHoCH optimisée"""
    if df is None or len(df) < 50: # Besoin d'un minimum de bougies
        return None, None, None
    
    if length is None:
        length = FRACTAL_LENGTHS_BY_TF.get(tf_code, FRACTAL_LENGTH)
    
    p = length // 2
    # Conversion en numpy arrays pour la performance
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    times = df['time'].values
    
    atr = calculate_atr(df)
    
    # Détection vectorisée (partielle) des fractales
    # Note: Une boucle complète reste plus lisible pour la logique séquentielle du CHOCH
    
    upper_fractal_val = None
    lower_fractal_val = None
    upper_fractal_crossed = True
    lower_fractal_crossed = True
    
    os = 0 # Order Structure
    
    choch_signal = None
    choch_time = None
    choch_index = -1
    confirmation_strength = "Neutre"

    # On itère sur les bougies. 
    # Attention: range doit permettre de vérifier i-p et i+p
    start_idx = p
    end_idx = len(df) - p - 1 # On s'arrête un peu avant la fin pour la fractale future
    
    # Tableaux booléens pour identifier les fractales
    is_bull = np.zeros(len(df), dtype=bool)
    is_bear = np.zeros(len(df), dtype=bool)

    # 1. Identifier toutes les fractales d'abord
    for i in range(start_idx, end_idx + 1):
        window_high = highs[i-p : i+p+1]
        window_low = lows[i-p : i+p+1]
        
        if highs[i] == np.max(window_high):
            is_bull[i] = True
        if lows[i] == np.min(window_low):
            is_bear[i] = True

    # 2. Logique séquentielle pour le CHOCH
    # On commence après p pour avoir l'historique
    for i in range(length, len(df)):
        # Mise à jour des niveaux fractals (basé sur la fractale confirmée à i-p)
        prev_idx = i - p
        if prev_idx >= 0:
            if is_bull[prev_idx]:
                upper_fractal_val = highs[prev_idx]
                upper_fractal_crossed = False
            if is_bear[prev_idx]:
                lower_fractal_val = lows[prev_idx]
                lower_fractal_crossed = False
        
        curr_c = closes[i]
        prev_c = closes[i-1]

        # Check Bullish CHOCH
        if upper_fractal_val is not None and not upper_fractal_crossed:
            if curr_c > upper_fractal_val and prev_c <= upper_fractal_val:
                if os == -1: # Changement de structure
                    choch_signal = "Bullish CHoCH"
                    choch_time = times[i]
                    choch_index = i
                    dist = curr_c - upper_fractal_val
                    confirmation_strength = "Fort" if (atr and dist > atr * 0.3) else "Normal"
                os = 1
                upper_fractal_crossed = True

        # Check Bearish CHOCH
        if lower_fractal_val is not None and not lower_fractal_crossed:
            if curr_c < lower_fractal_val and prev_c >= lower_fractal_val:
                if os == 1: # Changement de structure
                    choch_signal = "Bearish CHoCH"
                    choch_time = times[i]
                    choch_index = i
                    dist = lower_fractal_val - curr_c
                    confirmation_strength = "Fort" if (atr and dist > atr * 0.3) else "Normal"
                os = -1
                lower_fractal_crossed = True

    # Vérifier si le signal est récent
    if choch_signal and (len(df) - 1 - choch_index) <= RECENT_BARS_THRESHOLD:
        return choch_signal, pd.to_datetime(choch_time), confirmation_strength
    
    return None, None, None

def scan_wrapper(api_key, instrument, tf_name, tf_code):
    """Wrapper pour l'exécution threadée"""
    try:
        # Création d'une instance API par thread pour éviter les conflits
        client = API(access_token=api_key)
        df, msg = get_oanda_data(client, instrument, tf_code)
        
        if df is not None:
            signal, sig_time, strength = detect_choch_optimized(df, instrument, tf_code)
            if signal:
                return {
                    "Instrument": instrument.replace("_", "/"),
                    "Timeframe": tf_name,
                    "Ordre": "Achat" if "Bullish" in signal else "Vente",
                    "Signal": signal,
                    "Volatilité": VOLATILITY_LEVELS.get(instrument, "-"),
                    "Force": strength,
                    "Heure (UTC)": sig_time
                }
            return None # Pas de signal
        else:
            return {"error": True, "msg": f"{instrument} {tf_name}: {msg}"}
    except Exception as e:
        return {"error": True, "msg": f"Crash sur {instrument}: {str(e)}"}

# --- GENERATION PDF (Native FPDF sans dataframe_image) ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Rapport des Signaux CHoCH', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(df):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
    # Configuration des colonnes
    cols = df.columns.tolist()
    # Ajustement largeurs approx
    col_widths = [30, 25, 20, 40, 25, 20, 35] 
    
    # En-têtes
    pdf.set_font("Arial", 'B', 10)
    for i, col in enumerate(cols):
        # Convertir timestamps en str si besoin
        pdf.cell(col_widths[i], 10, str(col), 1, 0, 'C')
    pdf.ln()
    
    # Données
    pdf.set_font("Arial", size=9)
    for index, row in df.iterrows():
        for i, col in enumerate(cols):
            val = str(row[col])
            # Petite couleur pour Achat/Vente
            if col == "Ordre":
                if val == "Achat":
                    pdf.set_text_color(0, 128, 0)
                else:
                    pdf.set_text_color(200, 0, 0)
            else:
                pdf.set_text_color(0, 0, 0)
                
            pdf.cell(col_widths[i], 10, val, 1, 0, 'C')
        pdf.ln()
        
    return pdf.output(dest='S').encode('latin-1')

# --- MAIN APP ---
def main():
    st.markdown("""
        <h1 style='text-align: center; color: #089981;'>Scanner CHoCH Multi-Timeframe</h1>
        <p style='text-align: center;'>Détecteur de retournement de structure (Fractales + ATR)</p>
        <hr>
    """, unsafe_allow_html=True)

    # Gestion des secrets
    try:
        OANDA_ACCESS_TOKEN = st.secrets["OANDA_ACCESS_TOKEN"]
    except:
        st.warning("⚠️ Token Oanda introuvable. Vérifiez `.streamlit/secrets.toml`")
        OANDA_ACCESS_TOKEN = st.text_input("Ou entrez votre Token API ici:", type="password")
    
    if not OANDA_ACCESS_TOKEN:
        st.stop()

    # Bouton de lancement
    if st.button('🚀 Lancer le Scan', use_container_width=True):
        # Nettoyage
        st.session_state['scan_results'] = None
        st.session_state['failed_scans'] = []

        results = []
        errors = []
        
        # Barre de progression
        prog_bar = st.progress(0)
        status_text = st.empty()
        
        total_tasks = len(INSTRUMENTS_TO_SCAN) * len(TIME_FRAMES)
        completed = 0
        
        # ThreadPool
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for instrument in INSTRUMENTS_TO_SCAN:
                for tf_name, tf_code in TIME_FRAMES.items():
                    futures.append(executor.submit(scan_wrapper, OANDA_ACCESS_TOKEN, instrument, tf_name, tf_code))
            
            for future in as_completed(futures):
                res = future.result()
                if res:
                    if "error" in res:
                        errors.append(res["msg"])
                    else:
                        results.append(res)
                
                completed += 1
                prog_bar.progress(completed / total_tasks)
                status_text.text(f"Analyse: {completed}/{total_tasks}")

        prog_bar.empty()
        status_text.success("✅ Analyse terminée")
        
        if results:
            st.session_state['scan_results'] = pd.DataFrame(results)
        else:
            st.session_state['scan_results'] = pd.DataFrame()
            
        st.session_state['failed_scans'] = errors
        
        # Rerun pour afficher les résultats proprement
        st.rerun()

    # Affichage des résultats
    if 'scan_results' in st.session_state:
        df_res = st.session_state['scan_results']
        
        if df_res is None or df_res.empty:
            st.info("Aucun signal CHoCH détecté sur les 10 dernières bougies.")
        else:
            # Mise en forme des dates
            df_display = df_res.copy()
            df_display['Heure (UTC)'] = pd.to_datetime(df_display['Heure (UTC)']).dt.strftime('%Y-%m-%d %H:%M')

            # Tableau interactif avec style
            st.subheader("📊 Signaux Détectés")
            
            def color_direction(val):
                color = '#d4edda' if val == 'Achat' else '#f8d7da'
                text_color = '#155724' if val == 'Achat' else '#721c24'
                return f'background-color: {color}; color: {text_color}'

            st.dataframe(
                df_display.style.applymap(color_direction, subset=['Ordre']),
                use_container_width=True,
                hide_index=True
            )
            
            # Export CSV
            csv = df_display.to_csv(index=False).encode('utf-8')
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    "📥 Télécharger CSV",
                    csv,
                    "signaux_choch.csv",
                    "text/csv",
                    key='download-csv'
                )
            
            with col2:
                try:
                    pdf_bytes = generate_pdf(df_display)
                    st.download_button(
                        "📄 Télécharger PDF",
                        data=pdf_bytes,
                        file_name="signaux_choch.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Erreur génération PDF: {e}")

        # Affichage des erreurs si besoin
        if st.session_state.get('failed_scans'):
            with st.expander("Voir les erreurs de connexion"):
                for err in st.session_state['failed_scans']:
                    st.text(err)

if __name__ == "__main__":
    main()
