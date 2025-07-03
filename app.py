import streamlit as st
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import time as time_module

# --- CONFIGURATION ---
# Ces valeurs seront récupérées depuis les secrets de Streamlit lors du déploiement
# Exemple de configuration dans le fichier secrets.toml de Streamlit:
# [oanda]
# account_id = "101-xxxxxxxx-xxxx"
# access_token = "ac365b55exxxxxxxxx"

# Liste des instruments à scanner
INSTRUMENTS_TO_SCAN = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
    "XAU_USD",    # Or
    "US30_USD",   # Dow Jones
    "NAS100_USD", # Nasdaq 100
    "SPX500_USD"  # S&P 500
]

# Unités de temps à scanner (Format API Oanda)
TIME_FRAMES = {
    "H1": "H1",
    "H4": "H4",
    "D1": "D",
    "Weekly": "W"
}

# Paramètres de l'indicateur
FRACTAL_LENGTH = 5 # 'length' dans le script TV
RECENT_BARS_THRESHOLD = 3 # Signaler un CHOCH s'il est apparu dans les X dernières bougies

# --- Fonctions de l'application ---

def get_oanda_data(api_client, instrument, granularity, count=250):
    """Récupère les données de chandeliers depuis Oanda."""
    params = {"count": count, "granularity": granularity}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    try:
        api_client.request(r)
        data = r.response.get('candles')
        if not data:
            return None
            
        df = pd.DataFrame([
            {
                "time": pd.to_datetime(candle['time']),
                "open": float(candle['mid']['o']),
                "high": float(candle['mid']['h']),
                "low": float(candle['mid']['l']),
                "close": float(candle['mid']['c'])
            } for candle in data if candle['complete']
        ])
        return df
    except Exception:
        return None

def detect_choch(df, length=5):
    """
    Traduction de la logique Pine Script en Python pour détecter les CHoCH.
    Retourne le type de CHOCH ('Bullish' ou 'Bearish') et la date si détecté récemment.
    """
    if df is None or len(df) < length:
        return None, None

    p = length // 2
    
    df['is_bull_fractal'] = (df['high'] == df['high'].rolling(window=length, center=True, min_periods=length).max())
    df['is_bear_fractal'] = (df['low'] == df['low'].rolling(window=length, center=True, min_periods=length).min())
    
    os = 0  # Orderflow Switch: 0=neutre, 1=haussier, -1=baissier
    upper_fractal_value = None
    lower_fractal_value = None
    
    choch_signal = None
    choch_time = None
    choch_bar_index = -1

    for i in range(p, len(df)):
        if df['is_bull_fractal'].iloc[i-p]:
            upper_fractal_value = df['high'].iloc[i-p]
        if df['is_bear_fractal'].iloc[i-p]:
            lower_fractal_value = df['low'].iloc[i-p]

        current_close = df['close'].iloc[i]
        previous_close = df['close'].iloc[i-1]

        if upper_fractal_value is not None and current_close > upper_fractal_value and previous_close <= upper_fractal_value:
            if os == -1:
                choch_signal = "Bullish CHoCH"
                choch_time = df['time'].iloc[i]
                choch_bar_index = i
            os = 1
            upper_fractal_value = None 

        if lower_fractal_value is not None and current_close < lower_fractal_value and previous_close >= lower_fractal_value:
            if os == 1:
                choch_signal = "Bearish CHoCH"
                choch_time = df['time'].iloc[i]
                choch_bar_index = i
            os = -1
            lower_fractal_value = None
            
    if choch_signal and (len(df) - 1 - choch_bar_index) < RECENT_BARS_THRESHOLD:
        return choch_signal, choch_time
        
    return None, None

def main():
    """Fonction principale de l'application Streamlit."""
    st.set_page_config(page_title="Scanner de CHoCH", layout="wide")
    st.title("📈 Scanner de Change of Character (CHoCH)")
    st.write(f"Ce scanner recherche des signaux de CHoCH survenus dans les **{RECENT_BARS_THRESHOLD}** dernières bougies.")
    st.info("Avertissement : Basé sur l'indicateur 'Market Structure' de LuxAlgo (Licence CC BY-NC-SA 4.0). Cet outil est à but éducatif. Faites toujours vos propres analyses.")

    # Vérification de la configuration des secrets
    if "oanda" not in st.secrets or "access_token" not in st.secrets.oanda or "account_id" not in st.secrets.oanda:
        st.error("Erreur de configuration : Veuillez configurer vos identifiants OANDA dans les secrets de Streamlit.")
        st.code("""
# Exemple de format pour vos secrets :
[oanda]
account_id = "YOUR-ACCOUNT-ID"
access_token = "YOUR-ACCESS-TOKEN"
        """)
        st.stop()

    results_placeholder = st.empty()
    
    if st.button('Lancer le Scan'):
        try:
            api_client = API(access_token=st.secrets.oanda.access_token)
        except Exception as e:
            st.error(f"Erreur d'initialisation de l'API Oanda. Vérifiez vos secrets. Erreur: {e}")
            st.stop()
            
        with st.spinner('Scan en cours... Ceci peut prendre une minute.'):
            results = []
            total_scans = len(INSTRUMENTS_TO_SCAN) * len(TIME_FRAMES)
            progress_bar = st.progress(0)
            progress_status = st.empty()
            i = 0

            for instrument in INSTRUMENTS_TO_SCAN:
                for tf_name, tf_code in TIME_FRAMES.items():
                    i += 1
                    progress_bar.progress(i / total_scans)
                    progress_status.text(f"Scan de {instrument} sur {tf_name}...")
                    
                    df = get_oanda_data(api_client, instrument, tf_code)
                    
                    if df is not None:
                        signal, signal_time = detect_choch(df, length=FRACTAL_LENGTH)
                        if signal:
                            results.append({
                                "Instrument": instrument.replace("_", "/"),
                                "Timeframe": tf_name,
                                "Signal": signal,
                                "Heure (UTC)": signal_time.strftime('%Y-%m-%d %H:%M')
                            })
                    else:
                        st.warning(f"Données non disponibles pour {instrument} sur {tf_name}.")

                    time_module.sleep(0.2) # Pour ne pas surcharger l'API Oanda

            progress_status.success("Scan terminé !")

            if results:
                results_df = pd.DataFrame(results).sort_values(by="Timeframe")
                # Style pour colorer les signaux
                def color_signal(val):
                    color = 'green' if 'Bullish' in val else 'red'
                    return f'color: {color}; font-weight: bold;'
                
                styled_df = results_df.style.applymap(color_signal, subset=['Signal'])
                results_placeholder.dataframe(styled_df, use_container_width=True)
            else:
                results_placeholder.success("✅ Aucun signal de CHoCH récent détecté.")

if __name__ == "__main__":
    main()
