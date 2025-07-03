import streamlit as st
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import time as time_module

# --- CONFIGURATION ---
# REMPLACEZ AVEC VOS PROPRES IDENTIFIANTS OANDA
OANDA_ACCOUNT_ID = "101-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # REMPLACEZ ICI
OANDA_ACCESS_TOKEN = "ac365b55exxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" # REMPLACEZ ICI

# Liste des instruments à scanner
INSTRUMENTS_TO_SCAN = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
    "XAU_USD",  # Or
    "US30_USD", # Dow Jones
    "NAS100_USD",# Nasdaq 100
    "SPX500_USD" # S&P 500
]

# Unités de temps à scanner (Format API Oanda)
TIME_FRAMES = {
    "H1": "H1",
    "H4": "H4",
    "D1": "D",
    "Weekly": "W"
}

# Paramètres de l'indicateur (comme dans TradingView)
FRACTAL_LENGTH = 5 # 'length' dans le script TV
RECENT_BARS_THRESHOLD = 3 # On ne signale un CHOCH que s'il est apparu dans les X dernières bougies

# --- FIN DE LA CONFIGURATION ---

# Initialisation de l'API Oanda
# Attention : Ne partagez jamais votre token d'accès publiquement.
# Il est préférable d'utiliser des variables d'environnement pour plus de sécurité.
try:
    api = API(access_token=OANDA_ACCESS_TOKEN)
except Exception as e:
    st.error(f"Erreur de connexion à l'API Oanda. Vérifiez votre token. Erreur: {e}")
    st.stop()


def get_oanda_data(instrument, granularity, count=250):
    """Récupère les données de chandeliers depuis Oanda."""
    params = {
        "count": count,
        "granularity": granularity
    }
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    try:
        api.request(r)
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
    except Exception as e:
        # st.warning(f"Impossible de récupérer les données pour {instrument} sur {granularity}: {e}")
        return None

def detect_choch(df, length=5):
    """
    Traduction de la logique Pine Script en Python pour détecter les CHoCH.
    Retourne le type de CHOCH ('Bullish' ou 'Bearish') et la date si détecté récemment, sinon None.
    """
    if df is None or len(df) < length:
        return None, None

    p = length // 2
    
    # Détection des fractales (simplifiée pour la performance en Python)
    # Une fractale haussière est un point haut entouré de points hauts plus bas.
    # Une fractale baissière est un point bas entouré de points bas plus hauts.
    df['is_bull_fractal'] = (df['high'] == df['high'].rolling(window=length, center=True, min_periods=length).max())
    df['is_bear_fractal'] = (df['low'] == df['low'].rolling(window=length, center=True, min_periods=length).min())
    
    os = 0  # Orderflow Switch: 0=neutre, 1=haussier, -1=baissier
    upper_fractal_value = None
    lower_fractal_value = None
    
    choch_signal = None
    choch_time = None
    choch_bar_index = -1

    # On itère sur les bougies pour simuler le comportement de TradingView
    for i in range(p, len(df)):
        # Mise à jour des dernières fractales valides
        if df['is_bull_fractal'].iloc[i-p]:
            upper_fractal_value = df['high'].iloc[i-p]
            
        if df['is_bear_fractal'].iloc[i-p]:
            lower_fractal_value = df['low'].iloc[i-p]

        # Logique de détection du CHoCH
        current_close = df['close'].iloc[i]
        previous_close = df['close'].iloc[i-1]

        # 1. Vérification de la cassure haussière (crossover)
        if upper_fractal_value is not None and current_close > upper_fractal_value and previous_close <= upper_fractal_value:
            # Est-ce un CHoCH ? (Changement de caractère)
            if os == -1:
                choch_signal = "Bullish CHoCH"
                choch_time = df['time'].iloc[i]
                choch_bar_index = i
            # Mise à jour de l'état du marché à "haussier"
            os = 1
            # On "consomme" la fractale pour éviter de la redéclencher
            upper_fractal_value = None 

        # 2. Vérification de la cassure baissière (crossunder)
        if lower_fractal_value is not None and current_close < lower_fractal_value and previous_close >= lower_fractal_value:
            # Est-ce un CHoCH ? (Changement de caractère)
            if os == 1:
                choch_signal = "Bearish CHoCH"
                choch_time = df['time'].iloc[i]
                choch_bar_index = i
            # Mise à jour de l'état du marché à "baissier"
            os = -1
            # On "consomme" la fractale
            lower_fractal_value = None
            
    # On retourne le signal seulement s'il est récent
    if choch_signal and (len(df) - 1 - choch_bar_index) < RECENT_BARS_THRESHOLD:
        return choch_signal, choch_time
        
    return None, None


def main():
    """Fonction principale de l'application Streamlit."""
    st.set_page_config(page_title="Scanner de CHoCH", layout="wide")
    st.title("📈 Scanner de Change of Character (CHoCH)")
    st.write(f"Basé sur la logique de l'indicateur 'Market Structure' de LuxAlgo. Un signal est affiché s'il est apparu dans les **{RECENT_BARS_THRESHOLD}** dernières bougies.")
    st.info("Avertissement : Ce script est fourni à titre éducatif uniquement. Il est basé sur une interprétation du script PineScript. Faites toujours vos propres recherches. La licence de l'indicateur original (CC BY-NC-SA 4.0) s'applique à la logique utilisée.")

    # Espace pour afficher le tableau des résultats
    results_placeholder = st.empty()
    
    if st.button('Lancer le Scan'):
        with st.spinner('Scan en cours... Veuillez patienter.'):
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
                    
                    # Récupération des données
                    df = get_oanda_data(instrument, tf_code)
                    
                    if df is not None:
                        # Détection du CHoCH
                        signal, signal_time = detect_choch(df, length=FRACTAL_LENGTH)
                        
                        if signal:
                            results.append({
                                "Instrument": instrument,
                                "Timeframe": tf_name,
                                "Signal": signal,
                                "Heure (UTC)": signal_time.strftime('%Y-%m-%d %H:%M')
                            })
                    else:
                        st.warning(f"Données non disponibles pour {instrument} sur {tf_name}.")

                    time_module.sleep(0.2) # Petite pause pour ne pas surcharger l'API Oanda

            progress_status.success("Scan terminé !")

            if results:
                results_df = pd.DataFrame(results)
                results_placeholder.table(results_df)
            else:
                results_placeholder.success("✅ Aucun signal de CHoCH récent détecté sur les paires et unités de temps scannées.")

if __name__ == "__main__":
    main()
