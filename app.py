import streamlit as st
import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import time as time_module
from datetime import datetime
from fpdf import FPDF
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Scanner CHoCH", layout="wide")

# --- CSS POUR FORCER L'AFFICHAGE LARGE ET PROPRE ---
st.markdown("""
<style>
    .stDataFrame { width: 100% !important; }
    /* Cache l'index si nécessaire et ajuste les headers */
    thead tr th:first-child { display:none }
    tbody th { display:none }
</style>
""", unsafe_allow_html=True)

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
MAX_WORKERS = 4

FRACTAL_LENGTHS_BY_TF = { "H1": 5, "H4": 6, "D1": 7, "Weekly": 8 }

# --- FONCTIONS LOGIQUES ---

def calculate_atr(df, period=14):
    if df is None or len(df) < period: return None
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    return np.mean(tr[-period:])

def get_oanda_data(api_client, instrument, granularity, count=250, max_retries=3):
    params = {"count": count, "granularity": granularity}
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    for attempt in range(max_retries):
        try:
            api_client.request(r)
            data = r.response.get('candles')
            if not data: return None, "Aucune donnée"
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
            return pd.DataFrame(records), "Succès"
        except Exception as e:
            if attempt + 1 == max_retries: return None, str(e)
            time_module.sleep(1)
    return None, "Erreur inconnue"

def detect_choch_optimized(df, instrument, tf_code, length=None):
    if df is None or len(df) < 50: return None, None, None
    if length is None: length = FRACTAL_LENGTHS_BY_TF.get(tf_code, FRACTAL_LENGTH)
    
    p = length // 2
    highs, lows, closes, times = df['high'].values, df['low'].values, df['close'].values, df['time'].values
    atr = calculate_atr(df)
    
    upper_val, lower_val = None, None
    upper_crossed, lower_crossed = True, True
    os = 0
    
    choch_sig, choch_t, choch_idx = None, None, -1
    strength = "Neutre"

    is_bull = np.zeros(len(df), dtype=bool)
    is_bear = np.zeros(len(df), dtype=bool)

    for i in range(p, len(df) - p):
        if highs[i] == np.max(highs[i-p : i+p+1]): is_bull[i] = True
        if lows[i] == np.min(lows[i-p : i+p+1]): is_bear[i] = True

    for i in range(length, len(df)):
        prev_idx = i - p
        if prev_idx >= 0:
            if is_bull[prev_idx]: upper_val, upper_crossed = highs[prev_idx], False
            if is_bear[prev_idx]: lower_val, lower_crossed = lows[prev_idx], False
        
        curr, prev = closes[i], closes[i-1]

        if upper_val and not upper_crossed:
            if curr > upper_val and prev <= upper_val:
                if os == -1:
                    choch_sig, choch_t, choch_idx = "Bullish CHoCH", times[i], i
                    strength = "Fort" if (atr and (curr - upper_val) > atr * 0.3) else "Moyen"
                os, upper_crossed = 1, True

        if lower_val and not lower_crossed:
            if curr < lower_val and prev >= lower_val:
                if os == 1:
                    choch_sig, choch_t, choch_idx = "Bearish CHoCH", times[i], i
                    strength = "Fort" if (atr and (lower_val - curr) > atr * 0.3) else "Moyen"
                os, lower_crossed = -1, True

    if choch_sig and (len(df) - 1 - choch_idx) <= RECENT_BARS_THRESHOLD:
        return choch_sig, pd.to_datetime(choch_t), strength
    return None, None, None

def scan_wrapper(api_key, instrument, tf_name, tf_code):
    try:
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
                    "Volatilité": VOLATILITY_LEVELS.get(instrument, "Inconnue"),
                    "Force": strength,
                    "Heure (UTC)": sig_time
                }
            return None
        return {"error": True, "msg": f"{instrument} {tf_name}: {msg}"}
    except Exception as e:
        return {"error": True, "msg": f"Crash {instrument}: {str(e)}"}

# --- FONCTIONS STYLE (Mise à jour pour Pandas récents) ---
def apply_custom_style(df):
    def color_signal(val):
        color = "#089981" if "Bullish" in str(val) else "#f23645" if "Bearish" in str(val) else "black"
        return f'color: {color}; font-weight: bold;'
    
    def style_order(val):
        bg = "#089981" if val == "Achat" else "#f23645"
        return f'background-color: {bg}; color: white; border-radius: 4px; text-align: center; font-weight: bold;'
    
    def style_volatility(val):
        colors = {"Basse": "#089981", "Moyenne": "#FFA500", "Haute": "#FF6B6B", "Très Haute": "#f23645"}
        return f'background-color: {colors.get(val, "white")}; color: white; border-radius: 4px; text-align: center;'

    def style_force(val):
        bg = "#089981" if val == "Fort" else "#FFA500"
        return f'background-color: {bg}; color: white; border-radius: 4px; text-align: center;'

    # Utilisation de .map() au lieu de .applymap() pour éviter les erreurs de dépréciation
    return df.style.map(color_signal, subset=['Signal']) \
                   .map(style_order, subset=['Ordre']) \
                   .map(style_volatility, subset=['Volatilité']) \
                   .map(style_force, subset=['Force']) \
                   .format({'Heure (UTC)': lambda x: x.strftime('%Y-%m-%d %H:%M')})

# --- PDF ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Rapport des Signaux CHoCH', 0, 1, 'C')
        self.ln(5)

def generate_pdf(df_full):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    df_export = df_full.drop(columns=['Top'], errors='ignore') if 'Top' in df_full.columns else df_full
    cols = df_export.columns.tolist()
    widths = [30, 20, 20, 40, 25, 20, 35]
    pdf.set_font("Arial", 'B', 9)
    for i, h in enumerate(cols):
        pdf.cell(widths[i], 8, str(h), 1, 0, 'C')
    pdf.ln()
    pdf.set_font("Arial", size=8)
    for _, row in df_export.iterrows():
        for i, c in enumerate(cols):
            val = str(row[c])
            pdf.set_text_color(0,0,0)
            if c == 'Ordre':
                pdf.set_text_color(0,100,0) if val == 'Achat' else pdf.set_text_color(200,0,0)
            pdf.cell(widths[i], 8, val, 1, 0, 'C')
        pdf.ln()
    return pdf.output(dest='S').encode('latin-1')

# --- MAIN APP ---
def main():
    st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 20px;">
            <h1 style="color: #f23645;">Scanner</h1>
            <h1 style="color: #089981; margin-left: 10px;">CHoCH</h1>
        </div>
    """, unsafe_allow_html=True)

    try:
        OANDA_ACCESS_TOKEN = st.secrets["OANDA_ACCESS_TOKEN"]
    except:
        st.error("Token OANDA manquant dans les secrets.")
        st.stop()

    if st.button('🚀 Lancer le Scan', use_container_width=True):
        st.session_state['scan_results'] = None
        st.session_state['failed_scans'] = []
        
        results, errors = [], []
        total = len(INSTRUMENTS_TO_SCAN) * len(TIME_FRAMES)
        
        bar = st.progress(0)
        status = st.empty()
        done = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for inst in INSTRUMENTS_TO_SCAN:
                for tf_n, tf_c in TIME_FRAMES.items():
                    futures.append(executor.submit(scan_wrapper, OANDA_ACCESS_TOKEN, inst, tf_n, tf_c))
            
            for f in as_completed(futures):
                res = f.result()
                if res:
                    if "error" in res: errors.append(res["msg"])
                    else: results.append(res)
                done += 1
                bar.progress(done / total)
                status.text(f"Scan en cours: {done}/{total}")

        bar.empty()
        status.success("Terminé !")
        st.session_state['scan_results'] = pd.DataFrame(results) if results else pd.DataFrame()
        st.session_state['failed_scans'] = errors
        st.rerun()

    if 'scan_results' in st.session_state:
        df = st.session_state['scan_results']
        
        if df.empty:
            st.info("Aucun signal détecté.")
        else:
            # Export
            c1, c2 = st.columns(2)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            df_export = df.copy()
            csv = df_export.to_csv(index=False).encode('utf-8')
            c1.download_button("📥 CSV", csv, f"scan_{timestamp}.csv", "text/csv", use_container_width=True)
            try:
                pdf_data = generate_pdf(df_export)
                c2.download_button("📄 PDF", pdf_data, f"scan_{timestamp}.pdf", "application/pdf", use_container_width=True)
            except: pass

            # --- AFFICHAGE PAR TIMEFRAME ---
            ordered_tfs = ["H1", "H4", "D1", "Weekly"]
            
            for tf in ordered_tfs:
                tf_df = df[df['Timeframe'] == tf].copy()
                
                if not tf_df.empty:
                    st.markdown(f"### ⏱️ {tf}")
                    
                    tf_df = tf_df.sort_values(by='Heure (UTC)', ascending=False)
                    
                    # Logique de l'étoile
                    tf_df.insert(0, 'Top', '') 
                    if len(tf_df) > 0:
                        tf_df.iloc[0, tf_df.columns.get_loc('Top')] = '⭐'
                    
                    tf_df_display = tf_df.drop(columns=['Timeframe'])
                    
                    # Style
                    styled_df = apply_custom_style(tf_df_display)
                    
                    # Calcul Hauteur pour éviter le scroll interne
                    # 38px par ligne approx + 40px header
                    height_calc = (len(tf_df) + 1) * 38 

                    st.dataframe(
                        styled_df, 
                        use_container_width=True, 
                        hide_index=True,
                        height=height_calc
                    )
                    st.markdown("---")

        if st.session_state.get('failed_scans'):
            with st.expander("Voir les erreurs techniques"):
                st.write(st.session_state['failed_scans'])

if __name__ == "__main__":
    main()
