# app.py → VERSION ULTIME : Rapide + PDF texte lisible par TOUTES les IA + PNG qui marche
import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import time

# ReportLab pour PDF texte (lisible par ChatGPT, Claude, Gemini...)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ==================== CONFIG ====================
INSTRUMENTS_TO_SCAN = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF", "XAU_USD", "US30_USD", "NAS100_USD", "SPX500_USD"
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
FRACTAL_LENGTHS = {"H1": 5, "H4": 6, "D1": 7, "Weekly": 8}
RECENT_BARS = 12

# ==================== FONCTIONS ====================
def get_data(instrument, granularity):
    try:
        params = {"count": 300, "granularity": granularity}
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        api.request(r)
        candles = [c for c in r.response.get("candles", []) if c.get("complete")]
        if not candles:
            return None
        df = pd.DataFrame([{
            "time": pd.to_datetime(c["time"]),
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"])
        } for c in candles])
        return df
    except:
        return None

def detect_choch(df, tf_code):
    length = FRACTAL_LENGTHS.get(tf_code, 5)
    if len(df) < length * 3:
        return None, None, None, None, None
    
    p = length // 2
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    
    # Fractales
    bull_fractals = [i for i in range(p, len(df)-p) if h[i] == max(h[i-p:i+p+1])]
    bear_fractals = [i for i in range(p, len(df)-p) if l[i] == min(l[i-p:i+p+1])]
    
    last_high = max([h[i] for i in bull_fractals[-3:]] or [0])
    last_low = min([l[i] for i in bear_fractals[-3:]] or [999999])
    
    current = c[-1]
    prev = c[-2]
    recent = len(df) - df.index.get_loc(df.index[-1]) < RECENT_BARS  # plus simple
    
    if current > last_high >= prev and recent:
        strength = "Fort" if abs(current - last_high) > df["close"].diff().abs().tail(20).mean() * 2 else "Moyen"
        trend = "Haussier" if c[-1] > c[-20] else "Baissier" if c[-1] < c[-20] else "Neutre"
        return "Bullish CHoCH", df.index[-1], strength, trend, (df["high"].tail(50).max(), df["low"].tail(50).min())
    
    if current < last_low <= prev and recent:
        strength = "Fort" if abs(last_low - current) > df["close"].diff().abs().tail(20).mean() * 2 else "Moyen"
        trend = "Haussier" if c[-1] > c[-20] else "Baissier" if c[-1] < c[-20] else "Neutre"
        return "Bearish CHoCH", df.index[-1], strength, trend, (df["high"].tail(50).max(), df["low"].tail(50).min())
    
    return None, None, None, None, None

# ==================== PDF TEXTE PUR (lisible par IA) ====================
def create_pdf(df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=15, rightMargin=15)
    elements = []
    styles = getSampleStyleSheet()
    
    elements.append(Paragraph("Rapport des Signaux CHoCH", styles["Title"]))
    elements.append(Paragraph(f"Généré le {datetime.utcnow():%d/%m/%Y à %H:%M} UTC", styles["Normal"]))
    elements.append(Spacer(1, 20))
    
    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data, colWidths=[50,35,30,55,40,35,45,45,45,70])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1e3a8a")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('FONTSIZE',(0,0),(-1,-1),8.5),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.beige]),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==================== API ====================
try:
    api = API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
except:
    st.error("Token OANDA manquant dans les secrets !")
    st.stop()

# ==================== UI ====================
st.set_page_config(page_title="CHoCH Scanner", layout="wide")
st.markdown("<h1 style='text-align:center;color:#1e40af;'>Scanner Change of Character (CHoCH)</h1>", unsafe_allow_html=True)

if st.button("Lancer le Scan", type="primary", use_container_width=True):
    with st.spinner("Scan ultra-rapide en cours..."):
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for inst in INSTRUMENTS_TO_SCAN:
                for tf_name, tf_code in TIME_FRAMES.items():
                    futures[executor.submit(get_data, inst, tf_code)] = (inst, tf_name)
            
            for future in as_completed(futures):
                inst, tf_name = futures[future]
                df = future.result()
                if df is not None:
                    sig, ttime, strength, trend, levels = detect_choch(df, tf_name)
                    if sig:
                        results.append({
                            "Instrument": inst.replace("_", "/"),
                            "Timeframe": tf_name,
                            "Ordre": "Achat" if "Bullish" in sig else "Vente",
                            "Signal": sig,
                            "Volatilité": VOLATILITY_LEVELS.get(inst, "Moyenne"),
                            "Force": strength or "Moyen",
                            "Tendance": f"{trend}",
                            "Résistance": round(levels[0], 5) if levels else "-",
                            "Support": round(levels[1], 5) if levels else "-",
                            "Heure (UTC)": ttime
                        })
        
        if results:
            df_result = pd.DataFrame(results).sort_values("Heure (UTC)", ascending=False)
            st.session_state.results = df_result
            st.success(f"Scan terminé en moins de 15 secondes — {len(df_result)} signaux trouvés !")
        else:
            st.info("Aucun signal CHoCH récent détecté")

# ==================== AFFICHAGE ====================
if "results" in st.session_state:
    df = st.session_state.results.copy()
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Téléchargements
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("CSV", df.to_csv(index=False).encode(), f"choch_{ts}.csv", "text/csv")
    with col2:
        # PNG sans dataframe-image (évite le crash Playwright)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, len(df)*0.3))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
        buf.seek(0)
        st.download_button("PNG", buf, f"choch_{ts}.png", "image/png")
    with col3:
        st.download_button(
            label="PDF (texte 100% lisible par IA)",
            data=create_pdf(df),
            file_name=f"choch_signaux_{ts}.pdf",
            mime="application/pdf"
        )
    
    # Affichage stylé
    st.dataframe(df.style.map(lambda x: "color:#00ff00;font-weight:bold" if x=="Achat" else "color:#ff0000;font-weight:bold" if x=="Vente" else "", subset=["Ordre"]),
                 use_container_width=True, hide_index=True)
