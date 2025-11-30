# app.py - VERSION FINALE FONCTIONNELLE - PDF TEXTE LISIBLE PAR TOUTES LES IA
import streamlit as st
import pandas as pd
import numpy as np
import io
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import dataframe_image as dfi

# ReportLab pour PDF avec vrai texte
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

st.set_page_config(page_title="CHoCH Scanner", layout="wide")
st.markdown("<h1 style='text-align:center;color:#1e40af;'>Scanner Change of Character (CHoCH)</h1>", unsafe_allow_html=True)

# ==================== CONFIG ====================
INSTRUMENTS_TO_SCAN = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","USD_CAD","AUD_USD","NZD_USD","EUR_GBP","EUR_JPY","EUR_CHF",
    "EUR_AUD","EUR_CAD","EUR_NZD","GBP_JPY","GBP_CHF","GBP_AUD","GBP_CAD","GBP_NZD","AUD_JPY","AUD_CAD",
    "AUD_CHF","AUD_NZD","CAD_JPY","CAD_CHF","CHF_JPY","NZD_JPY","NZD_CAD","NZD_CHF","XAU_USD","US30_USD",
    "NAS100_USD","SPX500_USD"
]

VOLATILITY_LEVELS = {
    "EUR_USD":"Basse","GBP_USD":"Basse","USD_JPY":"Basse","USD_CHF":"Basse","USD_CAD":"Basse",
    "AUD_USD":"Moyenne","NZD_USD":"Moyenne","EUR_GBP":"Moyenne","EUR_JPY":"Moyenne","EUR_CHF":"Moyenne",
    "EUR_AUD":"Moyenne","EUR_CAD":"Moyenne","EUR_NZD":"Moyenne","GBP_JPY":"Haute","GBP_CHF":"Haute",
    "GBP_AUD":"Haute","GBP_CAD":"Haute","GBP_NZD":"Haute","AUD_JPY":"Haute","AUD_CAD":"Moyenne",
    "AUD_CHF":"Haute","AUD_NZD":"Moyenne","CAD_JPY":"Haute","CAD_CHF":"Haute","CHF_JPY":"Haute",
    "NZD_JPY":"Haute","NZD_CAD":"Moyenne","NZD_CHF":"Haute","XAU_USD":"Très Haute",
    "US30_USD":"Très Haute","NAS100_USD":"Très Haute","SPX500_USD":"Très Haute"
}

TIME_FRAMES = {"H1":"H1", "H4":"H4", "D1":"D", "Weekly":"W"}
FRACTAL_LENGTHS = {"H1":5, "H4":6, "D1":7, "Weekly":8}
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
        return pd.DataFrame([{
            "time": pd.to_datetime(c["time"]),
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"])
        } for c in candles])
    return df
    except:
        return None

def calculate_atr(df, period=14):
    if len(df) < period:
        return 0.0
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    return np.mean(tr[-period:])

def detect_choch(df, tf_code):
    length = FRACTAL_LENGTHS.get(tf_code, 5)
    if len(df) < length*3:
        return None,None,None,None,None
    p = length // 2
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    bull_fractal = [i for i in range(p, len(df)-p) if h[i] == max(h[i-p:i+p+1])]
    bear_fractal = [i for i in range(p, len(df)-p) if l[i] == min(l[i-p:i+p+1])]

    atr = calculate_atr(df)
    state = 0
    last_high = last_low = None

    for i in bull_fractal + bear_fractal:
        i = int(i)
        if i in bull_fractal:
            last_high = h[i]
        if i in bear_fractal:
            last_low = l[i]

        if last_high and c[i] > last_high and c[i-1] <= last_high and state <= 0:
            if len(df)-1-i < RECENT_BARS:
                strength = "Fort" if atr > 0 and (c[i]-last_high) > atr*0.5 else "Moyen"
                return "Bullish CHoCH", df["time"].iloc[i], strength, None, None
            state = 1

        if last_low and c[i] < last_low and c[i-1] >= last_low and state >= 0:
            if len(df)-1-i < RECENT_BARS:
                strength = "Fort" if atr > 0 and (last_low-c[i]) > atr*0.5 else "Moyen"
                return "Bearish CHoCH", df["time"].iloc[i], strength, None, None
            state = -1
    return None,None,None,None,None

# ==================== PDF TEXTE PUR ====================
def make_pdf(df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=20, rightMargin=20, topMargin=30, bottomMargin=30)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Rapport des Signaux CHoCH", styles["Title"]))
    elements.append(Paragraph(f"Généré le {datetime.utcnow().strftime('%d/%m/%Y à %H:%M UTC')}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data, colWidths=[60,40,35,70,45,40,50,50,50,80])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1e40af")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('FONTSIZE',(0,0),(-1,-1),9),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.beige]),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==================== API & SCAN ====================
try:
    api = API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
except:
    st.error("Clé API OANDA manquante dans Secrets !")
    st.stop()

if st.button("Lancer le Scan CHoCH", type="primary", use_container_width=True):
    with st.spinner("Scan en cours sur 124 timeframes..."):
        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_pair = {
                executor.submit(get_data, inst, tf_code): (inst, tf_name)
                for inst in INSTRUMENTS_TO_SCAN
                for tf_name, tf_code in TIME_FRAMES.items()
            }
            for future in as_completed(future_to_pair):
                inst, tf_name = future_to_pair[future]
                df = future.result()
                if df is not None and len(df) > 50:
                    signal, ttime, strength, _, _ = detect_choch(df, TIME_FRAMES[tf_name])
                    if signal:
                        results.append({
                            "Instrument": inst.replace("_","/"),
                            "Timeframe": tf_name,
                            "Ordre": "Achat" if "Bull" in signal else "Vente",
                            "Signal": signal,
                            "Volatilité": VOLATILITY_LEVELS.get(inst, "Moyenne"),
                            "Force": strength or "Moyen",
                            "Heure (UTC)": ttime
                        })

        if results:
            df_result = pd.DataFrame(results).sort_values("Heure (UTC)", ascending=False)
            st.session_state.df = df_result
            st.success(f"{len(df_result)} signaux trouvés !")
        else:
            st.info("Aucun signal récent")

# ==================== AFFICHAGE & DOWNLOADS ====================
if "df" in st.session_state:
    df = st.session_state.df.copy()
    df["Heure (UTC)"] = pd.to_datetime(df["Heure (UTC)"])
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("CSV", df.to_csv(index=False).encode(), f"choch_{ts}.csv", "text/csv")
    with c2:
        buf = io.BytesIO()
        dfi.export(df.style.applymap(lambda x: "color:green" if "Achat" in str(x) else "color:red" if "Vente" in str(x) else "", subset=["Ordre"]), buf)
        buf.seek(0)
        st.download_button("PNG", buf, f"choch_{ts}.png", "image/png")
    with c3:
        pdf_buffer = make_pdf(df)
        st.download_button(
            label="PDF (texte 100% lisible par ChatGPT, Claude, Gemini…)",
            data=pdf_buffer,
            file_name=f"choch_signaux_{ts}.pdf",
            mime="application/pdf"
        )

    st.dataframe(df.style.applymap(lambda x: "color:green;font-weight:bold" if "Achat" in str(x) else "color:red;font-weight:bold" if "Vente" in str(x) else "", subset=["Ordre"]), use_container_width=True, hide_index=True)
