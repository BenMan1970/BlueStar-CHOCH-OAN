# app.py → VERSION FINALE STABLE & IA-FRIENDLY (à garder pour toujours)
import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import dataframe_image as dfi
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# === CONFIG ===
INSTRUMENTS = ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","USD_CAD","AUD_USD","NZD_USD","EUR_GBP","EUR_JPY","EUR_CHF",
               "EUR_AUD","EUR_CAD","EUR_NZD","GBP_JPY","GBP_CHF","GBP_AUD","GBP_CAD","GBP_NZD","AUD_JPY","AUD_CAD",
               "AUD_CHF","AUD_NZD","CAD_JPY","CAD_CHF","CHF_JPY","NZD_JPY","NZD_CAD","NZD_CHF","XAU_USD","US30_USD",
               "NAS100_USD","SPX500_USD"]

VOL = {"EUR_":"Basse","GBP_":"Basse","USD_JPY":"Basse","USD_CHF":"Basse","USD_CAD":"Basse",
       "AUD_":"Moyenne","NZD_":"Moyenne","EUR_GBP":"Moyenne","XAU_USD":"Très Haute",
       "US30_USD":"Très Haute","NAS100_USD":"Très Haute","SPX500_USD":"Très Haute"}
for k in INSTRUMENTS:
    for prefix, level in [("EUR_","Moyenne"),("GBP_","Haute"),("AUD_","Haute"),("NZD_","Haute"),("CAD_","Haute"),("CHF_","Haute")]:
        if k.startswith(prefix): VOL[k] = level

TIMEFRAMES = {"H1":"H1","H4":"H4","D1":"D","Weekly":"W"}

# === FONCTIONS ===
def get_data(inst, gran):
    try:
        r = instruments.InstrumentsCandles(instrument=inst, params={"count":300,"granularity":gran})
        api.request(r)
        candles = [c for c in r.response.get("candles",[]) if c.get("complete")]
        df = pd.DataFrame([{"time":pd.to_datetime(c["time"]),
                            "high":float(c["mid"]["h"]),"low":float(c["mid"]["l"]),"close":float(c["mid"]["c"])} 
                           for c in candles])
        return df if len(df)>50 else None
    except:
        return None

def has_choch(df):
    if len(df)<30: return None,None,None
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    recent_high = max(h[-25:-5])
    recent_low  = min(l[-25:-5])
    if c[-1]>recent_high and c[-2]<=recent_high: return "Bullish CHoCH", df.index[-1], "Fort"
    if c[-1]<recent_low  and c[-2]>=recent_low : return "Bearish CHoCH", df.index[-1], "Fort"
    return None,None,None

def make_pdf(df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=15, rightMargin=15)
    elements = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph("Rapport des Signaux CHoCH", styles["Title"]))
    elements.append(Paragraph(f"Généré le {datetime.utcnow():%d/%m/%Y %H:%M} UTC", styles["Normal"]))
    elements.append(Spacer(1,20))
    data = [df.columns.tolist()] + df.values.tolist()
    t = Table(data)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1e3a8a")),
                            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                            ('GRID',(0,0),(-1,-1),1,colors.grey),
                            ('FONTSIZE',(0,0),(-1,-1),10),
                            ('ALIGN',(0,0),(-1,-1),'CENTER')]))
    elements.append(t)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# === API ===
try:
    api = API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
except:
    st.error("Token OANDA manquant")
    st.stop()

# === UI ===
st.set_page_config(page_title="CHoCH Scanner", layout="wide")
st.markdown("# Scanner Change of Character (CHoCH)", unsafe_allow_html=True)

if st.button("Lancer le Scan", type="primary", use_container_width=True):
    with st.spinner("Scan en cours..."):
        results = []
        with ThreadPoolExecutor(max_workers=8) as exe:
            futures = {exe.submit(get_data, i, c): (i,n) for i in INSTRUMENTS for n,c in TIMEFRAMES.items()}
            for f in as_completed(futures):
                i,n = futures[f]
                df = f.result()
                if df is not None:
                    sig,time_sig,strength = has_choch(df)
                    if sig:
                        results.append({"Instrument":i.replace("_","/"),"Timeframe":n,"Ordre":"Achat" if "Bull" in sig else "Vente",
                                         "Signal":sig,"Volatilité":VOL.get(i,"Moyenne"),"Force":strength,
                                         "Heure (UTC)":time_sig})
        if results:
            df = pd.DataFrame(results).sort_values("Heure (UTC)", ascending=False)
            st.session_state.df = df
            st.success(f"{len(df)} signaux détectés !")
        else:
            st.info("Aucun signal récent")

if "df" in st.session_state:
    df = st.session_state.df.copy()
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    c1,c2,c3 = st.columns(3)
    with c1: st.download_button("CSV", df.to_csv(index=False).encode(), f"choch_{ts}.csv")
    with c2:
        buf=io.BytesIO()
        dfi.export(df, buf)
        buf.seek(0)
        st.download_button("PNG", buf, f"choch_{ts}.png", "image/png")
    with c3:
        st.download_button("PDF (100% lisible par IA)", make_pdf(df), f"choch_signaux_{ts}.pdf", "application/pdf")
    st.dataframe(df, use_container_width=True, hide_index=True)
