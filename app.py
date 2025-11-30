# app.py — Version finale fonctionnelle + PDF 100% texte lisible par toutes les IA
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

# ReportLab → PDF avec vrai texte sélectionnable
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ==================== CONFIG ====================
INSTRUMENTS = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","USD_CAD","AUD_USD","NZD_USD","EUR_GBP","EUR_JPY","EUR_CHF",
    "EUR_AUD","EUR_CAD","EUR_NZD","GBP_JPY","GBP_CHF","GBP_AUD","GBP_CAD","GBP_NZD","AUD_JPY","AUD_CAD",
    "AUD_CHF","AUD_NZD","CAD_JPY","CAD_CHF","CHF_JPY","NZD_JPY","NZD_CAD","NZD_CHF","XAU_USD",
    "US30_USD","NAS100_USD","SPX500_USD"
]

VOLATILITY = {
    "EUR_USD":"Basse","GBP_USD":"Basse","USD_JPY":"Basse","USD_CHF":"Basse","USD_CAD":"Basse",
    "AUD_USD":"Moyenne","NZD_USD":"Moyenne","EUR_GBP":"Moyenne","EUR_JPY":"Moyenne","EUR_CHF":"Moyenne",
    "EUR_AUD":"Moyenne","EUR_CAD":"Moyenne","EUR_NZD":"Moyenne","GBP_JPY":"Haute","GBP_CHF":"Haute",
    "GBP_AUD":"Haute","GBP_CAD":"Haute","GBP_NZD":"Haute","AUD_JPY":"Haute","AUD_CAD":"Moyenne",
    "AUD_CHF":"Haute","AUD_NZD":"Moyenne","CAD_JPY":"Haute","CAD_CHF":"Haute","CHF_JPY":"Haute",
    "NZD_JPY":"Haute","NZD_CAD":"Moyenne","NZD_CHF":"Haute","XAU_USD":"Très Haute",
    "US30_USD":"Très Haute","NAS100_USD":"Très Haute","SPX500_USD":"Très Haute"
}

TIMEFRAMES = {"H1":"H1", "H4":"H4", "D1":"D", "Weekly":"W"}
FRACTAL_LEN = {"H1":5, "H4":6, "D1":7, "Weekly":8}
RECENT = 12

# ==================== FONCTIONS ====================
def get_candles(instrument, gran):
    try:
        params = {"count": 300, "granularity": gran}
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

def detect_choch(df, tf):
    length = FRACTAL_LEN.get(tf, 5)
    if len(df) < 50:
        return None, None, None
    p = length // 2
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    # Derniers fractals
    last_high = None
    last_low = None
    for i in range(p, len(df)-p):
        if h[i] == max(h[i-p:i+p+1]):
            last_high = h[i]
        if l[i] == min(l[i-p:i+p+1]):
            last_low = l[i]

    if last_high and c[-1] > last_high and c[-2] <= last_high and (len(df)-df.index[-1].to_pydatetime()).days < 10:
        return "Bullish CHoCH", df.index[-1], "Fort" if abs(c[-1]-last_high) > df["close"].diff().abs().mean()*3 else "Moyen"
    if last_low and c[-1] < last_low and c[-2] >= last_low and (len(df)-df.index[-1].to_pydatetime()).days < 10:
        return "Bearish CHoCH", df.index[-1], "Fort" if abs(last_low-c[-1]) > df["close"].diff().abs().mean()*3 else "Moyen"
    return None, None, None

# ==================== PDF TEXTE PUR ====================
def create_pdf(df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=20, rightMargin=20, topMargin=30, bottomMargin=30)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Rapport des Signaux CHoCH", styles["Title"]))
    elements.append(Paragraph(f"Généré le {datetime.utcnow().strftime('%d/%m/%Y %H:%M')} UTC", styles["Normal"]))
    elements.append(Spacer(1, 20))

    data = [df.columns.tolist()] + df.values.tolist()
    table = Table(data, colWidths=[55,35,30,65,40,35,45,45,45,80])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1e40af")),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.5,colors.grey),
        ('FONTSIZE',(0,0),(-1,-1),9),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.lightgrey]),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ==================== API ====================
try:
    api = API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
except:
    st.error("OANDA_ACCESS_TOKEN manquant dans Secrets")
    st.stop()

# ==================== UI ====================
st.set_page_config(page_title="CHoCH Scanner", layout="wide")
st.markdown("<h1 style='text-align:center;color:#1e40af;'>Scanner Change of Character (CHoCH)</h1>", unsafe_allow_html=True)

if st.button("Lancer le Scan", type="primary", use_container_width=True):
    with st.spinner("Scan en cours…"):
        results = []
        with ThreadPoolExecutor(max_workers=8) as exe:
            futures = {exe.submit(get_candles, inst, code): (inst, name)
                       for inst in INSTRUMENTS for name, code in TIMEFRAMES.items()}
            for f in as_completed(futures):
                inst, tf = futures[f]
                df = f.result()
                if df is not None:
                    sig, time_sig, strength = detect_choch(df, tf)
                    if sig:
                        results.append({
                            "Instrument": inst.replace("_","/"),
                            "Timeframe": tf,
                            "Ordre": "Achat" if "Bull" in sig else "Vente",
                            "Signal": sig,
                            "Volatilité": VOLATILITY.get(inst, "Moyenne"),
                            "Force": strength or "Moyen",
                            "Heure (UTC)": time_sig
                        })

        if results:
            df_res = pd.DataFrame(results).sort_values("Heure (UTC)", ascending=False)
            st.session_state.df = df_res
            st.success(f"{len(df_res)} signaux trouvés !")
        else:
            st.info("Aucun signal récent")

if "df" in st.session_state:
    df = st.session_state.df.copy()
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("CSV", df.to_csv(index=False).encode(), f"choch_{ts}.csv", "text/csv")
    with c2:
        buf = io.BytesIO()
        dfi.export(df, buf)
        buf.seek(0)
        st.download_button("PNG", buf, f"choch_{ts}.png", "image/png")
    with c3:
        pdf_buf = create_pdf(df)
        st.download_button(
            label="PDF (texte 100% lisible par IA)",
            data=pdf_buf,
            file_name=f"choch_signaux_{ts}.pdf",
            mime="application/pdf"
        )

    # Style joli
    st.dataframe(df.style.applymap(
        lambda row: ["background:#90ee90" if row.Ordre=="Achat" else "background:#ffcccb" for _ in row], axis=1
    ), use_container_width=True, hide_index=True)
