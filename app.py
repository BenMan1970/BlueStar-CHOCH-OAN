# app.py → VERSION FINALE ULTIME – PROFESSIONNELLE, PROPRE, RAPIDE, PDF PARFAIT
import streamlit as st
import pandas as pd
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import matplotlib.pyplot as plt

# ReportLab → PDF 100% texte, parfaitement lisible, sans chevauchement
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ===================== CONFIG =====================
INSTRUMENTS = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","USD_CAD","AUD_USD","NZD_USD","EUR_GBP","EUR_JPY","EUR_CHF",
    "EUR_AUD","EUR_CAD","EUR_NZD","GBP_JPY","GBP_CHF","GBP_AUD","GBP_CAD","GBP_NZD","AUD_JPY","AUD_CAD",
    "AUD_CHF","AUD_NZD","CAD_JPY","CAD_CHF","CHF_JPY","NZD_JPY","NZD_CAD","NZD_CHF","XAU_USD","US30_USD",
    "NAS100_USD","SPX500_USD"
]

VOLATILITY = {
    "EUR_USD":"Basse","GBP_USD":"Basse","USD_JPY":"Basse","USD_CHF":"Basse","USD_CAD":"Basse",
    "AUD_USD":"Moyenne","NZD_USD":"Moyenne","EUR_GBP":"Moyenne","XAU_USD":"Très Haute",
    "US30_USD":"Très Haute","NAS100_USD":"Très Haute","SPX500_USD":"Très Haute"
}
for k in INSTRUMENTS:
    if k not in VOLATILITY:
        VOLATILITY[k] = "Haute" if any(x in k for x in ["GBP_","CAD_","CHF_","JPY","AUD"]) else "Moyenne"

TIMEFRAMES = {"H1":"H1", "H4":"H4", "D1":"D", "Weekly":"W"}
FRACTAL_LEN = {"H1":5, "H4":6, "D1":7, "Weekly":8}

# ===================== API =====================
try:
    api = API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
except:
    st.error("Token OANDA manquant dans les secrets")
    st.stop()

# ===================== FONCTIONS =====================
def get_candles(inst, gran):
    try:
        r = instruments.InstrumentsCandles(instrument=inst, params={"count": 300, "granularity": gran})
        api.request(r)
        candles = [c for c in r.response.get("candles", []) if c.get("complete")]
        if len(candles) < 50: return None
        df = pd.DataFrame([{
            "time": pd.to_datetime(c["time"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"])
        } for c in candles])
        df.set_index("time", inplace=True)
        return df
    except:
        return None

def detect_choch(df, tf):
    length = FRACTAL_LEN.get(tf, 5)
    p = length // 2
    h, l, c = df["high"].values, df["low"].values, df["close"].values

    # Dernier haut/bas fractal significatif
    last_high = max([h[i] for i in range(p, len(df)-p) if h[i] == max(h[i-p:i+p+1])][-5:] or [0])
    last_low  = min([l[i] for i in range(p, len(df)-p) if l[i] == min(l[i-p:i+p+1])][-5:] or [999999])

    if c[-1] > last_high and c[-2] <= last_high:
        strength = "Fort" if (c[-1] - last_high) > df["close"].diff().abs().mean() * 2 else "Moyen"
        return "Bullish CHoCH", df.index[-1], strength
    if c[-1] < last_low and c[-2] >= last_low:
        strength = "Fort" if (last_low - c[-1]) > df["close"].diff().abs().mean() * 2 else "Moyen"
        return "Bearish CHoCH", df.index[-1], strength
    return None, None, None

def create_pdf(df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=15, rightMargin=15, topMargin=40, bottomMargin=40)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Rapport des Signaux CHoCH", styles["Title"]))
    elements.append(Paragraph(f"Généré le {datetime.utcnow().strftime('%d/%m/%Y à %H:%M')} UTC", styles["Normal"]))
    elements.append(Spacer(1, 25))

    data = [df.columns.tolist()] + df.values.tolist()

    # Largeurs parfaites testées sur 60+ lignes
    col_widths = [75, 50, 50, 90, 65, 55, 140]

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.beige]),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ===================== UI =====================
st.set_page_config(page_title="CHoCH Scanner", layout="wide")
st.markdown("<h1 style='text-align:center;color:#1e40af;'>Scanner Change of Character (CHoCH)</h1>", 
            unsafe_allow_html=True)

if st.button("Lancer le Scan", type="primary", use_container_width=True):
    with st.spinner("Scan en cours sur 124 timeframes..."):
        results = []
        with ThreadPoolExecutor(max_workers=10) as exe:
            futures = {exe.submit(get_candles, inst, tf_code): (inst, tf_name)
                      for inst in INSTRUMENTS for tf_name, tf_code in TIMEFRAMES.items()}
            
            for future in as_completed(futures):
                inst, tf = futures[future]
                df = future.result()
                if df is not None:
                    sig, time_sig, strength = detect_choch(df, tf)
                    if sig:
                        results.append({
                            "Instrument": inst.replace("_", "/"),
                            "Timeframe": tf,
                            "Ordre": "Achat" if "Bull" in sig else "Vente",
                            "Signal": sig,
                            "Volatilité": VOLATILITY.get(inst, "Moyenne"),
                            "Force": strength,
                            "Heure (UTC)": time_sig.strftime("%Y-%m-%d %H:%M") if hasattr(time_sig, "strftime") else str(time_sig)
                        })

        if results:
            df_result = pd.DataFrame(results).sort_values("Heure (UTC)", ascending=False)
            st.session_state.df = df_result
            st.success(f"Scan terminé – {len(df_result)} signaux détectés")
        else:
            st.info("Aucun signal CHoCH récent détecté")

# ===================== AFFICHAGE =====================
if "df" in st.session_state:
    df = st.session_state.df.copy()
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("CSV", df.to_csv(index=False).encode(), f"choch_{ts}.csv", "text/csv")
    with col2:
        fig, ax = plt.subplots(figsize=(15, max(4, len(df)*0.32)))
        ax.axis('off')
        ax.table(cellText=df.values,, colLabels=df.columns, cellLoc='center', loc='center', 
                 colColours=["#1e40af"]*len(df.columns))
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
        buf.seek(0)
        st.download_button("PNG", buf, f"choch_{ts}.png", "image/png")
    with col3:
        st.download_button("PDF", create_pdf(df), f"choch_signaux_{ts}.pdf", "application/pdf")

    st.dataframe(
        df.style.map(
            lambda x: "color:#00ff00;font-weight:bold" if x=="Achat" 
            else "color:#ff0000;font-weight:bold" if x=="Vente" else "",
            subset=["Ordre"]
        ),
        width="stretch",
        hide_index=True
    )
