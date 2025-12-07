# app.py → VERSION FINALE AVEC MONTHLY (5 TIMEFRAMES)
import streamlit as st
import pandas as pd
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import matplotlib.pyplot as plt

# ReportLab → PDF 100% texte sélectionnable, propre et professionnel
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ===================== CONFIG COMPLÈTE (PLATINE + MONTHLY) =====================
INSTRUMENTS = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","USD_CAD","AUD_USD","NZD_USD",
    "EUR_GBP","EUR_JPY","EUR_CHF","EUR_AUD","EUR_CAD","EUR_NZD",
    "GBP_JPY","GBP_CHF","GBP_AUD","GBP_CAD","GBP_NZD",
    "AUD_JPY","AUD_CAD","AUD_CHF","AUD_NZD","CAD_JPY","CAD_CHF","CHF_JPY",
    "NZD_JPY","NZD_CAD","NZD_CHF",
    "XAU_USD",      # Or
    "XPT_USD",      # Platine
    "US30_USD",     # Dow Jones
    "NAS100_USD",   # Nasdaq
    "SPX500_USD"    # S&P 500
]

VOLATILITY = {
    "EUR_USD":"Basse","GBP_USD":"Basse","USD_JPY":"Basse","USD_CHF":"Basse","USD_CAD":"Basse",
    "AUD_USD":"Moyenne","NZD_USD":"Moyenne","EUR_GBP":"Moyenne",
    "EUR_JPY":"Moyenne","EUR_CHF":"Moyenne","EUR_AUD":"Moyenne","EUR_CAD":"Moyenne","EUR_NZD":"Moyenne",
    "GBP_JPY":"Haute","GBP_CHF":"Haute","GBP_AUD":"Haute","GBP_CAD":"Haute","GBP_NZD":"Haute",
    "AUD_JPY":"Haute","AUD_CAD":"Moyenne","AUD_CHF":"Haute","AUD_NZD":"Moyenne",
    "CAD_JPY":"Haute","CAD_CHF":"Haute","CHF_JPY":"Haute","NZD_JPY":"Haute",
    "NZD_CAD":"Moyenne","NZD_CHF":"Haute",
    "XAU_USD":"Très Haute",     # Or
    "XPT_USD":"Très Haute",     # Platine
    "US30_USD":"Très Haute",    # Dow
    "NAS100_USD":"Très Haute",  # Nasdaq
    "SPX500_USD":"Très Haute"   # S&P 500
}

# ← MONTHLY AJOUTÉ ICI
TIMEFRAMES = {"H1":"H1", "H4":"H4", "D1":"D", "Weekly":"W", "Monthly":"M"}
FRACTAL_LEN = {"H1":5, "H4":6, "D1":7, "Weekly":8, "Monthly":9}

# ===================== API =====================
try:
    api = API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
except:
    st.error("Token OANDA manquant dans les secrets Streamlit")
    st.stop()

# ===================== FONCTIONS (TA LOGIQUE ORIGINALE 100%) =====================
def get_candles(inst, gran):
    try:
        # Plus d'historique pour Monthly
        count = 500 if gran == "M" else 300
        r = instruments.InstrumentsCandles(instrument=inst, params={"count": count, "granularity": gran})
        api.request(r)
        candles = [c for c in r.response.get("candles", []) if c.get("complete")]
        if len(candles) < 50:
            return None
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

    last_high = None
    last_low = None
    for i in range(p, len(df)-p):
        if h[i] == max(h[i-p:i+p+1]):
            last_high = h[i]
        if l[i] == min(l[i-p:i+p+1]):
            last_low = l[i]

    if last_high and c[-1] > last_high and c[-2] <= last_high:
        strength = "Fort" if abs(c[-1]-last_high) > df["close"].diff().abs().mean()*2.5 else "Moyen"
        return "Bullish CHoCH", df.index[-1], strength
    if last_low and c[-1] < last_low and c[-2] >= last_low:
        strength = "Fort" if abs(last_low-c[-1]) > df["close"].diff().abs().mean()*2.5 else "Moyen"
        return "Bearish CHoCH", df.index[-1], strength
    return None, None, None

# ===================== PDF PROPRE & PRO =====================
def create_pdf(df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=20, rightMargin=20, topMargin=40, bottomMargin=40)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Rapport des Signaux CHoCH", styles["Title"]))
    elements.append(Paragraph(f"Généré le {datetime.utcnow().strftime('%d/%m/%Y à %H:%M')} UTC", styles["Normal"]))
    elements.append(Spacer(1, 20))

    data = [df.columns.tolist()] + df.values.tolist()
    col_widths = [80, 55, 50, 95, 70, 60, 145]

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
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ===================== UI PRO & ÉLÉGANTE =====================
st.set_page_config(page_title="CHoCH Scanner", layout="wide")
st.markdown("<h1 style='text-align:center;color:#1e40af;margin-bottom:30px;'>Scanner Change of Character (CHoCH)</h1>", unsafe_allow_html=True)

if st.button("Lancer le Scan", type="primary", use_container_width=True):
    with st.spinner("Scan en cours sur 160 timeframes... (Forex + Or + Platine + Indices + Monthly)"):
        results = []
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {
                executor.submit(get_candles, inst, code): (inst, name)
                for inst in INSTRUMENTS
                for name, code in TIMEFRAMES.items()
            }
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
                            "Force": strength or "Moyen",
                            "Heure (UTC)": time_sig.strftime("%Y-%m-%d %H:%M")
                        })

        if results:
            df_result = pd.DataFrame(results).sort_values("Heure (UTC)", ascending=False)
            st.session_state.df = df_result
            st.success(f"Scan terminé – {len(df_result)} signaux détectés sur 160 timeframes !")
        else:
            st.info("Aucun signal CHoCH récent détecté")

# ===================== AFFICHAGE FINAL PARFAIT =====================
if "df" in st.session_state:
    df = st.session_state.df.copy()
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    # Boutons d'export
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("CSV", df.to_csv(index=False).encode(), f"choch_{ts}.csv", "text/csv")
    with col2:
        fig, ax = plt.subplots(figsize=(15, max(5, len(df)*0.35)))
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1.2, 1.8)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=200)
        plt.close(fig)
        buf.seek(0)
        st.download_button("PNG", buf, f"choch_{ts}.png", "image/png")
    with col3:
        st.download_button("PDF", create_pdf(df), f"choch_signaux_{ts}.pdf", "application/pdf")

    # Tableau avec couleurs douces et pro
    st.dataframe(
        df.style.map(
            lambda x: "color:#00c853;font-weight:bold" if x == "Achat"
            else "color:#ff5252;font-weight:bold" if x == "Vente" else "",
            subset=["Ordre"]
        ).map(
            lambda x: "color:#00c853" if "Bull" in str(x) else "color:#ff5252" if "Bear" in str(x) else "",
            subset=["Signal"]
        ),
        hide_index=True,
        use_container_width=True
    )
