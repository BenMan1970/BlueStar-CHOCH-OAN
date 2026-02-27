# app.py → BLUESTAR CHoCH SCANNER v4.1 — CHoCH ONLY, PROMPT-READY
# Améliorations vs v3 :
#   ✅ ATR réel (True Range 14p) → Force Fort/Moyen/Faible fiable
#   ✅ Ratio Breakout/ATR calculé et affiché
#   ✅ ATR multi-TF (Daily, H1, M15) par instrument signalé
#   ✅ Âge du signal en heures (depuis maintenant UTC)
#   ✅ Monthly/Weekly isolés en Watchlist (rejetés par protocole v3.0)
#   ✅ Event Risk : checkbox par devise → paires bloquées marquées dans PDF
#   ✅ PDF unifié 2 sections : Actifs + Watchlist + Event Risk

import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                Paragraph, Spacer, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
INSTRUMENTS = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","USD_CAD","AUD_USD","NZD_USD",
    "EUR_GBP","EUR_JPY","EUR_CHF","EUR_AUD","EUR_CAD","EUR_NZD",
    "GBP_JPY","GBP_CHF","GBP_AUD","GBP_CAD","GBP_NZD",
    "AUD_JPY","AUD_CAD","AUD_CHF","AUD_NZD","CAD_JPY","CAD_CHF","CHF_JPY",
    "NZD_JPY","NZD_CAD","NZD_CHF",
    "XAU_USD","XPT_USD","US30_USD","NAS100_USD","SPX500_USD"
]

VOLATILITY = {
    "EUR_USD":"Basse","GBP_USD":"Basse","USD_JPY":"Basse",
    "USD_CHF":"Basse","USD_CAD":"Basse",
    "AUD_USD":"Moyenne","NZD_USD":"Moyenne","EUR_GBP":"Moyenne",
    "EUR_JPY":"Moyenne","EUR_CHF":"Moyenne","EUR_AUD":"Moyenne",
    "EUR_CAD":"Moyenne","EUR_NZD":"Moyenne",
    "GBP_JPY":"Haute","GBP_CHF":"Haute","GBP_AUD":"Haute",
    "GBP_CAD":"Haute","GBP_NZD":"Haute",
    "AUD_JPY":"Haute","AUD_CAD":"Moyenne","AUD_CHF":"Haute",
    "AUD_NZD":"Moyenne","CAD_JPY":"Haute","CAD_CHF":"Haute",
    "CHF_JPY":"Haute","NZD_JPY":"Haute","NZD_CAD":"Moyenne","NZD_CHF":"Haute",
    "XAU_USD":"Très Haute","XPT_USD":"Très Haute",
    "US30_USD":"Très Haute","NAS100_USD":"Très Haute","SPX500_USD":"Très Haute",
}

# Devises → instruments (pour event risk)
CCY_MAP = {
    "USD": ["EUR_USD","GBP_USD","USD_JPY","USD_CHF","USD_CAD","AUD_USD","NZD_USD",
            "US30_USD","NAS100_USD","SPX500_USD","XAU_USD","XPT_USD"],
    "EUR": ["EUR_USD","EUR_GBP","EUR_JPY","EUR_CHF","EUR_AUD","EUR_CAD","EUR_NZD"],
    "GBP": ["GBP_USD","EUR_GBP","GBP_JPY","GBP_CHF","GBP_AUD","GBP_CAD","GBP_NZD"],
    "JPY": ["USD_JPY","EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","CHF_JPY","NZD_JPY"],
    "CAD": ["USD_CAD","EUR_CAD","GBP_CAD","AUD_CAD","CAD_JPY","CAD_CHF","NZD_CAD"],
    "AUD": ["AUD_USD","EUR_AUD","GBP_AUD","AUD_JPY","AUD_CAD","AUD_CHF","AUD_NZD"],
    "NZD": ["NZD_USD","EUR_NZD","GBP_NZD","AUD_NZD","NZD_JPY","NZD_CAD","NZD_CHF"],
    "CHF": ["USD_CHF","EUR_CHF","GBP_CHF","AUD_CHF","CAD_CHF","CHF_JPY","NZD_CHF"],
}

# TF protocole : actifs (scorés) vs watchlist (contexte macro seulement)
ACTIVE_TF    = {"H1": "H1", "H4": "H4", "Daily": "D"}
WATCHLIST_TF = {"Weekly": "W", "Monthly": "M"}
FRACTAL_LEN  = {"H1": 5, "H4": 6, "Daily": 7, "Weekly": 8, "Monthly": 9}

# TF ATR à récupérer pour chaque instrument signalé
ATR_FETCH = {"Daily": "D", "H1": "H1", "M15": "M15"}

# ═══════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════
try:
    api = API(access_token=st.secrets["OANDA_ACCESS_TOKEN"])
except Exception as e:
    st.error(f"Token OANDA manquant dans les secrets Streamlit : {e}")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# FONCTIONS TECHNIQUES
# ═══════════════════════════════════════════════════════════════

def get_candles(inst, gran, count=300):
    """Récupère les bougies OANDA → DataFrame OHLC."""
    try:
        cnt = 500 if gran in ("M", "W") else count
        r = instruments.InstrumentsCandles(
            instrument=inst,
            params={"count": cnt, "granularity": gran}
        )
        api.request(r)
        candles = [c for c in r.response.get("candles", []) if c.get("complete")]
        if len(candles) < 50:
            return None
        df = pd.DataFrame([{
            "time":  pd.to_datetime(c["time"]),
            "open":  float(c["mid"]["o"]),
            "high":  float(c["mid"]["h"]),
            "low":   float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
        } for c in candles])
        df.set_index("time", inplace=True)
        return df
    except Exception:
        return None


def calc_atr(df, period=14):
    """ATR réel (True Range) 14 périodes — valeur de la dernière bougie."""
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:]  - close[:-1])
        )
    )
    if len(tr) < period:
        return np.nan
    return round(float(np.mean(tr[-period:])), 5)


def calc_force(atr, breakout_size, tf):
    """
    Force basée sur le ratio Breakout/ATR réel (seuils différenciés par TF).
    H1  : >1.2×ATR → Fort | 0.6–1.2× → Moyen | <0.6× → Faible
    H4  : >1.5×ATR → Fort | 0.8–1.5× → Moyen | <0.8× → Faible
    D1  : >1.5×ATR → Fort | 0.8–1.5× → Moyen | <0.8× → Faible
    W/M : >2.0×ATR → Fort | 1.0–2.0× → Moyen | <1.0× → Faible
    """
    if np.isnan(atr) or atr == 0:
        return "Moyen", "N/A"
    ratio = breakout_size / atr
    ratio_str = f"{ratio:.2f}×"
    if tf == "H1":
        force = "Fort" if ratio > 1.2 else ("Moyen" if ratio > 0.6 else "Faible")
    elif tf in ("Weekly", "Monthly"):
        force = "Fort" if ratio > 2.0 else ("Moyen" if ratio > 1.0 else "Faible")
    else:  # H4, Daily
        force = "Fort" if ratio > 1.5 else ("Moyen" if ratio > 0.8 else "Faible")
    return force, ratio_str


def get_atr_multitf(inst):
    """ATR Daily, H1, M15 pour un instrument — appelé uniquement pour les signaux actifs."""
    result = {"ATR Daily": "N/A", "ATR H1": "N/A", "ATR M15": "N/A"}
    mapping = {"ATR Daily": "D", "ATR H1": "H1", "ATR M15": "M15"}
    for key, gran in mapping.items():
        df = get_candles(inst, gran, count=50)
        if df is not None:
            val = calc_atr(df, 14)
            result[key] = val if not np.isnan(val) else "N/A"
    return result


def age_hours(signal_time):
    """Âge du signal en heures depuis maintenant (UTC)."""
    try:
        now = datetime.now(timezone.utc)
        if signal_time.tzinfo is None:
            signal_time = signal_time.replace(tzinfo=timezone.utc)
        delta = now - signal_time
        return round(delta.total_seconds() / 3600, 1)
    except Exception:
        return "N/A"


def detect_choch(df, tf):
    """
    Détection CHoCH par fractal adaptatif (longueur selon TF).
    Retourne : (signal_type, timestamp, force, ratio_str, atr_val)
    """
    length = FRACTAL_LEN.get(tf, 5)
    p = length // 2
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    last_high = None
    last_low  = None
    for i in range(p, len(df) - p):
        if h[i] == max(h[i-p:i+p+1]):
            last_high = h[i]
        if l[i] == min(l[i-p:i+p+1]):
            last_low = l[i]

    atr_val = calc_atr(df, 14)

    if last_high and c[-1] > last_high and c[-2] <= last_high:
        breakout      = abs(c[-1] - last_high)
        force, ratio  = calc_force(atr_val, breakout, tf)
        return "Bullish CHoCH", df.index[-1], force, ratio, atr_val

    if last_low and c[-1] < last_low and c[-2] >= last_low:
        breakout      = abs(last_low - c[-1])
        force, ratio  = calc_force(atr_val, breakout, tf)
        return "Bearish CHoCH", df.index[-1], force, ratio, atr_val

    return None, None, None, None, None


# ═══════════════════════════════════════════════════════════════
# PDF PROMPT-READY
# ═══════════════════════════════════════════════════════════════

def build_pdf(df_active, df_watch, blocked_pairs, scan_time):
    """
    PDF structuré en 2 sections :
      Section 1 — Signaux actifs (H1/H4/Daily) avec ATR Daily/H1/M15
      Section 2 — Watchlist (Weekly/Monthly — contexte macro uniquement)
    + bandeau Event Risk
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=12*mm, rightMargin=12*mm,
        topMargin=10*mm, bottomMargin=10*mm,
    )
    styles   = getSampleStyleSheet()
    elements = []

    BLUE   = colors.HexColor("#1e40af")
    DBLUE  = colors.HexColor("#0f2060")
    GREEN  = colors.HexColor("#00a854")
    RED    = colors.HexColor("#e02020")
    ORANGE = colors.HexColor("#d06000")
    LGRAY  = colors.HexColor("#f0f4f8")
    MGRAY  = colors.HexColor("#c8d4e0")
    WHITE  = colors.white

    T = lambda txt, style: Paragraph(txt, style)
    title_s = ParagraphStyle("t", fontSize=15, textColor=DBLUE,
                             fontName="Helvetica-Bold", spaceAfter=3, alignment=TA_CENTER)
    sub_s   = ParagraphStyle("s", fontSize=8.5, textColor=colors.gray,
                             fontName="Helvetica", spaceAfter=2, alignment=TA_CENTER)
    sec_s   = ParagraphStyle("sec", fontSize=11, textColor=DBLUE,
                             fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=4)
    note_s  = ParagraphStyle("n", fontSize=7.5, textColor=colors.gray,
                             fontName="Helvetica-Oblique")
    warn_s  = ParagraphStyle("w", fontSize=8.5, textColor=RED,
                             fontName="Helvetica-Bold")
    disc_s  = ParagraphStyle("d", fontSize=6.5, textColor=colors.gray,
                             fontName="Helvetica-Oblique")

    # ── En-tête ──────────────────────────────────────────────────────────────
    elements.append(T("BLUESTAR CHoCH SCANNER — Rapport Prompt-Ready", title_s))
    elements.append(T(
        f"Généré le {scan_time.strftime('%d/%m/%Y à %H:%M')} UTC  |  "
        f"Signaux actifs : {len(df_active)}  |  "
        f"Watchlist W/M : {len(df_watch)}  |  "
        f"Paires bloquées : {len(blocked_pairs)}",
        sub_s
    ))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=5))

    # ── Event Risk banner ─────────────────────────────────────────────────────
    if blocked_pairs:
        elements.append(T(
            f"⚠  EVENT RISK — VETO ABSOLU  |  Paires bloquées : {' | '.join(sorted(blocked_pairs))}",
            warn_s
        ))
        elements.append(Spacer(1, 4))

    # ── Section 1 : Signaux Actifs ────────────────────────────────────────────
    elements.append(T("SECTION 1 — Signaux CHoCH Actifs (H1 / H4 / Daily)", sec_s))
    elements.append(T(
        "Force = ratio Breakout/ATR réel (14p). "
        "Seuils : H1 >1.2×=Fort, >0.6×=Moyen  |  H4/Daily >1.5×=Fort, >0.8×=Moyen. "
        "ATR Daily/H1/M15 = données brutes pour le scoring S/R et SL.",
        note_s
    ))
    elements.append(Spacer(1, 3))

    COLS_ACTIVE = [
        "Instrument", "TF", "Ordre", "Signal",
        "Volatilité", "Force", "Breakout/ATR",
        "ATR Daily", "ATR H1", "ATR M15",
        "Heure (UTC)", "Âge (h)",
    ]
    CW_ACTIVE = [24, 13, 13, 28, 16, 13, 18, 18, 15, 15, 32, 13]  # mm

    if not df_active.empty:
        for col in COLS_ACTIVE:
            if col not in df_active.columns:
                df_active[col] = "N/A"
        data = [COLS_ACTIVE] + df_active[COLS_ACTIVE].astype(str).values.tolist()

        tbl = Table(data, colWidths=[c*mm for c in CW_ACTIVE], repeatRows=1)
        ts_style = TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), BLUE),
            ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,0), 8),
            ("FONTSIZE",      (0,1), (-1,-1), 7.5),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("GRID",          (0,0), (-1,-1), 0.4, MGRAY),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LGRAY]),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ])
        blocked_raw = {p.replace("/", "_") for p in blocked_pairs}
        for i, row in enumerate(df_active.itertuples(), start=1):
            inst_raw = row.Instrument.replace("/", "_")
            # Surlignage paires bloquées
            if inst_raw in blocked_raw:
                ts_style.add("BACKGROUND", (0,i), (-1,i), colors.HexColor("#fff0f0"))
                ts_style.add("TEXTCOLOR",  (0,i), (0,i),  RED)
            # Couleur Ordre
            ordre_col = GREEN if row.Ordre == "Achat" else RED
            ts_style.add("TEXTCOLOR", (2,i), (2,i), ordre_col)
            ts_style.add("FONTNAME",  (2,i), (2,i), "Helvetica-Bold")
            # Couleur Force
            force_col = (GREEN if row.Force == "Fort"
                         else RED if row.Force == "Faible"
                         else colors.HexColor("#e07000"))
            ts_style.add("TEXTCOLOR", (5,i), (5,i), force_col)
            ts_style.add("FONTNAME",  (5,i), (5,i), "Helvetica-Bold")
        tbl.setStyle(ts_style)
        elements.append(tbl)
    else:
        elements.append(T("Aucun signal CHoCH actif détecté (H1/H4/Daily).", note_s))

    elements.append(Spacer(1, 8))

    # ── Section 2 : Watchlist Weekly / Monthly ────────────────────────────────
    elements.append(HRFlowable(width="100%", thickness=0.7, color=MGRAY, spaceAfter=4))
    elements.append(T(
        "SECTION 2 — Watchlist (Weekly / Monthly — Contexte macro, EXCLUS du protocole scoring)",
        sec_s
    ))
    elements.append(T(
        "Ces signaux sont informationnels. Weekly et Monthly sont rejetés à l'Étape 1 du protocole CHoCH v3.0. "
        "Ils donnent la tendance structurelle de fond uniquement.",
        note_s
    ))
    elements.append(Spacer(1, 3))

    COLS_WATCH = [
        "Instrument", "TF", "Ordre", "Signal",
        "Volatilité", "Force", "Breakout/ATR",
        "ATR Daily", "Heure (UTC)",
    ]
    CW_WATCH = [28, 16, 15, 30, 18, 14, 18, 20, 35]

    if not df_watch.empty:
        for col in COLS_WATCH:
            if col not in df_watch.columns:
                df_watch[col] = "N/A"
        data2 = [COLS_WATCH] + df_watch[COLS_WATCH].astype(str).values.tolist()
        tbl2 = Table(data2, colWidths=[c*mm for c in CW_WATCH], repeatRows=1)
        ts2 = TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), ORANGE),
            ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,0), 8),
            ("FONTSIZE",      (0,1), (-1,-1), 7.5),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("GRID",          (0,0), (-1,-1), 0.4, MGRAY),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, LGRAY]),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ])
        for i, row in enumerate(df_watch.itertuples(), start=1):
            c = GREEN if row.Ordre == "Achat" else RED
            ts2.add("TEXTCOLOR", (2,i), (2,i), c)
            ts2.add("FONTNAME",  (2,i), (2,i), "Helvetica-Bold")
        tbl2.setStyle(ts2)
        elements.append(tbl2)
    else:
        elements.append(T("Aucun signal Weekly/Monthly détecté.", note_s))

    # ── Footer ────────────────────────────────────────────────────────────────
    elements.append(Spacer(1, 6))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=MGRAY, spaceAfter=3))
    elements.append(T(
        "DISCLAIMER : Ce rapport est produit automatiquement à des fins d'assistance à l'analyse technique. "
        "Il ne constitue pas un conseil financier. Appliquez une gestion des risques rigoureuse. "
        "— Bluestar CHoCH Scanner v4.1",
        disc_s
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# ═══════════════════════════════════════════════════════════════
# UI STREAMLIT
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Bluestar CHoCH Scanner",
    layout="wide",
    page_icon="⬡"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&display=swap');
  body, .stApp { background:#0a0c0f; color:#c8d4e8; }
  .stButton>button {
    background:#1e40af; color:white; border:none; border-radius:4px;
    font-family:'IBM Plex Mono',monospace; font-weight:600;
    font-size:14px; padding:12px 20px; width:100%; transition:all .2s;
  }
  .stButton>button:hover { background:#2563eb; transform:translateY(-1px); }
  .kpi-card {
    background:#141820; border:1px solid #1e2535; border-radius:6px;
    padding:16px; text-align:center;
  }
  .kpi-label { font-size:10px; text-transform:uppercase; letter-spacing:1.5px; color:#6b7fa0; margin-bottom:6px; }
  .kpi-val   { font-family:'IBM Plex Mono',monospace; font-size:30px; font-weight:600; line-height:1; }
  .green  { color:#00c87a; }
  .red    { color:#ff3d57; }
  .yellow { color:#f5c842; }
  .blue   { color:#3d8eff; }
  .orange { color:#ff7c35; }
  .event-bar {
    background:rgba(255,61,87,.1); border:1px solid rgba(255,61,87,.4);
    border-left:3px solid #ff3d57; border-radius:3px;
    padding:10px 14px; font-size:12px; color:#ff3d57; margin:8px 0;
  }
  div[data-testid="stDataFrame"] { border:1px solid #1e2535 !important; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:20px 0 8px;'>
  <div style='font-family:"IBM Plex Mono",monospace;font-size:28px;font-weight:600;color:#3d8eff;letter-spacing:-1px;'>⬡ BLUESTAR</div>
  <div style='font-family:"IBM Plex Mono",monospace;font-size:12px;color:#6b7fa0;margin-top:4px;letter-spacing:2.5px;'>CHoCH SCANNER v4.1 — PROMPT-READY</div>
</div>
""", unsafe_allow_html=True)

# ── Event Risk ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**⚠️ Event Risk** — Cocher les devises avec news High Impact dans les **4 prochaines heures**")
st.caption("Les paires correspondantes seront surlignées comme bloquées dans le PDF (Étape 0 du protocole).")

ev_cols = st.columns(8)
blocked_ccys = []
for i, ccy in enumerate(["USD","EUR","GBP","JPY","CAD","AUD","NZD","CHF"]):
    with ev_cols[i]:
        if st.checkbox(ccy, key=f"ev_{ccy}"):
            blocked_ccys.append(ccy)

blocked_pairs = set()
for ccy in blocked_ccys:
    for inst in CCY_MAP.get(ccy, []):
        blocked_pairs.add(inst.replace("_", "/"))

if blocked_pairs:
    pairs_str = " | ".join(sorted(blocked_pairs))
    st.markdown(f"<div class='event-bar'>🚨 <strong>Paires bloquées :</strong> {pairs_str}</div>",
                unsafe_allow_html=True)

# ── Scan Button ───────────────────────────────────────────────────────────────
st.markdown("---")
run = st.button("🔍  Lancer le Scan CHoCH", type="primary", use_container_width=True)

if run:
    scan_time = datetime.now(timezone.utc)
    progress  = st.progress(0, text="Initialisation...")

    results_active = []
    results_watch  = []

    # ── 1. Scan CHoCH sur tous les TF ────────────────────────────────────────
    all_tasks = (
        [(inst, tf_name, "active") for inst in INSTRUMENTS for tf_name in ACTIVE_TF]
      + [(inst, tf_name, "watch")  for inst in INSTRUMENTS for tf_name in WATCHLIST_TF]
    )
    total = len(all_tasks)
    done  = 0

    with ThreadPoolExecutor(max_workers=14) as executor:
        future_map = {}
        for inst, tf_name, category in all_tasks:
            gran   = ACTIVE_TF.get(tf_name) or WATCHLIST_TF.get(tf_name)
            future = executor.submit(get_candles, inst, gran)
            future_map[future] = (inst, tf_name, category)

        for future in as_completed(future_map):
            inst, tf_name, category = future_map[future]
            df_c = future.result()
            done += 1
            progress.progress(
                done / total * 0.65,
                text=f"Scan CHoCH — {inst.replace('_','/')} {tf_name}  ({done}/{total})"
            )
            if df_c is None:
                continue

            sig, sig_time, force, ratio_str, atr_tf = detect_choch(df_c, tf_name)
            if sig is None:
                continue

            record = {
                "Instrument":    inst.replace("_", "/"),
                "TF":            tf_name,
                "Ordre":         "Achat" if "Bull" in sig else "Vente",
                "Signal":        sig,
                "Volatilité":    VOLATILITY.get(inst, "Moyenne"),
                "Force":         force,
                "Breakout/ATR":  ratio_str,
                "ATR Daily":     "N/A",
                "ATR H1":        "N/A",
                "ATR M15":       "N/A",
                "Heure (UTC)":   sig_time.strftime("%Y-%m-%d %H:%M"),
                "Âge (h)":       age_hours(sig_time),
                "_inst_raw":     inst,
            }
            if category == "active":
                results_active.append(record)
            else:
                results_watch.append(record)

    # ── 2. Enrichissement ATR multi-TF (actifs uniquement) ───────────────────
    active_insts = list({r["_inst_raw"] for r in results_active})
    total_enrich = len(active_insts)

    if total_enrich > 0:
        atr_cache = {}
        with ThreadPoolExecutor(max_workers=12) as executor:
            fut_atr = {executor.submit(get_atr_multitf, inst): inst
                       for inst in active_insts}
            enriched = 0
            for future in as_completed(fut_atr):
                enriched += 1
                progress.progress(
                    0.65 + enriched / total_enrich * 0.30,
                    text=f"Récupération ATR Daily/H1/M15 ({enriched}/{total_enrich})..."
                )
                atr_cache[fut_atr[future]] = future.result()

        for rec in results_active:
            atr_data = atr_cache.get(rec["_inst_raw"], {})
            rec["ATR Daily"] = atr_data.get("ATR Daily", "N/A")
            rec["ATR H1"]    = atr_data.get("ATR H1",    "N/A")
            rec["ATR M15"]   = atr_data.get("ATR M15",   "N/A")

    # Watchlist : récupérer ATR Daily seulement (pour contexte)
    watch_insts = list({r["_inst_raw"] for r in results_watch})
    if watch_insts:
        with ThreadPoolExecutor(max_workers=8) as executor:
            fut_w = {executor.submit(get_candles, inst, "D", 50): inst
                     for inst in watch_insts}
            for future in as_completed(fut_w):
                inst_raw = fut_w[future]
                df_w = future.result()
                if df_w is not None:
                    atr_d = calc_atr(df_w, 14)
                    for rec in results_watch:
                        if rec["_inst_raw"] == inst_raw:
                            rec["ATR Daily"] = atr_d if not np.isnan(atr_d) else "N/A"

    progress.progress(1.0, text="✓ Scan terminé")

    # ── Build DataFrames ──────────────────────────────────────────────────────
    def make_df(records):
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df.drop(columns=["_inst_raw"], inplace=True, errors="ignore")
        return df.sort_values("Heure (UTC)", ascending=False).reset_index(drop=True)

    st.session_state.df_active  = make_df(results_active)
    st.session_state.df_watch   = make_df(results_watch)
    st.session_state.blocked    = list(blocked_pairs)
    st.session_state.scan_time  = scan_time
    progress.empty()


# ── Affichage Résultats ───────────────────────────────────────────────────────
if "df_active" in st.session_state:
    df_active = st.session_state.df_active
    df_watch  = st.session_state.df_watch
    blocked   = st.session_state.blocked
    scan_time = st.session_state.scan_time

    # KPIs
    n_a     = len(df_active)
    n_bull  = len(df_active[df_active["Ordre"] == "Achat"])  if not df_active.empty else 0
    n_bear  = len(df_active[df_active["Ordre"] == "Vente"])  if not df_active.empty else 0
    n_fort  = len(df_active[df_active["Force"] == "Fort"])   if not df_active.empty else 0
    n_block = len(blocked)

    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        ("Signaux Actifs",  n_a,     "blue"),
        ("Bullish ▲",        n_bull,  "green"),
        ("Bearish ▼",        n_bear,  "red"),
        ("Fort 💪",           n_fort,  "yellow"),
        ("Bloquées ⚠",       n_block, "orange" if n_block else "blue"),
    ]
    for col, (lbl, val, cls) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(
                f"<div class='kpi-card'><div class='kpi-label'>{lbl}</div>"
                f"<div class='kpi-val {cls}'>{val}</div></div>",
                unsafe_allow_html=True
            )

    st.markdown("")

    # ── Tableau Actifs ────────────────────────────────────────────────────────
    if not df_active.empty:
        st.markdown("#### 📊 Signaux CHoCH Actifs — H1 / H4 / Daily")

        DISPLAY = [
            "Instrument", "TF", "Ordre", "Signal",
            "Volatilité", "Force", "Breakout/ATR",
            "ATR Daily", "ATR H1", "ATR M15",
            "Heure (UTC)", "Âge (h)",
        ]
        display_df = df_active[[c for c in DISPLAY if c in df_active.columns]]

        st.dataframe(
            display_df.style
            .map(lambda x: "color:#00c87a;font-weight:bold" if x == "Achat"
                 else "color:#ff3d57;font-weight:bold" if x == "Vente" else "",
                 subset=["Ordre"] if "Ordre" in display_df.columns else [])
            .map(lambda x: "color:#00c87a;font-weight:bold" if x == "Fort"
                 else "color:#ff3d57;font-weight:bold" if x == "Faible"
                 else "color:#ff7c35;font-weight:bold" if x == "Moyen" else "",
                 subset=["Force"] if "Force" in display_df.columns else []),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("Aucun signal CHoCH actif (H1/H4/Daily) détecté lors de ce scan.")

    # ── Watchlist ─────────────────────────────────────────────────────────────
    if not df_watch.empty:
        with st.expander(
            f"📋 Watchlist Weekly/Monthly — {len(df_watch)} signaux "
            f"(contexte macro · exclus du protocole scoring)"
        ):
            watch_disp = [c for c in
                ["Instrument","TF","Ordre","Signal","Volatilité","Force",
                 "Breakout/ATR","ATR Daily","Heure (UTC)"]
                if c in df_watch.columns]
            st.dataframe(df_watch[watch_disp], hide_index=True, use_container_width=True)

    # ── Exports ───────────────────────────────────────────────────────────────
    st.markdown("---")
    ts = scan_time.strftime("%Y%m%d_%H%M")

    ec1, ec2, ec3 = st.columns(3)

    with ec1:
        full_csv = pd.concat([df_active, df_watch], ignore_index=True)
        st.download_button(
            "📄 CSV Complet",
            full_csv.to_csv(index=False).encode("utf-8"),
            f"choch_complet_{ts}.csv",
            "text/csv",
            use_container_width=True,
        )

    with ec2:
        if not df_active.empty:
            st.download_button(
                "📄 CSV Actifs (H1/H4/Daily)",
                df_active.to_csv(index=False).encode("utf-8"),
                f"choch_actifs_{ts}.csv",
                "text/csv",
                use_container_width=True,
            )
        else:
            st.button("📄 CSV Actifs (vide)", disabled=True, use_container_width=True)

    with ec3:
        pdf_buf = build_pdf(
            df_active.copy() if not df_active.empty else pd.DataFrame(),
            df_watch.copy()  if not df_watch.empty  else pd.DataFrame(),
            blocked,
            scan_time,
        )
        st.download_button(
            "📑 PDF Prompt-Ready",
            pdf_buf,
            f"choch_signaux_{ts}.pdf",
            "application/pdf",
            type="primary",
            use_container_width=True,
        )

    st.markdown(
        f"<div style='font-size:10px;color:#3a4a65;text-align:right;margin-top:4px;'>"
        f"Scan : {scan_time.strftime('%d/%m/%Y %H:%M')} UTC  |  Bluestar CHoCH Scanner v4.1"
        f"</div>",
        unsafe_allow_html=True,
    )
