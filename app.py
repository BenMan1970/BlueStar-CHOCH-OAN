# ============================================================
# SCANNER CHOCH OANDA — v5.5 PATCHED
# Corrections appliquées (voir commentaires # PATCH:) :
#   P1  - Thread-safe API via threading.local()
#   P2  - try/finally sur le bouton (deadlock UI)
#   P3  - Weekend Bug : statut calculé en bougies, pas en heures
#   P4  - Memory leak Matplotlib : API OOP Figure()
#   P5  - Logique fractale : dernier swing chronologique (SMC correct)
#   P6  - BOS/CHoCH : convention SMC standard corrigée
#   P7  - DRY : True Range extrait en utilitaire
#   P8  - Gestion d'erreur API non silencieuse + logging
#   P9  - Division par zéro robuste (np.isclose)
#   P10 - Tri déterministe sur datetime brut avant drop_duplicates
#   P11 - normalize_pair() supprimé (dead code)
#   P12 - Count API adaptatif par timeframe
#   P13 - Filtre Range étendu (Range + Moyen aussi filtré)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import logging
import threading

import matplotlib
matplotlib.use('Agg')  # PATCH P4: backend non-interactif, évite memory leak
from matplotlib.figure import Figure  # PATCH P4: API OOP stricte

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.exceptions import V20Error

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ===================== LOGGING =====================
# PATCH P8: logging structuré au lieu d'exceptions silencieuses
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ===================== CONFIG =====================
INSTRUMENTS = [
    # 28 paires Forex
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    # 5 indices et métaux
    "DE30_EUR", "XAU_USD", "SPX500_USD", "NAS100_USD", "US30_USD",
]

VOLATILITY_STATIC = {
    "EUR_USD": "Basse",   "GBP_USD": "Basse",   "USD_JPY": "Basse",
    "USD_CHF": "Basse",   "USD_CAD": "Basse",
    "AUD_USD": "Moyenne", "NZD_USD": "Moyenne",  "EUR_GBP": "Moyenne",
    "EUR_JPY": "Moyenne", "EUR_CHF": "Moyenne",  "EUR_AUD": "Moyenne",
    "EUR_CAD": "Moyenne", "EUR_NZD": "Moyenne",
    "GBP_JPY": "Haute",   "GBP_CHF": "Haute",    "GBP_AUD": "Haute",
    "GBP_CAD": "Haute",   "GBP_NZD": "Haute",
    "AUD_JPY": "Haute",   "AUD_CAD": "Moyenne",  "AUD_CHF": "Haute",
    "AUD_NZD": "Moyenne", "CAD_JPY": "Haute",    "CAD_CHF": "Haute",
    "CHF_JPY": "Haute",   "NZD_JPY": "Haute",
    "NZD_CAD": "Moyenne", "NZD_CHF": "Haute",
    "DE30_EUR":   "Très Haute", "XAU_USD":    "Très Haute",
    "SPX500_USD": "Très Haute", "NAS100_USD": "Très Haute",
    "US30_USD":   "Très Haute",
}

TIMEFRAMES     = {"H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"}
FRACTAL_LEN    = {"H1": 5, "H4": 5, "D1": 7, "Weekly": 7}
FRACTAL_WINDOW = {"H1": 120, "H4": 90, "D1": 60, "Weekly": 26}

TF_STATUT = {
    "H1":     {"Fresh": 4,  "Aged": 12},
    "H4":     {"Fresh": 3,  "Aged": 8},
    "D1":     {"Fresh": 2,  "Aged": 5},
    "Weekly": {"Fresh": 2,  "Aged": 4},
}

TF_LOOKBACK     = {"H1": 5, "H4": 5, "D1": 4, "Weekly": 3}
TF_EMA_LOOKBACK = {"H1": 5, "H4": 5, "D1": 10, "Weekly": 15}

# PATCH P12: nombre de bougies adaptatif — inutile de demander 500 bougies Weekly
TF_COUNT = {"H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"}
GRAN_COUNT = {"H1": 400, "H4": 300, "D": 200, "W": 120}

DISPLAY_COLS = [
    "Instrument", "Timeframe", "Type", "Ordre", "Signal",
    "Niveau", "Distance%", "Volatilité", "Force", "BB_Width", "Statut", "Heure (UTC)"
]
EXPORT_COLS = [
    "Paire", "Timeframe", "Type", "Ordre", "Signal",
    "Niveau", "Distance%", "Volatilité", "Force", "BB_Width", "Statut", "Heure (UTC)"
]

# ===================== API THREAD-SAFE =====================
# PATCH P1: threading.local() — chaque thread possède sa propre instance API
# Évite la corruption de requêtes concurrentes sur l'objet requests.Session partagé
_thread_local = threading.local()

def _get_api() -> API:
    """Retourne une instance API isolée par thread (thread-local singleton)."""
    if not hasattr(_thread_local, "api"):
        try:
            _thread_local.api = API(
                access_token=st.secrets["OANDA_ACCESS_TOKEN"],
                request_params={"timeout": 12}
            )
        except Exception as e:
            logger.critical(f"Impossible d'initialiser l'API OANDA : {e}")
            raise
    return _thread_local.api

# Validation du secret au démarrage (fail-fast, avant le scan)
try:
    _ = st.secrets["OANDA_ACCESS_TOKEN"]
except Exception as e:
    st.error(f"Token OANDA manquant dans les secrets Streamlit : {e}")
    st.stop()


# ===================== FONCTIONS =====================

def get_candles(inst: str, gran: str) -> pd.DataFrame | None:
    """
    Récupère les bougies OANDA pour un instrument et une granularité.
    Thread-safe via _get_api(). Erreurs loguées, jamais silencieuses.
    """
    count = GRAN_COUNT.get(gran, 300)  # PATCH P12
    try:
        r = instruments.InstrumentsCandles(
            instrument=inst,
            params={"count": count, "granularity": gran}
        )
        _get_api().request(r)  # PATCH P1
        candles = [c for c in r.response.get("candles", []) if c.get("complete")]
        if len(candles) < 50:
            return None
        df = pd.DataFrame([{
            "time":  pd.to_datetime(c["time"], utc=True),
            "open":  float(c["mid"]["o"]),
            "high":  float(c["mid"]["h"]),
            "low":   float(c["mid"]["l"]),
            "close": float(c["mid"]["c"])
        } for c in candles])
        df.set_index("time", inplace=True)
        return df

    except V20Error as e:
        # PATCH P8: distingue les erreurs d'auth (critiques) des erreurs passagères
        if e.code in (401, 403):
            logger.error(f"Authentification OANDA échouée [{inst}/{gran}]: {e}")
        elif e.code == 429:
            logger.warning(f"Rate limit OANDA [{inst}/{gran}] — signal ignoré ce tick")
        else:
            logger.warning(f"V20Error [{inst}/{gran}] code={e.code}: {e}")
        return None
    except Exception as e:
        logger.error(f"Erreur inattendue get_candles [{inst}/{gran}]: {type(e).__name__}: {e}")
        return None


# PATCH P7: True Range extrait en utilitaire — calcul DRY (une seule source de vérité)
def _compute_true_range(df: pd.DataFrame) -> np.ndarray:
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    return np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1]))
    )


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    tr = _compute_true_range(df)
    if len(tr) < period * 3:  # warmup EWM insuffisant → NaN plutôt que valeur biaisée
        return np.nan
    return float(pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


def atr_to_volatility(atr_val: float, inst: str, df: pd.DataFrame) -> str:
    if np.isnan(atr_val) or len(df) < 28:
        return VOLATILITY_STATIC.get(inst, "Moyenne")
    tr = _compute_true_range(df)  # PATCH P7: réutilise l'utilitaire
    window = tr[-100:] if len(tr) >= 100 else tr
    median_tr = float(np.median(window))
    if np.isclose(median_tr, 0, atol=1e-10):  # PATCH P9
        return VOLATILITY_STATIC.get(inst, "Moyenne")
    ratio = atr_val / median_tr
    if ratio >= 1.8: return "Très Haute"
    if ratio >= 1.2: return "Haute"
    if ratio >= 0.7: return "Moyenne"
    return "Basse"


def compute_bb_width(df: pd.DataFrame, length: int = 20, std: int = 2) -> str:
    close = df["close"]
    if len(close) < length * 2:
        return "N/A"
    sma     = close.rolling(length).mean()
    std_dev = close.rolling(length).std()
    upper   = sma + std * std_dev
    lower   = sma - std * std_dev
    bb_w    = (upper - lower) / sma
    bb_avg  = bb_w.rolling(length).mean()
    avg_last = bb_avg.iloc[-1]

    # PATCH P9: np.isclose au lieu de == 0
    if pd.isna(avg_last) or np.isclose(avg_last, 0, atol=1e-10):
        return "N/A"

    bb_avg_safe = bb_avg.replace(0, np.nan)  # PATCH P9: évite div/0 dans le calcul pct
    pct = ((bb_w - bb_avg) / bb_avg_safe * 100).iloc[-1]
    if pd.isna(pct):
        return "N/A"

    sign = "+" if pct >= 0 else ""
    if pct < -25: return f"{sign}{pct:.0f}%_Squeeze"
    if pct > 25:  return f"{sign}{pct:.0f}%_Expansion"
    return f"{sign}{pct:.0f}%_Normal"


def compute_statut(idx_sig: int | None, len_df: int, tf: str) -> str:
    """
    PATCH P3: Weekend Bug corrigé.
    Calcule le statut en comptant le nombre de bougies écoulées depuis le signal
    (index dans le DataFrame) et non en heures réelles. Ainsi un signal de vendredi
    soir reste "Fresh" lundi matin s'il n'y a eu que 0 bougie depuis.
    """
    if idx_sig is None:
        return "N/A"
    candles_elapsed = (len_df - 1) - idx_sig
    thresholds = TF_STATUT.get(tf, {"Fresh": 2, "Aged": 5})
    if candles_elapsed <= thresholds["Fresh"]: return "Fresh"
    if candles_elapsed <= thresholds["Aged"]:  return "Aged"
    return "Stale"


def get_trend_context(df: pd.DataFrame, tf: str) -> str:
    close = df["close"]
    if len(close) < 55:
        return "Unknown"
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    e20_last = ema20.iloc[-1]
    e50_last = ema50.iloc[-1]
    lookback = TF_EMA_LOOKBACK.get(tf, 5)
    e20_prev = ema20.iloc[-(lookback + 1)] if len(ema20) > lookback else ema20.iloc[0]
    if e20_last > e50_last and e20_last > e20_prev: return "Uptrend"
    if e20_last < e50_last and e20_last < e20_prev: return "Downtrend"
    return "Range"


def detect_choch(df: pd.DataFrame, tf: str):
    """
    Détecte le dernier CHoCH ou BOS valide sur le DataFrame fourni.

    PATCH P5: La boucle fractale collecte maintenant TOUTES les fractales valides
    de la fenêtre et retient la DERNIÈRE chronologiquement (SMC : niveau de
    structure le plus récent, pas le max absolu).

    PATCH P6: La convention BOS/CHoCH est conforme au standard SMC :
      - CHoCH = cassure CONTRE la tendance (signal de retournement)
      - BOS   = cassure DANS le sens de la tendance (continuation)
    L'étiquetage est désormais calculé en dehors de detect_choch() côté appelant.
    """
    length   = FRACTAL_LEN.get(tf, 5)
    p        = length // 2
    lookback = TF_LOOKBACK.get(tf, 3)
    window   = FRACTAL_WINDOW.get(tf, 60)

    h_all = df["high"].values
    l_all = df["low"].values
    c_all = df["close"].values
    o_all = df["open"].values

    min_len = p * 2 + 1 + lookback + 1
    if len(c_all) < min_len:
        return None, None, None, None, None, None, None

    # Percentiles sur les 50 dernières bougies — volatilité récente, pas historique
    recent_ranges = h_all[-50:] - l_all[-50:]
    p25 = float(np.percentile(recent_ranges, 25)) if len(recent_ranges) >= 10 else None
    p75 = float(np.percentile(recent_ranges, 75)) if len(recent_ranges) >= 10 else None

    def get_force(rng: float) -> str:
        if p25 is None or p75 is None: return "Moyen"
        if rng >= p75: return "Fort"
        if rng <= p25: return "Faible"
        return "Moyen"

    def is_valid_breakout_candle(idx: int) -> bool:
        rng = h_all[idx] - l_all[idx]
        if np.isclose(rng, 0, atol=1e-10):  # PATCH P9
            return False
        body = abs(c_all[idx] - o_all[idx])
        return (body / rng) >= 0.3

    for offset in range(lookback):
        idx_cur  = len(df) - 1 - offset
        idx_prev = idx_cur - 1
        if idx_prev < p:
            break

        # Slices bornés à idx_cur + 1 — pas de lookahead sur bougies futures
        h = h_all[:idx_cur + 1]
        l = l_all[:idx_cur + 1]
        c = c_all[:idx_cur + 1]

        # PATCH P5: borne explicite pour éliminer le biais de lookahead dans les fractales
        # La recherche s'arrête à idx_cur - p (la fractal doit être confirmée AVANT idx_cur)
        window_start     = max(p, len(h) - window)
        fractal_search_end = idx_cur - p  # bougies [window_start .. idx_cur - p - 1]

        if fractal_search_end <= window_start:
            continue

        # PATCH P5: collecte de TOUTES les fractales valides de la fenêtre,
        # puis sélection de la DERNIÈRE chronologiquement (index le plus élevé)
        last_high_idx = None
        last_low_idx  = None

        for i in range(window_start, fractal_search_end):
            # PATCH P9: tolérance relative sur l'égalité float
            local_max = max(h[i - p:i + p + 1])
            local_min = min(l[i - p:i + p + 1])
            if abs(h[i] - local_max) < 1e-9:
                last_high_idx = i   # on écrase → on garde le plus récent
            if abs(l[i] - local_min) < 1e-9:
                last_low_idx = i    # idem

        last_high = float(h[last_high_idx]) if last_high_idx is not None else None
        last_low  = float(l[last_low_idx])  if last_low_idx  is not None else None

        if last_high is None and last_low is None:
            continue

        if not is_valid_breakout_candle(idx_cur):
            continue

        breakout_range = h_all[idx_cur] - l_all[idx_cur]
        sig_raw        = None
        time_sig       = None
        force          = None
        niveau_casse   = None

        # Condition de cassure stricte : close[cur] franchit, close[prev] ne franchissait pas
        if last_high is not None and c[idx_cur] > last_high and c[idx_prev] <= last_high:
            sig_raw      = "Bullish"
            time_sig     = df.index[idx_cur]
            force        = get_force(breakout_range)
            niveau_casse = last_high

        elif last_low is not None and c[idx_cur] < last_low and c[idx_prev] >= last_low:
            sig_raw      = "Bearish"
            time_sig     = df.index[idx_cur]
            force        = get_force(breakout_range)
            niveau_casse = last_low

        if sig_raw is None:
            continue

        trend = get_trend_context(df.iloc[:idx_cur + 1], tf)

        # PATCH P13: filtre Range étendu — Moyen aussi filtré (faux breakouts en range)
        if trend == "Range" and force in ("Faible", "Moyen"):
            continue

        close_actuel = float(c_all[-1])
        return sig_raw, time_sig, force, idx_cur, trend, niveau_casse, close_actuel

    return None, None, None, None, None, None, None


def format_niveau(niveau: float | None, inst: str) -> str:
    if niveau is None:
        return "N/A"
    if any(k in inst for k in ["SPX500", "NAS100", "US30", "DE30", "XAU", "XAG"]):
        return f"{niveau:.2f}"
    if "JPY" in inst:
        return f"{niveau:.3f}"
    return f"{niveau:.5f}"


def format_distance(niveau: float | None, close_actuel: float | None, inst: str) -> str:
    # PATCH P9: garde robuste — isclose au lieu de == 0, sanity check > 100%
    if niveau is None or close_actuel is None or np.isclose(niveau, 0, atol=1e-8):
        return "N/A"
    dist = abs(close_actuel - niveau) / abs(niveau) * 100
    if dist > 100:  # valeur aberrante → sûrement un bug de données
        return "N/A"
    return f"{dist:.3f}%"


# PATCH P11: normalize_pair() supprimé (dead code — fonction identité pure)


# ===================== PDF =====================
def create_pdf(df_export: pd.DataFrame) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        leftMargin=20, rightMargin=20, topMargin=40, bottomMargin=40
    )
    elements = []
    styles   = getSampleStyleSheet()

    elements.append(Paragraph("Rapport des Signaux CHoCH", styles["Title"]))
    elements.append(Paragraph(
        f"Généré le {datetime.now(timezone.utc).strftime('%d/%m/%Y à %H:%M')} UTC"
        " — Fresh & Aged uniquement — Monthly exclu",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 20))

    cols_present = [c for c in EXPORT_COLS if c in df_export.columns]
    data         = [cols_present] + df_export[cols_present].values.tolist()
    col_widths   = [65, 48, 42, 42, 82, 68, 52, 58, 45, 90, 45, 105]
    col_widths   = col_widths[:len(cols_present)]

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ('TEXTCOLOR',     (0, 0), (-1, 0), colors.white),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME',      (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0, 0), (-1, 0), 9),
        ('FONTSIZE',      (0, 1), (-1, -1), 8),
        ('GRID',          (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [colors.white, colors.beige]),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING',   (0, 0), (-1, -1), 5),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 5),
        ('TOPPADDING',    (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ===================== PNG =====================
def generate_png(df: pd.DataFrame, display_cols: list) -> io.BytesIO:
    """
    PATCH P4: Utilisation stricte de l'API OOP matplotlib (Figure, pas plt).
    Évite le memory leak sur serveur Streamlit persistant (pas de pyplot global state).
    """
    fig = Figure(figsize=(22, max(5, len(df) * 0.35)))  # PATCH P4
    ax  = fig.add_subplot(111)
    ax.axis('off')
    disp = df[[c for c in display_cols if c in df.columns]]
    tbl  = ax.table(cellText=disp.values, colLabels=disp.columns,
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.8)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=200)  # PATCH P4
    buf.seek(0)
    return buf


# ===================== UI =====================
st.set_page_config(page_title="CHoCH Scanner", layout="wide")
st.markdown(
    "<h1 style='text-align:center;color:#1e40af;margin-bottom:30px;'>"
    "Scanner Change of Character (CHoCH) — v5.5</h1>",
    unsafe_allow_html=True
)

if "scanning" not in st.session_state:
    st.session_state.scanning = False

if st.button(
    "Lancer le Scan",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.scanning
):
    st.session_state.scanning = True
    n_combos = len(INSTRUMENTS) * len(TIMEFRAMES)

    # PATCH P2: try/finally garantit que scanning=False même en cas d'exception
    try:
        with st.spinner(f"Scan en cours sur {n_combos} combinaisons…"):
            results: list[dict] = []
            errors:  list[str]  = []

            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {
                    executor.submit(get_candles, inst, tf_code): (inst, tf_name)
                    for inst in INSTRUMENTS
                    for tf_name, tf_code in TIMEFRAMES.items()
                }
                for future in as_completed(futures):
                    inst, tf_name = futures[future]
                    try:
                        df = future.result()
                    except Exception as e:
                        errors.append(f"{inst}/{tf_name}: {e}")
                        continue

                    if df is None:
                        errors.append(f"{inst}/{tf_name}: données insuffisantes")
                        continue

                    sig_raw, time_sig, strength, idx_sig, trend, niveau_casse, close_actuel = \
                        detect_choch(df, tf_name)

                    if not sig_raw:
                        continue

                    atr_val    = calc_atr(df)
                    volatilite = atr_to_volatility(atr_val, inst, df)
                    df_sig     = df.iloc[:idx_sig + 1] if idx_sig is not None else df

                    # PATCH P6: convention SMC standard
                    # CHoCH = cassure contre la tendance (retournement potentiel)
                    # BOS   = cassure dans le sens de la tendance (continuation)
                    is_choch = (
                        (trend == "Uptrend"   and sig_raw == "Bearish") or
                        (trend == "Downtrend" and sig_raw == "Bullish")
                    )
                    type_label   = "CHoCH" if is_choch else "BOS"
                    signal_label = f"{sig_raw} {type_label}"

                    inst_display = inst.replace("_", "/")
                    # PATCH P11: inst_export = inst_display directement (normalize_pair supprimé)

                    # PATCH P3: statut calculé en bougies (weekend-safe)
                    statut = compute_statut(idx_sig, len(df), tf_name)

                    # PATCH P10: on stocke le datetime brut pour un tri déterministe
                    results.append({
                        "Instrument":  inst_display,
                        "Paire":       inst_display,
                        "_time_sort":  time_sig,               # PATCH P10: tri datetime brut
                        "Timeframe":   tf_name,
                        "Type":        type_label,
                        "Ordre":       "Achat" if sig_raw == "Bullish" else "Vente",
                        "Signal":      signal_label,
                        "Niveau":      format_niveau(niveau_casse, inst),
                        "Distance%":   format_distance(niveau_casse, close_actuel, inst),
                        "Volatilité":  volatilite,
                        "Force":       strength or "Moyen",
                        "BB_Width":    compute_bb_width(df_sig),
                        "Statut":      statut,
                        "Heure (UTC)": time_sig.strftime("%Y-%m-%d %H:%M"),
                    })

            if errors:
                st.warning(f"{len(errors)} combinaison(s) en erreur / données insuffisantes.")

            if results:
                # PATCH P10: tri sur datetime brut puis drop_duplicates → déterministe
                df_result = (
                    pd.DataFrame(results)
                    .sort_values("_time_sort", ascending=False)
                    .drop_duplicates(subset=["Instrument", "Timeframe"], keep="first")
                    .drop(columns=["_time_sort"])
                    .reset_index(drop=True)
                )
                st.session_state.df      = df_result
                st.session_state.png_buf = None
                st.success(f"Scan terminé – {len(df_result)} signaux sur {n_combos} combinaisons !")
            else:
                st.info("Aucun signal CHoCH/BOS récent détecté")

    except Exception as e:
        st.error(f"Erreur critique inattendue lors du scan : {e}")
        logger.exception("Erreur critique scan")
    finally:
        # PATCH P2: TOUJOURS déverrouiller le bouton, même si exception
        st.session_state.scanning = False


# ===================== AFFICHAGE =====================
if "df" in st.session_state:
    df_all = st.session_state.df.copy()
    ts     = datetime.now().strftime("%Y%m%d_%H%M")

    df_export = df_all[df_all["Statut"].isin(["Fresh", "Aged"])].copy()

    if st.session_state.get("png_buf") is None:
        st.session_state.png_buf = generate_png(df_all, DISPLAY_COLS)

    n_stale = len(df_all[df_all["Statut"] == "Stale"])
    if n_stale > 0:
        st.info(
            f"{n_stale} signal(s) Stale visible(s) dans le tableau — "
            "exclus du PDF/CSV/JSON (Fresh & Aged uniquement dans les exports)."
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button(
            "CSV",
            df_export[[c for c in EXPORT_COLS if c in df_export.columns]]
                .to_csv(index=False).encode(),
            f"choch_{ts}.csv", "text/csv"
        )
    with col2:
        st.session_state.png_buf.seek(0)
        st.download_button(
            "PNG",
            st.session_state.png_buf,
            f"choch_{ts}.png", "image/png"
        )
    with col3:
        st.download_button(
            "PDF",
            create_pdf(df_export),
            f"choch_signaux_{ts}.pdf", "application/pdf"
        )
    with col4:
        st.download_button(
            "JSON",
            df_export[[c for c in EXPORT_COLS if c in df_export.columns]]
                .to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
            f"choch_{ts}.json", "application/json"
        )

    # ===================== STYLES =====================
    def style_bb(val: str) -> str:
        if "Squeeze"   in str(val): return "color:#ff9800;font-weight:bold"
        if "Expansion" in str(val): return "color:#ab47bc;font-weight:bold"
        return "color:#90a4ae"

    def style_distance(val: str) -> str:
        try:
            v = float(str(val).replace("%", ""))
            if v <= 0.15: return "color:#00c853;font-weight:bold"
            if v <= 0.40: return "color:#ff9800;font-weight:bold"
            return "color:#ff5252;font-weight:bold"
        except Exception:
            return "color:#90a4ae"

    cols_display = [c for c in DISPLAY_COLS if c in df_all.columns]
    st.dataframe(
        df_all[cols_display].style
        .map(
            lambda x: "color:#e879f9;font-weight:bold" if x == "CHoCH"
            else "color:#94a3b8" if x == "BOS" else "",
            subset=["Type"]
        )
        .map(
            lambda x: "color:#00c853;font-weight:bold" if x == "Achat"
            else "color:#ff5252;font-weight:bold" if x == "Vente" else "",
            subset=["Ordre"]
        )
        .map(
            lambda x: "color:#00c853" if "Bull" in str(x)
            else "color:#ff5252" if "Bear" in str(x) else "",
            subset=["Signal"]
        )
        .map(
            lambda x: "color:#00c853;font-weight:bold" if x == "Fort"
            else "color:#ff5252" if x == "Faible"
            else "color:#ff9800" if x == "Moyen" else "",
            subset=["Force"]
        )
        .map(style_bb, subset=["BB_Width"])
        .map(style_distance, subset=["Distance%"])
        .map(
            lambda x: "color:#00c853;font-weight:bold" if x == "Fresh"
            else "color:#ff9800;font-weight:bold" if x == "Aged"
            else "color:#ff5252;font-weight:bold" if x == "Stale" else "",
            subset=["Statut"]
        ),
        hide_index=True,
        use_container_width=True
    )
