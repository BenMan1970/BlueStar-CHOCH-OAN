# ============================================================
# SCANNER SMC — v6.1 "BLUESTAR PIPELINE EDITION"
# Conçu exclusivement pour alimenter le prompt BLUESTAR DIRECT v6.0
#
# CE QUE CE SCANNER FAIT DANS LE PIPELINE :
#   → Détecte MSS/BOS réels (swing points H/L structurels)
#   → Fournit OB + FVG (zones d'entrée précises pour le LLM)
#   → Fournit TP naturel (prochain swing opposé = TP2 candidat)
#   → Fournit price_state (au-dessus/dans/sous les zones)
#   → Fournit is_valid (invalidation si niveau re-cassé)
#   → Deux scores séparés : structure_score (filtrage interne)
#     et confluence_score (visible dans le payload pour le LLM)
#   → Filtre de distance désactivé par défaut (c'est le LLM qui décide)
#   → Seuil de sortie bas (40) pour ne rien cacher au pipeline
#   → Toutes les paires scannées, même sans OB/FVG (flag absent)
#
# CE QUE CE SCANNER NE FAIT PAS (géré par les autres composants) :
#   → Biais directionnel (GPS MTF)
#   → Zones S/R institutionnelles (S/R scanner)
#   → Momentum RSI (RSI scanner)
#   → Calendrier économique (BLUESTAR DIRECT prompt)
#   → Sélection finale des setups (BLUESTAR DIRECT prompt)
#
# FORMAT DE SORTIE :
#   JSON pipeline-grade compatible BLUESTAR DIRECT v6.0
#   Chaque signal contient tout ce dont le LLM a besoin
#   pour appliquer les 7 étapes du prompt sans ambiguïté.
#
# INSTRUMENTS : alignés sur la liste GPS MTF + S/R scanner
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import logging
import threading

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Optional

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.exceptions import V20Error

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

SCANNER_VERSION = "6.1"

# ===================== INSTRUMENTS =====================
# Aligné sur GPS MTF + S/R scanner (même univers que BLUESTAR)
INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD",
    "AUD_USD", "NZD_USD", "EUR_GBP", "EUR_JPY", "EUR_CHF",
    "EUR_AUD", "EUR_CAD", "EUR_NZD", "GBP_JPY", "GBP_CHF",
    "GBP_AUD", "GBP_CAD", "GBP_NZD", "AUD_JPY", "AUD_CAD",
    "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "XAU_USD", "SPX500_USD", "NAS100_USD", "US30_USD", "DE30_EUR",
]

# Timeframes de signal (swing trading H4/D1)
TIMEFRAMES = {
    "H4": {"gran": "H4", "htf_gran": "D",  "signal_count": 350, "htf_count": 250},
    "D1": {"gran": "D",  "htf_gran": "W",  "signal_count": 250, "htf_count": 100},
}

VOLATILITY_STATIC = {
    "EUR_USD":"Basse","GBP_USD":"Basse","USD_JPY":"Basse","USD_CHF":"Basse","USD_CAD":"Basse",
    "AUD_USD":"Moyenne","NZD_USD":"Moyenne","EUR_GBP":"Moyenne","EUR_JPY":"Moyenne",
    "EUR_CHF":"Moyenne","EUR_AUD":"Moyenne","EUR_CAD":"Moyenne","EUR_NZD":"Moyenne",
    "GBP_JPY":"Haute","GBP_CHF":"Haute","GBP_AUD":"Haute","GBP_CAD":"Haute","GBP_NZD":"Haute",
    "AUD_JPY":"Haute","AUD_CAD":"Moyenne","AUD_CHF":"Haute","AUD_NZD":"Moyenne",
    "CAD_JPY":"Haute","CAD_CHF":"Haute","CHF_JPY":"Haute","NZD_JPY":"Haute",
    "NZD_CAD":"Moyenne","NZD_CHF":"Haute",
    "XAU_USD":"Très Haute","SPX500_USD":"Très Haute","NAS100_USD":"Très Haute",
    "US30_USD":"Très Haute","DE30_EUR":"Très Haute",
}

# ── Paramètres SMC ─────────────────────────────────────────────────────────────
SWING_LOOKBACK  = 10   # bougies de chaque côté pour valider un vrai swing point
SWING_HISTORY   = 60   # bougies max dans lesquelles chercher les swings
OB_MAX_LOOKBACK = 8    # bougies avant le breakout pour l'Order Block
FVG_LOOKBACK    = 5    # bougies autour du breakout pour le FVG
MIN_FVG_PCT     = 0.03 # FVG minimum en % (filtre micro-gaps)

# ── Seuils de scoring ──────────────────────────────────────────────────────────
# structure_score  : seuil interne bas → on sort tout ce qui est structurellement valide
# Le LLM BLUESTAR croise avec GPS/RSI/S/R pour décider
MIN_STRUCTURE_SCORE = 30   # seuil interne très bas : OB seul suffit
MIN_CONFLUENCE_SCORE = 0   # pas de filtre sur confluence_score en sortie pipeline

# ── Statut ─────────────────────────────────────────────────────────────────────
STATUT_THRESHOLDS = {
    "H4": {"Fresh": 3, "Aged": 10},
    "D1": {"Fresh": 2, "Aged": 6},
}

# ── Timeouts ───────────────────────────────────────────────────────────────────
SCAN_GLOBAL_TIMEOUT   = 240
FUTURE_RESULT_TIMEOUT = 25
MAX_WORKERS           = 8

# ── Colonnes UI ────────────────────────────────────────────────────────────────
DISPLAY_COLS = [
    "Paire", "TF", "Type", "Direction", "Sc.Struct", "Sc.Conf",
    "OB_Zone", "FVG_Zone", "TP_Naturel", "Price_State",
    "Valid", "HTF", "Session", "Statut", "Heure (UTC)"
]


# ===================== DATACLASSES =====================

@dataclass
class OrderBlock:
    top:      float
    bottom:   float
    idx:      int
    polarity: str   # "Bullish" | "Bearish"
    body_pct: float # qualité du corps (0-1)


@dataclass
class FairValueGap:
    top:      float
    bottom:   float
    idx:      int
    polarity: str
    size_pct: float  # taille relative en %


@dataclass
class SwingPoint:
    idx:   int
    price: float
    kind:  str   # "HH" | "HL" | "LH" | "LL"


@dataclass
class SMCSignal:
    # ── Identité ──────────────────────────────────────
    instrument:   str
    timeframe:    str
    direction:    str       # "Bullish" | "Bearish"
    sig_type:     str       # "MSS" | "BOS"
    signal_time:  datetime

    # ── Niveaux ───────────────────────────────────────
    broken_level:  float
    close_price:   float
    distance_pct:  Optional[float]

    # ── Zones d'entrée ────────────────────────────────
    ob:   Optional[OrderBlock]
    fvg:  Optional[FairValueGap]

    # ── TP naturel (prochain swing opposé) ────────────
    tp_natural:    Optional[float]   # niveau du prochain swing opposé

    # ── État du signal au moment du scan ──────────────
    price_state:   str   # "above_ob"|"inside_ob"|"inside_fvg"|"below_fvg"|"below_ob"|"no_ob"
    is_valid:      bool  # False si niveau re-cassé dans l'autre sens

    # ── Contexte HTF ──────────────────────────────────
    htf_trend:    str    # "Bullish"|"Bearish"|"Range"|"Unknown"
    htf_aligned:  bool

    # ── Scores ────────────────────────────────────────
    # structure_score : qualité SMC pure (OB/FVG/swing) — filtre interne
    # confluence_score : structure + HTF + session → visible dans payload pour LLM
    structure_score:   int
    confluence_score:  int

    # ── Contexte ──────────────────────────────────────
    session:       str
    volatility:    str
    atr_h4:        float   # ATR H4 exposé pour calcul SL dans BLUESTAR prompt
    statut:        str
    candles_since: int
    scan_time:     datetime


# ===================== API THREAD-SAFE =====================
_thread_local = threading.local()

def _get_api() -> API:
    if not hasattr(_thread_local, "api"):
        _thread_local.api = API(
            access_token=st.secrets["OANDA_ACCESS_TOKEN"],
            request_params={"timeout": 15}
        )
    return _thread_local.api

try:
    _ = st.secrets["OANDA_ACCESS_TOKEN"]
except Exception as e:
    st.error(f"Token OANDA manquant : {e}")
    st.stop()


# ===================== UTILITAIRES =====================

def instrument_precision(inst: str) -> int:
    if any(k in inst for k in ["SPX500", "NAS100", "US30", "DE30", "XAU", "XAG"]):
        return 2
    if "JPY" in inst:
        return 3
    return 5


def fmt_price(price: float, inst: str) -> str:
    prec = instrument_precision(inst)
    return f"{price:.{prec}f}"


def fmt_zone(top: float, bot: float, inst: str) -> str:
    return f"{fmt_price(bot, inst)}–{fmt_price(top, inst)}"


def calc_distance_pct(level: float, close: float) -> Optional[float]:
    if level is None or close is None or np.isclose(level, 0, atol=1e-8):
        return None
    dist = abs(close - level) / abs(level) * 100
    return round(dist, 4) if dist <= 100 else None


def get_session(dt: datetime) -> str:
    h = dt.hour
    london = 7 <= h < 16
    ny     = 13 <= h < 22
    tokyo  = 0 <= h < 9
    if london and ny: return "London_NY_Overlap"
    if london:        return "London"
    if ny:            return "NewYork"
    if tokyo:         return "Tokyo"
    return "Off"


def is_session_premium(session: str) -> bool:
    return session in ("London", "NewYork", "London_NY_Overlap")


def calc_atr(df: pd.DataFrame, period: int = 14) -> float:
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.maximum(h[1:] - l[1:], np.maximum(
        np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])
    ))
    if len(tr) < period * 2:
        return float("nan")
    return float(pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


def calc_volatility(atr: float, df: pd.DataFrame, inst: str) -> str:
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    tr = np.maximum(h[1:] - l[1:], np.maximum(
        np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])
    ))
    if np.isnan(atr) or len(tr) < 10:
        return VOLATILITY_STATIC.get(inst, "Moyenne")
    window    = tr[-100:] if len(tr) >= 100 else tr
    median_tr = float(np.median(window))
    if np.isclose(median_tr, 0, atol=1e-10):
        return VOLATILITY_STATIC.get(inst, "Moyenne")
    ratio = atr / median_tr
    if ratio >= 1.8: return "Très Haute"
    if ratio >= 1.2: return "Haute"
    if ratio >= 0.7: return "Moyenne"
    return "Basse"


def compute_statut(candles_since: int, tf: str) -> str:
    t = STATUT_THRESHOLDS.get(tf, {"Fresh": 2, "Aged": 6})
    if candles_since <= t["Fresh"]: return "Fresh"
    if candles_since <= t["Aged"]:  return "Aged"
    return "Stale"


# ===================== CANDLES =====================

def get_candles(inst: str, gran: str, count: int) -> Optional[pd.DataFrame]:
    try:
        r = instruments.InstrumentsCandles(
            instrument=inst,
            params={"count": count, "granularity": gran}
        )
        _get_api().request(r)
        candles = [c for c in r.response.get("candles", []) if c.get("complete")]
        if len(candles) < 60:
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
        logger.warning(f"V20Error [{inst}/{gran}] code={e.code}: {e}")
        return None
    except Exception as e:
        logger.error(f"Erreur [{inst}/{gran}]: {type(e).__name__}: {e}")
        return None


# ===================== SMC — SWING POINTS =====================

def detect_swing_points(df: pd.DataFrame,
                        lookback: int = SWING_LOOKBACK,
                        history:  int = SWING_HISTORY) -> list[SwingPoint]:
    """
    Détecte les vrais pivots de structure (lookback bougies de chaque côté).
    Classe chaque pivot en HH/HL/LH/LL selon la structure séquentielle.
    """
    h = df["high"].values
    l = df["low"].values
    n = len(h)

    start = max(lookback, n - history - lookback)
    end   = n - lookback - 1

    raw: list[tuple[int, float, str]] = []
    for i in range(start, end):
        if h[i] == max(h[i - lookback:i + lookback + 1]):
            raw.append((i, h[i], "H"))
        if l[i] == min(l[i - lookback:i + lookback + 1]):
            raw.append((i, l[i], "L"))

    raw.sort(key=lambda x: x[0])

    swings: list[SwingPoint] = []
    prev_high: Optional[float] = None
    prev_low:  Optional[float] = None

    for idx, price, kind in raw:
        if kind == "H":
            label = "HH" if (prev_high is None or price > prev_high) else "LH"
            swings.append(SwingPoint(idx=idx, price=price, kind=label))
            prev_high = price
        else:
            label = "HL" if (prev_low is None or price > prev_low) else "LL"
            swings.append(SwingPoint(idx=idx, price=price, kind=label))
            prev_low = price

    return swings


def get_market_structure(swings: list[SwingPoint]) -> str:
    if len(swings) < 4:
        return "Unknown"
    recent = swings[-6:]
    highs  = [s for s in recent if s.kind in ("HH", "LH")]
    lows   = [s for s in recent if s.kind in ("HL", "LL")]
    if not highs or not lows:
        return "Unknown"
    if highs[-1].kind == "HH" and lows[-1].kind == "HL":
        return "Bullish"
    if highs[-1].kind == "LH" and lows[-1].kind == "LL":
        return "Bearish"
    return "Range"


# ===================== SMC — MSS / BOS =====================

def detect_mss_bos(df: pd.DataFrame, swings: list[SwingPoint],
                   structure: str) -> tuple[
                       Optional[str], Optional[str],
                       Optional[float], Optional[int]]:
    """
    MSS = cassure CONTRE la structure (retournement)
    BOS = cassure DANS la structure (continuation)
    Confirmation : clôture au-delà du niveau.
    Niveau doit avoir tenu ≥ 3 bougies avant le breakout.
    Recherche sur les 5 dernières bougies.
    """
    if structure == "Unknown" or not swings:
        return None, None, None, None

    c = df["close"].values
    n = len(c)

    for offset in range(5):
        idx = n - 1 - offset
        if idx < 3:
            break

        if structure == "Bullish":
            hl = [s for s in swings if s.kind == "HL" and s.idx < idx - 2]
            if hl and c[idx] < hl[-1].price and c[idx - 1] >= hl[-1].price:
                return "MSS", "Bearish", hl[-1].price, idx
            hh = [s for s in swings if s.kind == "HH" and s.idx < idx - 2]
            if hh and c[idx] > hh[-1].price and c[idx - 1] <= hh[-1].price:
                return "BOS", "Bullish", hh[-1].price, idx

        elif structure == "Bearish":
            lh = [s for s in swings if s.kind == "LH" and s.idx < idx - 2]
            if lh and c[idx] > lh[-1].price and c[idx - 1] <= lh[-1].price:
                return "MSS", "Bullish", lh[-1].price, idx
            ll = [s for s in swings if s.kind == "LL" and s.idx < idx - 2]
            if ll and c[idx] < ll[-1].price and c[idx - 1] >= ll[-1].price:
                return "BOS", "Bearish", ll[-1].price, idx

        else:  # Range : MSS sur les extrêmes seulement
            all_h = [s for s in swings if s.kind in ("HH","LH") and s.idx < idx - 2]
            all_l = [s for s in swings if s.kind in ("HL","LL") and s.idx < idx - 2]
            if all_h:
                rh = max(all_h, key=lambda s: s.price)
                if c[idx] > rh.price and c[idx - 1] <= rh.price:
                    return "MSS", "Bullish", rh.price, idx
            if all_l:
                rl = min(all_l, key=lambda s: s.price)
                if c[idx] < rl.price and c[idx - 1] >= rl.price:
                    return "MSS", "Bearish", rl.price, idx

    return None, None, None, None


# ===================== SMC — ORDER BLOCK =====================

def detect_order_block(df: pd.DataFrame, direction: str,
                       breakout_idx: int) -> Optional[OrderBlock]:
    """
    Dernière bougie de couleur opposée avant le mouvement impulsif.
    Corps ≥ 40% de la range pour filtrer les dojis.
    """
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values

    search_start = max(0, breakout_idx - OB_MAX_LOOKBACK)
    last_ob_idx: Optional[int] = None

    for i in range(search_start, breakout_idx):
        rng  = h[i] - l[i]
        body = abs(c[i] - o[i])
        if np.isclose(rng, 0, atol=1e-10) or (body / rng) < 0.40:
            continue
        if direction == "Bullish" and c[i] < o[i]:
            last_ob_idx = i
        elif direction == "Bearish" and c[i] > o[i]:
            last_ob_idx = i

    if last_ob_idx is None:
        return None

    rng  = float(h[last_ob_idx] - l[last_ob_idx])
    body = float(abs(c[last_ob_idx] - o[last_ob_idx]))
    return OrderBlock(
        top      = float(h[last_ob_idx]),
        bottom   = float(l[last_ob_idx]),
        idx      = last_ob_idx,
        polarity = direction,
        body_pct = round(body / rng, 2) if not np.isclose(rng, 0) else 0.0,
    )


# ===================== SMC — FAIR VALUE GAP =====================

def detect_fvg(df: pd.DataFrame, direction: str,
               breakout_idx: int) -> Optional[FairValueGap]:
    """
    Gap entre mèche[i-1] et mèche[i+1] autour de la bougie impulsive.
    Filtre : gap > MIN_FVG_PCT% du prix.
    Garde le plus grand FVG trouvé dans la fenêtre.
    """
    h = df["high"].values
    l = df["low"].values
    n = len(h)

    search_start = max(1, breakout_idx - FVG_LOOKBACK)
    search_end   = min(n - 1, breakout_idx + 1)

    best: Optional[FairValueGap] = None
    best_size = 0.0

    for i in range(search_start, search_end):
        if i + 1 >= n:
            break
        if direction == "Bullish":
            gap_bot = h[i - 1]
            gap_top = l[i + 1]
            if gap_top > gap_bot:
                size = gap_top - gap_bot
                mid  = (gap_top + gap_bot) / 2
                size_pct = size / mid * 100 if mid > 0 else 0
                if size_pct >= MIN_FVG_PCT and size > best_size:
                    best = FairValueGap(top=gap_top, bottom=gap_bot,
                                        idx=i, polarity="Bullish",
                                        size_pct=round(size_pct, 4))
                    best_size = size
        elif direction == "Bearish":
            gap_top = l[i - 1]
            gap_bot = h[i + 1]
            if gap_top > gap_bot:
                size = gap_top - gap_bot
                mid  = (gap_top + gap_bot) / 2
                size_pct = size / mid * 100 if mid > 0 else 0
                if size_pct >= MIN_FVG_PCT and size > best_size:
                    best = FairValueGap(top=gap_top, bottom=gap_bot,
                                        idx=i, polarity="Bearish",
                                        size_pct=round(size_pct, 4))
                    best_size = size

    return best


# ===================== SMC — TP NATUREL =====================

def detect_tp_natural(swings: list[SwingPoint], direction: str,
                      breakout_idx: int) -> Optional[float]:
    """
    TP naturel = prochain swing point OPPOSÉ au-delà du niveau cassé.
    Bullish → prochain HH ou LH (résistance naturelle)
    Bearish → prochain HL ou LL (support naturel)
    Utilisé comme TP2 candidat dans BLUESTAR DIRECT step 5.
    """
    if direction == "Bullish":
        candidates = [s for s in swings
                      if s.kind in ("HH", "LH") and s.idx > breakout_idx]
    else:
        candidates = [s for s in swings
                      if s.kind in ("HL", "LL") and s.idx > breakout_idx]

    if not candidates:
        # Pas de swing post-breakout dans l'historique : chercher le swing opposé
        # le plus récent AVANT le breakout comme cible potentielle
        if direction == "Bullish":
            prior = [s for s in swings
                     if s.kind in ("HH", "LH") and s.idx < breakout_idx]
            if prior:
                above = [s for s in prior if s.price > swings[-1].price
                         if swings else True]
                return float(prior[-1].price) if prior else None
        else:
            prior = [s for s in swings
                     if s.kind in ("HL", "LL") and s.idx < breakout_idx]
            return float(prior[0].price) if prior else None
    return float(candidates[0].price)


# ===================== SMC — PRICE STATE =====================

def compute_price_state(close: float, ob: Optional[OrderBlock],
                        fvg: Optional[FairValueGap],
                        direction: str, broken_level: float) -> tuple[str, bool]:
    """
    Calcule l'état du prix par rapport aux zones et valide le signal.

    price_state :
      "above_ob"   : prix au-dessus de l'OB (Bearish) / en-dessous (Bullish) → retest attendu
      "inside_fvg" : prix dans le FVG → zone d'entrée active
      "inside_ob"  : prix dans l'OB → zone d'entrée active
      "below_fvg"  : prix sous le FVG (Bearish déjà passé) → peut être trop loin
      "no_ob"      : pas d'OB détecté
      "invalidated": prix a re-cassé le niveau dans l'autre sens

    is_valid :
      False si le prix a re-cassé le broken_level dans le sens opposé au signal.
    """
    # Vérification invalidation
    if direction == "Bullish" and close < broken_level:
        return "invalidated", False
    if direction == "Bearish" and close > broken_level:
        return "invalidated", False

    if ob is None:
        return "no_ob", True

    if direction == "Bearish":
        if close > ob.top:
            state = "above_ob"      # prix au-dessus de l'OB → retest en attente
        elif ob.bottom <= close <= ob.top:
            # dans l'OB : affiner si FVG disponible
            if fvg and fvg.bottom <= close <= fvg.top:
                state = "inside_fvg"
            else:
                state = "inside_ob"
        else:
            state = "below_fvg"     # prix déjà passé sous l'OB
    else:  # Bullish
        if close < ob.bottom:
            state = "above_ob"      # prix sous l'OB (pour Bullish) → retest en attente
        elif ob.bottom <= close <= ob.top:
            if fvg and fvg.bottom <= close <= fvg.top:
                state = "inside_fvg"
            else:
                state = "inside_ob"
        else:
            state = "below_fvg"

    return state, True


# ===================== SMC — SCORING =====================

def compute_scores(ob: Optional[OrderBlock], fvg: Optional[FairValueGap],
                   htf_aligned: bool, session: str,
                   sig_type: str, price_state: str) -> tuple[int, int]:
    """
    structure_score (0-60) : qualité SMC pure — filtre interne
      OB présent et de qualité  : +30
      FVG présent               : +20
      MSS (vs BOS)              : +10

    confluence_score (0-100) : structure + contexte — exposé au LLM
      = structure_score
      + HTF aligné              : +25
      + Session premium         : +10
      + Price state favorable   : +5
    """
    s = 0
    if ob is not None:
        s += 30
        if ob.body_pct >= 0.6:   # OB de haute qualité
            s += 5
    if fvg is not None:
        s += 20
    if sig_type == "MSS":
        s += 10
    structure_score = min(s, 60)

    c = structure_score
    if htf_aligned:
        c += 25
    if is_session_premium(session):
        c += 10
    if price_state in ("inside_fvg", "inside_ob", "above_ob"):
        c += 5
    confluence_score = min(c, 100)

    return structure_score, confluence_score


# ===================== HTF CONTEXT =====================

def get_htf_trend(inst: str, htf_gran: str, htf_count: int) -> str:
    df_htf = get_candles(inst, htf_gran, htf_count)
    if df_htf is None or len(df_htf) < 40:
        return "Unknown"
    swings = detect_swing_points(df_htf, lookback=5, history=30)
    return get_market_structure(swings)


# ===================== PIPELINE PAR INSTRUMENT =====================

def scan_instrument(inst: str, tf_name: str, tf_config: dict,
                    scan_time: datetime) -> Optional[SMCSignal]:
    """
    Pipeline complet pour un instrument + timeframe.
    Retourne un SMCSignal si structure_score >= MIN_STRUCTURE_SCORE.
    Pas de filtre de distance : le LLM BLUESTAR décide.
    """
    gran      = tf_config["gran"]
    htf_gran  = tf_config["htf_gran"]
    sig_count = tf_config["signal_count"]
    htf_count = tf_config["htf_count"]

    # ── 1. Données ────────────────────────────────────────────────────────────
    df = get_candles(inst, gran, sig_count)
    if df is None or len(df) < 80:
        return None

    # ── 2. Structure ──────────────────────────────────────────────────────────
    swings    = detect_swing_points(df)
    structure = get_market_structure(swings)

    # ── 3. MSS / BOS ──────────────────────────────────────────────────────────
    sig_type, direction, broken_level, breakout_idx = detect_mss_bos(
        df, swings, structure
    )
    if not sig_type:
        return None

    # ── 4. HTF ────────────────────────────────────────────────────────────────
    htf_trend   = get_htf_trend(inst, htf_gran, htf_count)
    htf_aligned = (
        (direction == "Bullish" and htf_trend == "Bullish") or
        (direction == "Bearish" and htf_trend == "Bearish")
    )

    # ── 5. OB + FVG (sur df tronqué au moment du signal) ─────────────────────
    df_ctx = df.iloc[:breakout_idx + 1]
    ob     = detect_order_block(df_ctx, direction, breakout_idx)
    fvg    = detect_fvg(df_ctx, direction, breakout_idx)

    # ── 6. TP naturel ─────────────────────────────────────────────────────────
    tp_natural = detect_tp_natural(swings, direction, breakout_idx)

    # ── 7. Session ────────────────────────────────────────────────────────────
    signal_time = df.index[breakout_idx]
    session     = get_session(signal_time)

    # ── 8. Price state + validité ─────────────────────────────────────────────
    close_price = float(df["close"].iloc[-1])
    price_state, is_valid = compute_price_state(
        close_price, ob, fvg, direction, float(broken_level)
    )

    # ── 9. Scores ─────────────────────────────────────────────────────────────
    structure_score, confluence_score = compute_scores(
        ob, fvg, htf_aligned, session, sig_type, price_state
    )

    # Filtre interne minimal : au moins un OB ou FVG
    if structure_score < MIN_STRUCTURE_SCORE:
        logger.info(f"Ignoré {inst}/{tf_name}: structure_score {structure_score}")
        return None

    # ── 10. Statut + volatilité + ATR ─────────────────────────────────────────
    candles_since = (len(df) - 1) - breakout_idx
    statut        = compute_statut(candles_since, tf_name)
    atr_val       = calc_atr(df_ctx)
    volatility    = calc_volatility(atr_val, df_ctx, inst)

    return SMCSignal(
        instrument     = inst,
        timeframe      = tf_name,
        direction      = direction,
        sig_type       = sig_type,
        signal_time    = signal_time,
        broken_level   = float(broken_level),
        close_price    = close_price,
        distance_pct   = calc_distance_pct(float(broken_level), close_price),
        ob             = ob,
        fvg            = fvg,
        tp_natural     = tp_natural,
        price_state    = price_state,
        is_valid       = is_valid,
        htf_trend      = htf_trend,
        htf_aligned    = htf_aligned,
        structure_score   = structure_score,
        confluence_score  = confluence_score,
        session        = session,
        volatility     = volatility,
        atr_h4         = round(atr_val, instrument_precision(inst) + 1)
                         if not np.isnan(atr_val) else 0.0,
        statut         = statut,
        candles_since  = candles_since,
        scan_time      = scan_time,
    )


# ===================== PAYLOAD JSON PIPELINE =====================

def build_pipeline_payload(sig: SMCSignal) -> dict:
    """
    Payload JSON optimisé pour BLUESTAR DIRECT v6.0.
    Contient tout ce dont le LLM a besoin pour les étapes 4, 5, 6 du prompt.
    """
    prec = instrument_precision(sig.instrument)
    inst = sig.instrument

    return {
        # ── Identité ───────────────────────────────────────────────────────────
        "signal_id":       f"{inst}__{sig.timeframe}__{sig.signal_time.strftime('%Y%m%dT%H%M')}",
        "scanner_version": SCANNER_VERSION,
        "generated_at":    sig.scan_time.isoformat(),

        # ── Instrument ────────────────────────────────────────────────────────
        "pair":            inst.replace("_", "/"),
        "pair_oanda":      inst,
        "timeframe":       sig.timeframe,

        # ── Signal SMC ────────────────────────────────────────────────────────
        "type":            sig.sig_type,          # "MSS" | "BOS"
        "direction":       sig.direction,          # "Bullish" | "Bearish"
        "is_bullish":      sig.direction == "Bullish",
        "order":           "buy" if sig.direction == "Bullish" else "sell",

        # ── Validité — critique pour pipeline ─────────────────────────────────
        "is_valid":        sig.is_valid,           # False = signal invalidé
        "price_state":     sig.price_state,        # état du prix vs zones

        # ── Scores ────────────────────────────────────────────────────────────
        "structure_score":  sig.structure_score,   # qualité SMC pure (0-60)
        "confluence_score": sig.confluence_score,  # SMC + HTF + session (0-100)

        # ── HTF context ───────────────────────────────────────────────────────
        "htf_trend":        sig.htf_trend,
        "htf_aligned":      sig.htf_aligned,

        # ── Niveaux ───────────────────────────────────────────────────────────
        "broken_level":    round(sig.broken_level, prec),
        "close_price":     round(sig.close_price, prec),
        "distance_pct":    sig.distance_pct,

        # ── Order Block ───────────────────────────────────────────────────────
        # Zone d'entrée institutionnelle — Entry dans BLUESTAR step 5
        "order_block": {
            "top":      round(sig.ob.top,      prec),
            "bottom":   round(sig.ob.bottom,   prec),
            "body_pct": sig.ob.body_pct,
        } if sig.ob else None,

        # ── Fair Value Gap ────────────────────────────────────────────────────
        # Imbalance — affine l'entrée dans l'OB
        "fair_value_gap": {
            "top":      round(sig.fvg.top,      prec),
            "bottom":   round(sig.fvg.bottom,   prec),
            "size_pct": sig.fvg.size_pct,
        } if sig.fvg else None,

        # ── TP naturel ────────────────────────────────────────────────────────
        # Candidat TP2 pour BLUESTAR step 5
        "tp_natural":      round(sig.tp_natural, prec) if sig.tp_natural else None,

        # ── ATR exposé ────────────────────────────────────────────────────────
        # Utilisé par BLUESTAR step 5 pour calcul SL (ATR × BB_mult)
        # Exprimé dans le même timeframe que le signal
        "atr":             sig.atr_h4,

        # ── Contexte ──────────────────────────────────────────────────────────
        "volatility":      sig.volatility,
        "signal_time":     sig.signal_time.isoformat(),
        "session":         sig.session,
        "candles_elapsed": sig.candles_since,
        "status":          sig.statut,
    }


# ===================== EXPORT PDF =====================

def create_pdf(rows: list[dict], scan_time: datetime) -> io.BytesIO:
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=landscape(A4),
                               leftMargin=12, rightMargin=12,
                               topMargin=28, bottomMargin=28)
    elements = []
    styles   = getSampleStyleSheet()
    elements.append(Paragraph(
        f"BLUESTAR · SMC Scanner v{SCANNER_VERSION} — Composant CHoCH/MSS/BOS",
        styles["Title"]
    ))
    elements.append(Paragraph(
        f"Généré le {scan_time.strftime('%d/%m/%Y à %H:%M')} UTC "
        f"| Pipeline BLUESTAR DIRECT v6.0",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 14))

    if not rows:
        elements.append(Paragraph("Aucun signal SMC détecté.", styles["Normal"]))
        doc.build(elements)
        buffer.seek(0)
        return buffer

    headers = ["Paire","TF","Type","Dir","Sc.S","Sc.C","OB Zone","FVG Zone",
               "TP Nat.","State","Valid","HTF","Session","Statut","Heure UTC"]
    data = [headers]
    for r in rows:
        data.append([
            r.get("Paire",""), r.get("TF",""), r.get("Type",""),
            r.get("Direction",""), str(r.get("Sc.Struct","")),
            str(r.get("Sc.Conf","")), r.get("OB_Zone",""), r.get("FVG_Zone",""),
            r.get("TP_Naturel",""), r.get("Price_State",""),
            "✓" if r.get("Valid") else "✗",
            r.get("HTF",""), r.get("Session",""), r.get("Statut",""),
            r.get("Heure (UTC)",""),
        ])

    col_w = [46,30,36,40,30,30,78,78,60,56,30,46,72,36,82]
    table = Table(data, colWidths=col_w, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0),  colors.HexColor("#1B45B4")),
        ('TEXTCOLOR',     (0,0),(-1,0),  colors.white),
        ('ALIGN',         (0,0),(-1,-1), 'CENTER'),
        ('FONTNAME',      (0,0),(-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,0),  7),
        ('FONTSIZE',      (0,1),(-1,-1), 6.5),
        ('GRID',          (0,0),(-1,-1), 0.3, colors.HexColor("#dde3f5")),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.white, colors.HexColor("#f0f3fa")]),
        ('VALIGN',        (0,0),(-1,-1), 'MIDDLE'),
        ('LEFTPADDING',   (0,0),(-1,-1), 3),
        ('RIGHTPADDING',  (0,0),(-1,-1), 3),
        ('TOPPADDING',    (0,0),(-1,-1), 4),
        ('BOTTOMPADDING', (0,0),(-1,-1), 4),
    ]))
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ===================== SIGNAL → ROW UI =====================

def signal_to_row(sig: SMCSignal) -> dict:
    inst = sig.instrument
    return {
        "Paire":        inst.replace("_", "/"),
        "TF":           sig.timeframe,
        "Type":         sig.sig_type,
        "Direction":    ("↑ Bull" if sig.direction == "Bullish" else "↓ Bear"),
        "Sc.Struct":    sig.structure_score,
        "Sc.Conf":      sig.confluence_score,
        "OB_Zone":      fmt_zone(sig.ob.top, sig.ob.bottom, inst) if sig.ob  else "—",
        "FVG_Zone":     fmt_zone(sig.fvg.top, sig.fvg.bottom, inst) if sig.fvg else "—",
        "TP_Naturel":   fmt_price(sig.tp_natural, inst) if sig.tp_natural else "—",
        "Price_State":  sig.price_state,
        "Valid":        sig.is_valid,
        "HTF":          sig.htf_trend,
        "Session":      sig.session,
        "Statut":       sig.statut,
        "Heure (UTC)":  sig.signal_time.strftime("%Y-%m-%d %H:%M"),
        "_time_sort":   sig.signal_time,
        "_sc_sort":     sig.confluence_score,
    }


# ===================== UI STREAMLIT =====================
st.set_page_config(page_title="BLUESTAR · SMC v6.1", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
body { font-family: 'IBM Plex Sans', sans-serif; }
.bs-header { border-bottom: 2px solid #1B45B4; padding-bottom: 10px; margin-bottom: 16px; }
.bs-title  { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem;
             font-weight: 700; color: #1B45B4; letter-spacing: 2px; }
.bs-sub    { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
             color: #6B89D8; margin-top: 2px; }
</style>
<div class="bs-header">
  <div class="bs-title">◈ BLUESTAR · SMC SCANNER v6.1</div>
  <div class="bs-sub">COMPOSANT CHoCH/MSS/BOS · PIPELINE BLUESTAR DIRECT v6.0
  · Swing H4/D1 · OB · FVG · TP Naturel · Price State · Validation</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Filtres d'affichage")
    st.caption("Ces filtres n'affectent pas le JSON pipeline — tous les signaux valides sont toujours exportés.")

    show_invalid   = st.checkbox("Afficher signaux invalidés", value=False)
    show_stale     = st.checkbox("Afficher signaux Stale", value=False)
    htf_only_ui    = st.checkbox("HTF aligné uniquement (UI)", value=False)
    min_sc_ui      = st.slider("Score confluence min (UI)", 0, 100, 0, 5)
    selected_tfs   = st.multiselect("Timeframes", list(TIMEFRAMES.keys()),
                                     default=list(TIMEFRAMES.keys()))
    st.markdown("---")
    st.markdown("**Scoring SMC :**")
    st.markdown("🔷 OB présent : +30 (+5 si qualité)")
    st.markdown("🔷 FVG présent : +20")
    st.markdown("🔷 MSS (vs BOS) : +10")
    st.markdown("🔶 HTF aligné : +25")
    st.markdown("🔶 Session premium : +10")
    st.markdown("🔶 Price state favorable : +5")
    st.markdown("---")
    st.markdown("**Price states :**")
    st.markdown("`inside_fvg` → entrée idéale")
    st.markdown("`inside_ob` → entrée valide")
    st.markdown("`above_ob` → retest en attente")
    st.markdown("`below_fvg` → signal trop tardif")
    st.markdown("`invalidated` → signal annulé")

if "scanning" not in st.session_state:
    st.session_state.scanning = False

# ── Bouton scan ───────────────────────────────────────────────────────────────
if st.button("🔍  Scanner — Composant CHoCH/SMC", type="primary",
             use_container_width=True, disabled=st.session_state.scanning):

    st.session_state.scanning  = True
    st.session_state.scan_time = datetime.now(timezone.utc)
    scan_time  = st.session_state.scan_time
    tfs_to_run = {k: v for k, v in TIMEFRAMES.items() if k in selected_tfs}
    n_combos   = len(INSTRUMENTS) * len(tfs_to_run)

    try:
        with st.spinner(f"Scan SMC en cours — {n_combos} combinaisons…"):
            signals:  list[SMCSignal] = []
            payloads: list[dict]      = []
            errors:   list[str]       = []

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_map = {
                    executor.submit(scan_instrument, inst, tf_name, tf_cfg, scan_time):
                        (inst, tf_name)
                    for inst    in INSTRUMENTS
                    for tf_name, tf_cfg in tfs_to_run.items()
                }
                try:
                    for future in as_completed(future_map, timeout=SCAN_GLOBAL_TIMEOUT):
                        inst, tf_name = future_map[future]
                        try:
                            result = future.result(timeout=FUTURE_RESULT_TIMEOUT)
                            if result is not None:
                                signals.append(result)
                                # Pipeline JSON : tous les signaux valides Fresh/Aged
                                if result.is_valid and result.statut in ("Fresh", "Aged"):
                                    payloads.append(build_pipeline_payload(result))
                        except FuturesTimeoutError:
                            errors.append(f"{inst}/{tf_name}: timeout")
                            future.cancel()
                        except Exception as e:
                            errors.append(f"{inst}/{tf_name}: {e}")
                except FuturesTimeoutError:
                    st.warning("Timeout global — résultats partiels.")

        if errors:
            with st.expander(f"⚠️ {len(errors)} erreur(s)"):
                for e in errors[:20]:
                    st.text(e)

        st.session_state.signals  = signals
        st.session_state.payloads = payloads

        n_valid   = sum(1 for s in signals if s.is_valid)
        n_htf     = sum(1 for s in signals if s.htf_aligned and s.is_valid)
        n_fresh   = sum(1 for s in signals if s.statut == "Fresh" and s.is_valid)
        n_pipeline = len(payloads)

        st.success(
            f"✅ **{len(signals)} signaux SMC** détectés · "
            f"{n_valid} valides · {n_htf} HTF alignés · "
            f"{n_fresh} Fresh · **{n_pipeline} dans le pipeline JSON**"
        )

    except Exception as e:
        st.error(f"Erreur critique : {e}")
        logger.exception("Erreur critique scan SMC v6.1")
    finally:
        st.session_state.scanning = False


# ===================== AFFICHAGE =====================
if "signals" in st.session_state and st.session_state.signals:
    signals:  list[SMCSignal] = st.session_state.signals
    payloads: list[dict]      = st.session_state.get("payloads", [])
    scan_time_meta = st.session_state.get("scan_time", datetime.now(timezone.utc))
    ts = scan_time_meta.strftime("%Y%m%d_%H%M")

    # ── Filtres UI (n'affectent pas le JSON export) ───────────────────────────
    filtered_ui = signals[:]
    if not show_invalid:
        filtered_ui = [s for s in filtered_ui if s.is_valid]
    if not show_stale:
        filtered_ui = [s for s in filtered_ui if s.statut != "Stale"]
    if htf_only_ui:
        filtered_ui = [s for s in filtered_ui if s.htf_aligned]
    filtered_ui = [s for s in filtered_ui if s.confluence_score >= min_sc_ui]

    # ── Métriques ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Signaux totaux",      len(signals))
    c2.metric("Valides",             sum(1 for s in signals if s.is_valid))
    c3.metric("MSS",                 sum(1 for s in signals if s.sig_type == "MSS" and s.is_valid))
    c4.metric("OB + FVG",            sum(1 for s in signals if s.ob and s.fvg and s.is_valid))
    c5.metric("HTF alignés",         sum(1 for s in signals if s.htf_aligned and s.is_valid))
    c6.metric("Pipeline JSON",       len(payloads))

    st.divider()

    # ── Tableau ───────────────────────────────────────────────────────────────
    if filtered_ui:
        rows = [signal_to_row(s) for s in filtered_ui]
        df_ui = (
            pd.DataFrame(rows)
            .sort_values(["_sc_sort", "_time_sort"], ascending=[False, False])
            .drop(columns=["_time_sort", "_sc_sort"])
            .reset_index(drop=True)
        )

        def sty_type(v):
            return "color:#f97316;font-weight:800" if v == "MSS" else "color:#60a5fa;font-weight:600"
        def sty_dir(v):
            if "Bull" in str(v): return "color:#4ade80;font-weight:700"
            if "Bear" in str(v): return "color:#f87171;font-weight:700"
            return ""
        def sty_score_s(v):
            try:
                n = int(v)
                if n >= 50: return "color:#4ade80;font-weight:800"
                if n >= 35: return "color:#facc15"
                return "color:#94a3b8"
            except: return ""
        def sty_score_c(v):
            try:
                n = int(v)
                if n >= 80: return "color:#4ade80;font-weight:800"
                if n >= 60: return "color:#facc15;font-weight:700"
                if n >= 40: return "color:#fb923c"
                return "color:#94a3b8"
            except: return ""
        def sty_state(v):
            m = {"inside_fvg":"color:#4ade80;font-weight:800",
                 "inside_ob":"color:#86efac;font-weight:700",
                 "above_ob":"color:#facc15",
                 "below_fvg":"color:#fb923c",
                 "invalidated":"color:#f87171;font-weight:700",
                 "no_ob":"color:#94a3b8"}
            return m.get(str(v), "")
        def sty_valid(v):
            return "color:#4ade80;font-weight:700" if v else "color:#f87171;font-weight:700"
        def sty_htf(v):
            m = {"Bullish":"color:#4ade80","Bearish":"color:#f87171",
                 "Range":"color:#facc15","Unknown":"color:#94a3b8"}
            return m.get(str(v), "")
        def sty_statut(v):
            m = {"Fresh":"color:#4ade80;font-weight:700",
                 "Aged":"color:#fb923c;font-weight:700",
                 "Stale":"color:#f87171"}
            return m.get(str(v), "")
        def sty_session(v):
            if "Overlap" in str(v): return "color:#c084fc;font-weight:700"
            if v in ("London","NewYork"): return "color:#38bdf8"
            return "color:#94a3b8"

        cols_show = [c for c in DISPLAY_COLS if c in df_ui.columns]
        st.dataframe(
            df_ui[cols_show].style
            .map(sty_type,    subset=["Type"])
            .map(sty_dir,     subset=["Direction"])
            .map(sty_score_s, subset=["Sc.Struct"])
            .map(sty_score_c, subset=["Sc.Conf"])
            .map(sty_state,   subset=["Price_State"])
            .map(sty_valid,   subset=["Valid"])
            .map(sty_htf,     subset=["HTF"])
            .map(sty_statut,  subset=["Statut"])
            .map(sty_session, subset=["Session"]),
            hide_index=True,
            use_container_width=True,
            height=min(650, 65 + len(df_ui) * 38),
        )
    else:
        st.info("Aucun signal après filtres. Ajustez les paramètres dans la sidebar.")

    st.divider()

    # ── Note pipeline ─────────────────────────────────────────────────────────
    st.caption(
        "ℹ️ Le JSON pipeline contient **tous les signaux valides Fresh/Aged** "
        "indépendamment des filtres UI. C'est le LLM BLUESTAR DIRECT qui applique "
        "les filtres finaux (GPS/RSI/S/R/Calendrier)."
    )

    # ── Exports ───────────────────────────────────────────────────────────────
    st.markdown("#### Exports")
    e1, e2, e3 = st.columns(3)

    rows_export = [signal_to_row(s) for s in signals
                   if s.is_valid and s.statut in ("Fresh", "Aged")]

    with e1:
        df_csv = pd.DataFrame(rows_export).drop(
            columns=["_time_sort", "_sc_sort"], errors="ignore"
        )
        st.download_button("⬇️ CSV (valides Fresh/Aged)",
                           df_csv.to_csv(index=False).encode(),
                           f"smc_{ts}.csv", "text/csv",
                           use_container_width=True)
    with e2:
        st.download_button("⬇️ PDF",
                           create_pdf(rows_export, scan_time_meta),
                           f"smc_{ts}.pdf", "application/pdf",
                           use_container_width=True)
    with e3:
        # JSON pipeline : TOUS les signaux valides Fresh/Aged sans filtre UI
        pipeline_json = json.dumps({
            "meta": {
                "scanner_version":  SCANNER_VERSION,
                "pipeline_target":  "BLUESTAR_DIRECT_v6.0",
                "generated_at":     scan_time_meta.isoformat(),
                "signal_count":     len(payloads),
                "instruments_scanned": len(INSTRUMENTS),
                "timeframes":       list(TIMEFRAMES.keys()),
                "note": (
                    "Signaux valides Fresh/Aged uniquement. "
                    "structure_score = qualité SMC pure (0-60). "
                    "confluence_score = SMC+HTF+session (0-100). "
                    "price_state indique si le retest est en attente ou actif. "
                    "tp_natural = candidat TP2 (prochain swing opposé). "
                    "atr = ATR du TF du signal pour calcul SL BLUESTAR step 5."
                ),
            },
            "signals": payloads,
        }, ensure_ascii=False, indent=2, default=str).encode("utf-8")

        st.download_button("⬇️ JSON Pipeline BLUESTAR",
                           pipeline_json,
                           f"smc_pipeline_{ts}.json", "application/json",
                           use_container_width=True)

    # ── Aperçu JSON ───────────────────────────────────────────────────────────
    if payloads:
        with st.expander(f"📋 Aperçu JSON Pipeline ({len(payloads)} signaux)"):
            st.json(payloads[0])

    # ── Résumé pipeline ───────────────────────────────────────────────────────
    with st.expander("📊 Résumé pour BLUESTAR DIRECT"):
        bullish_sigs = [s for s in signals if s.direction == "Bullish" and s.is_valid and s.statut in ("Fresh","Aged")]
        bearish_sigs = [s for s in signals if s.direction == "Bearish" and s.is_valid and s.statut in ("Fresh","Aged")]
        mss_sigs     = [s for s in signals if s.sig_type == "MSS" and s.is_valid and s.statut in ("Fresh","Aged")]

        st.markdown(f"""
**Signaux dans le pipeline :** {len(payloads)}
- Bullish : {len(bullish_sigs)} | Bearish : {len(bearish_sigs)}
- MSS (retournements) : {len(mss_sigs)} | BOS (continuations) : {len(payloads) - len(mss_sigs)}
- HTF alignés : {sum(1 for s in signals if s.htf_aligned and s.is_valid and s.statut in ('Fresh','Aged'))}
- Avec OB + FVG : {sum(1 for s in signals if s.ob and s.fvg and s.is_valid and s.statut in ('Fresh','Aged'))}
- Price state `inside_fvg` : {sum(1 for s in signals if s.price_state == 'inside_fvg' and s.is_valid and s.statut in ('Fresh','Aged'))}
- Price state `inside_ob` : {sum(1 for s in signals if s.price_state == 'inside_ob' and s.is_valid and s.statut in ('Fresh','Aged'))}
- Price state `above_ob` (retest attendu) : {sum(1 for s in signals if s.price_state == 'above_ob' and s.is_valid and s.statut in ('Fresh','Aged'))}
        """)

elif "signals" in st.session_state and not st.session_state.signals:
    st.info("Aucun signal SMC détecté sur ce scan.")
