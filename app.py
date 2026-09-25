"""
CHoCH Scanner v5.20 — build de production.

Fichier unique (compatible Streamlit Cloud), organise en couches strictes :
    1. Constantes et registre des regles (figes, versionnes)
    2. Logging JSON structure
    3. Configuration et ressources (credentials, client OANDA, pool)
    4. Couche domaine pure (deterministe, sans effet de bord)
    5. Couche I/O (OANDA : retry, annulation)
    6. Orchestration (scan = fonction pure de ses entrees)
    7. Contrat JSON et exports (JSON / CSV / PDF / PNG)
    8. Couche UI (presentation uniquement)

Invariants :
    - Une seule source de verite par champ de signal (UI == JSON).
    - signal_id deterministe et reproductible.
    - Tout payload emis est auto-coherent : score, distances, session,
      statut, confirmation_time et age_minutes sont RECALCULABLES depuis
      le payload lui-meme et verifies par le validateur (fail-closed).
    - Couverture : somme des compteurs exclusifs == 132, sinon echec bruyant.
    - Aucun etat mutable partage entre sessions Streamlit.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

# pylint: disable=wrong-import-position
import hashlib
import io
import json
import logging
import math
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import (Any, Callable, Final, Literal, Mapping, Optional,
                    Sequence, TypedDict)
from xml.sax.saxutils import escape as _xml_escape
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from matplotlib.figure import Figure
from oandapyV20 import API
from oandapyV20.endpoints import instruments
from oandapyV20.exceptions import V20Error
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

# =====================================================================
# SECTION 1 — CONSTANTES & REGISTRE DES REGLES
# =====================================================================

SCANNER_VERSION: Final[str] = "5.20"
RULE_VERSION: Final[str] = "choch.v58.r11"
SCHEMA_VERSION: Final[str] = "3.1.0"
# CHANGELOG
# r11 (sorties modifiees -> nouveau RULE_VERSION, signal_id tous changes) :
#   R11-1 Fenetre de detection DERIVEE de TF_STATUT (Aged + 2). Avant, la
#         fenetre (H1 5, H4 5, D1 3, W 3) etait plus courte que le seuil
#         Aged : Stale etait inatteignable, Aged quasi inatteignable, et un
#         signal disparaissait sans trace apres 3-5 bougies (defaut 2,
#         NZD/CAD H1). Desormais : Fresh -> Aged -> Stale (1 bougie,
#         visible, non emis) -> sortie de fenetre.
#   R11-2 Stale prioritaire sur Invalidated : un signal expire n'est plus
#         emis, quel que soit l'etat de son niveau.
#   R11-3 distance_atr_multiple arrondi a 2 decimales AVANT filtre et score :
#         le score est recalculable exactement depuis le payload.
#   R11-4 distance_pct / current_distance_pct calcules depuis les prix
#         PUBLIES (arrondis a displayPrecision) : recalcul exact, plus de
#         tolerance.
# schema 3.1.0 (additif) : meta.status_counts. Aucun champ de signal modifie.
#   Validateur : recalcul exact du score, has_sweep interdit pour BOS,
#   confirmation_time == signal_time + 1 barre, statut coherent avec
#   candles_elapsed, age_minutes exact, unicite des signal_id.
# r10 : Invalidated emis (V3), session DailyClose coherente avec le score.
# r9  : tendance/ATR causaux, BOS emis, session DailyClose.

INSTRUMENTS: Final[tuple[str, ...]] = (
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "DE30_EUR", "XAU_USD", "SPX500_USD", "NAS100_USD", "US30_USD",
)

VOLATILITY_STATIC: Final[Mapping[str, str]] = {
    "EUR_USD": "Basse", "GBP_USD": "Basse", "USD_JPY": "Basse",
    "USD_CHF": "Basse", "USD_CAD": "Basse", "AUD_USD": "Moyenne",
    "NZD_USD": "Moyenne", "EUR_GBP": "Moyenne", "EUR_JPY": "Moyenne",
    "EUR_CHF": "Moyenne", "EUR_AUD": "Moyenne", "EUR_CAD": "Moyenne",
    "EUR_NZD": "Moyenne", "GBP_JPY": "Haute", "GBP_CHF": "Haute",
    "GBP_AUD": "Haute", "GBP_CAD": "Haute", "GBP_NZD": "Haute",
    "AUD_JPY": "Haute", "AUD_CAD": "Moyenne", "AUD_CHF": "Haute",
    "AUD_NZD": "Moyenne", "CAD_JPY": "Haute", "CAD_CHF": "Haute",
    "CHF_JPY": "Haute", "NZD_JPY": "Haute", "NZD_CAD": "Moyenne",
    "NZD_CHF": "Haute", "DE30_EUR": "Très Haute", "XAU_USD": "Très Haute",
    "SPX500_USD": "Très Haute", "NAS100_USD": "Très Haute",
    "US30_USD": "Très Haute",
}

# Nom de timeframe -> granularite OANDA
TIMEFRAMES: Final[Mapping[str, str]] = {
    "H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W",
}
BAR_SECONDS: Final[Mapping[str, int]] = {
    "H1": 3600, "H4": 14400, "D1": 86400, "Weekly": 604800,
}
SWING_LOOKBACK: Final[Mapping[str, int]] = {
    "H1": 5, "H4": 5, "D1": 4, "Weekly": 3,
}
SWING_HISTORY: Final[Mapping[str, int]] = {
    "H1": 120, "H4": 90, "D1": 60, "Weekly": 26,
}
GRAN_COUNT: Final[Mapping[str, int]] = {
    "H1": 400, "H4": 300, "D": 200, "W": 120,
}
MIN_CANDLES: Final[int] = 50

TF_STATUT: Final[Mapping[str, Mapping[str, int]]] = {
    "H1":     {"Fresh": 4, "Aged": 12},
    "H4":     {"Fresh": 3, "Aged": 8},
    "D1":     {"Fresh": 2, "Aged": 5},
    "Weekly": {"Fresh": 2, "Aged": 4},
}
# R11-1 : offsets 0..Aged+1 -> Fresh, Aged et Stale (Aged+1) atteignables.
DETECTION_LOOKBACK: Final[Mapping[str, int]] = {
    tf: thr["Aged"] + 2 for tf, thr in TF_STATUT.items()
}
EMITTED_STATUSES: Final[tuple[str, ...]] = ("Fresh", "Aged", "Invalidated")

# Bareme de confluence (max 25+10+20+15+15 = 85)
SCORE_BASE: Final[int] = 25
SCORE_TYPE_BONUS: Final[int] = 10          # CHoCH et BOS (D7-a)
SCORE_DIST_BONUS: Final[int] = 15
SCORE_DIST_BONUS_MAX_ATR: Final[float] = 1.0
SCORE_SWEEP_BONUS: Final[int] = 15         # CHoCH uniquement
SESSION_BONUS: Final[Mapping[str, int]] = {
    "London_NY_Overlap": 20, "London": 20, "NewYork": 20,
    "DailyClose": 10, "Tokyo": 0, "Off": 0,
}
MIN_SCORE: Final[int] = 65
ATR_DIST_MULT: Final[float] = 1.8

SCAN_GLOBAL_TIMEOUT: Final[int] = 180
SCAN_MAX_WORKERS: Final[int] = 6
OANDA_REQUEST_TIMEOUT: Final[int] = 12
CANDLES_CACHE_TTL_SECONDS: Final[int] = 60
MAX_AUTH_FAILURES: Final[int] = 3
OANDA_MAX_RETRIES: Final[int] = 2
OANDA_BACKOFF_BASE: Final[float] = 0.25
PNG_ROWS_PER_PAGE: Final[int] = 40

DISPLAY_COLS: Final[tuple[str, ...]] = (
    "Instrument", "Timeframe", "Type", "Ordre", "Signal",
    "Niveau", "Distance%", "Distance actuelle %", "Score",
    "Volatilité", "Force", "BB_Width", "Statut",
    "Ouverture bougie (UTC)", "Confirmation (UTC)", "signal_id",
)
_STATUS_RANK: Final[Mapping[str, int]] = {
    "Fresh": 0, "Aged": 1, "Stale": 2, "Invalidated": 3,
}

TrendT = Literal["Bullish", "Bearish", "Range"]
DirectionT = Literal["Bullish", "Bearish"]
SigTypeT = Literal["CHoCH", "BOS"]
StatusT = Literal["Fresh", "Aged", "Stale", "Invalidated"]
SessionT = Literal[
    "London_NY_Overlap", "London", "NewYork", "Tokyo", "Off", "DailyClose",
]
OutcomeT = Literal["no_signal", "no_data", "failed", "aborted"]


# =====================================================================
# SECTION 2 — LOGGING JSON STRUCTURE
# =====================================================================

class _JsonFormatter(logging.Formatter):
    """Formatter JSON sans dependance ; ne fait jamais crasher le process."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(
                record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                payload[key[4:]] = value
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return json.dumps({"ts": payload["ts"], "level": "ERROR",
                               "msg": "log_serialization_failure"})


def _configure_root_logger() -> logging.Logger:
    root = logging.getLogger("choch")
    if getattr(root, "_choch_configured", False):
        return root
    level_name = os.environ.get("CHOCH_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        print(f"CHOCH_LOG_LEVEL inconnu : '{level_name}' — repli sur INFO",
              file=sys.stderr)
        level = logging.INFO
    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False
    setattr(root, "_choch_configured", True)
    return root


logger = _configure_root_logger()


def _log(level: int, msg: str, **ctx: Any) -> None:
    logger.log(level, msg, extra={f"ctx_{k}": v for k, v in ctx.items()})


# =====================================================================
# SECTION 3 — CONFIGURATION & RESSOURCES
# =====================================================================

class ConfigError(Exception):
    """Configuration invalide (token, environnement)."""


@dataclass(frozen=True)
class OandaCredentials:
    token: str = field(repr=False)
    env: Literal["practice", "live"]
    source: str


def _secret(key: str) -> Optional[str]:
    """Lecture tolerante de st.secrets (absence de secrets.toml != crash)."""
    try:
        value = st.secrets.get(key)
    except Exception:  # noqa: BLE001 — fichier de secrets absent/illisible
        return None
    return None if value is None else str(value)


def resolve_credentials() -> OandaCredentials:
    """Resolu UNE fois par scan dans le thread principal (jamais en worker)."""
    env = (os.environ.get("OANDA_ENV", "").strip().lower()
           or (_secret("OANDA_ENV") or "practice").strip().lower())
    if env not in ("practice", "live"):
        raise ConfigError(
            f"OANDA_ENV invalide : {env!r} (attendu practice|live)")
    test_token = os.environ.get("CHOCH_TEST_TOKEN")
    if test_token:
        if env == "live":
            raise ConfigError(
                "CHOCH_TEST_TOKEN refuse en OANDA_ENV=live : un token de "
                "test ne doit jamais atteindre le compte reel")
        _log(logging.WARNING, "test_token_actif",
             detail="CHOCH_TEST_TOKEN remplace OANDA_ACCESS_TOKEN")
        return OandaCredentials(test_token, env, "CHOCH_TEST_TOKEN")
    token = _secret("OANDA_ACCESS_TOKEN")
    if not token:
        raise ConfigError("Clé API OANDA manquante (OANDA_ACCESS_TOKEN).")
    return OandaCredentials(token, env, "OANDA_ACCESS_TOKEN")


@lru_cache(maxsize=8)
def _tz(name: str) -> ZoneInfo:
    return ZoneInfo(name)


_thread_local = threading.local()


def _get_oanda_api(creds: OandaCredentials) -> API:
    """Un client par thread ET par credentials : requests.Session n'est pas
    garanti thread-safe, et un changement de token ne reutilise jamais un
    client construit avec l'ancien."""
    clients: Optional[dict[OandaCredentials, API]] = getattr(
        _thread_local, "clients", None)
    if clients is None:
        clients = {}
        _thread_local.clients = clients
    api = clients.get(creds)
    if api is None:
        api = API(access_token=creds.token, environment=creds.env,
                  request_params={"timeout": OANDA_REQUEST_TIMEOUT})
        clients[creds] = api
        _log(logging.INFO, "oanda_api_init", env=creds.env,
             source=creds.source)
    return api


@st.cache_resource(show_spinner=False)
def _get_scan_executor() -> ThreadPoolExecutor:
    """Pool unique par process. Limite connue : apres un timeout global, les
    requetes HTTP deja parties occupent un worker jusqu'a
    OANDA_REQUEST_TIMEOUT."""
    return ThreadPoolExecutor(max_workers=SCAN_MAX_WORKERS,
                              thread_name_prefix="choch-scan")


# =====================================================================
# SECTION 4 — COUCHE DOMAINE PURE
# =====================================================================

class SwingDict(TypedDict):
    idx: int
    price: float
    kind: Literal["HH", "LH", "HL", "LL"]


@dataclass(frozen=True)
class SignalCore:
    """Source de verite unique d'un signal (UI et JSON en derivent)."""
    sig_type: SigTypeT
    direction: DirectionT
    level: float
    close_price: float
    current_price: float
    has_sweep: bool
    atr_val: float
    volatilite: str
    trend: TrendT
    force: Literal["Fort", "Moyen"]
    dist_atr: float            # arrondi a 2 decimales (R11-3)
    score: int
    bb_width_pct: Optional[float]
    bb_regime: str
    signal_time_utc: datetime  # OUVERTURE de la bougie de cassure
    session: SessionT
    statut: StatusT
    candles_elapsed: int


# ---- 4.1 utilitaires numeriques -------------------------------------------

def _compute_true_range_vec(data: pd.DataFrame) -> np.ndarray:
    """True Range vectorise ; (high-low) seul a travers les gaps anormaux
    (> 2 x delta median), pour ne pas gonfler l'ATR sur les week-ends."""
    if len(data) < 2:
        return np.empty(0, dtype=np.float64)
    high = data["high"].to_numpy(dtype=np.float64)
    low = data["low"].to_numpy(dtype=np.float64)
    close = data["close"].to_numpy(dtype=np.float64)
    deltas = np.diff(
        data.index.values.astype("datetime64[s]").astype(np.int64))
    typical = max(float(np.median(deltas)), 1.0)
    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    tr = np.maximum(hl, np.maximum(hc, lc))
    return np.where(deltas > 2 * typical, hl, tr)


def calc_atr_bundle(data: pd.DataFrame, inst: str,
                    period: int = 14) -> tuple[float, str]:
    """(atr, regime). EWM alpha=1/14 sur les 100 derniers TR (~99,9 % de
    convergence apres ~66 periodes)."""
    fallback = VOLATILITY_STATIC[inst]
    tr = _compute_true_range_vec(data)
    if tr.size < period * 3:
        return float("nan"), fallback
    window = tr[-100:]
    atr_val = float(pd.Series(window).ewm(
        alpha=1.0 / period, adjust=False).mean().iloc[-1])
    if not math.isfinite(atr_val):
        return float("nan"), fallback
    median_tr = float(np.median(window))
    if not math.isfinite(median_tr) or median_tr < 1e-10:
        return atr_val, fallback
    ratio = atr_val / median_tr
    if ratio >= 1.8:
        return atr_val, "Très Haute"
    if ratio >= 1.2:
        return atr_val, "Haute"
    if ratio >= 0.7:
        return atr_val, "Moyenne"
    return atr_val, "Basse"


# displayPrecision OANDA figee (AccountInstruments, compte practice,
# releve du 2026-09-25) ; regles JPY=3 / autres=5 pour le reste.
_PRECISION_OANDA: Final[Mapping[str, int]] = {
    "DE30_EUR": 1, "XAU_USD": 3, "SPX500_USD": 1,
    "NAS100_USD": 1, "US30_USD": 1,
}


def instrument_precision(inst: str) -> int:
    if inst in _PRECISION_OANDA:
        return _PRECISION_OANDA[inst]
    return 3 if "JPY" in inst else 5


def calc_distance_pct(niveau: float, prix: float) -> Optional[float]:
    if not (math.isfinite(niveau) and math.isfinite(prix)):
        return None
    if abs(niveau) < 1e-12:
        return None
    dist = abs(prix - niveau) / abs(niveau) * 100.0
    return dist if dist <= 100.0 else None


def _round4(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v, 4)


def format_distance(dist_pct: Optional[float]) -> str:
    return "N/A" if dist_pct is None else f"{dist_pct:.3f}%"


def get_session(dt: datetime, tf: str) -> SessionT:
    """Session de la bougie (heure d'OUVERTURE). D1/Weekly -> DailyClose."""
    if tf in ("D1", "Weekly"):
        return "DailyClose"
    london_h = dt.astimezone(_tz("Europe/London")).hour
    ny_h = dt.astimezone(_tz("America/New_York")).hour
    tokyo_h = dt.astimezone(_tz("Asia/Tokyo")).hour
    london = 8 <= london_h < 17
    ny = 9 <= ny_h < 17
    if london and ny:
        return "London_NY_Overlap"
    if london:
        return "London"
    if ny:
        return "NewYork"
    if 9 <= tokyo_h < 18:
        return "Tokyo"
    return "Off"


def compute_confluence_score(dist_atr: float, session: str,
                             has_sweep: bool) -> int:
    """Bareme unique, utilise par le domaine ET par le validateur."""
    score = SCORE_BASE + SCORE_TYPE_BONUS + SESSION_BONUS[session]
    if dist_atr <= SCORE_DIST_BONUS_MAX_ATR:
        score += SCORE_DIST_BONUS
    if has_sweep:
        score += SCORE_SWEEP_BONUS
    return score


def status_for_elapsed(candles_elapsed: int, tf: str) -> StatusT:
    thr = TF_STATUT[tf]
    if candles_elapsed <= thr["Fresh"]:
        return "Fresh"
    if candles_elapsed <= thr["Aged"]:
        return "Aged"
    return "Stale"


def confirmation_time(signal_time: datetime, tf: str) -> datetime:
    """Cloture de la bougie de cassure = ouverture + 1 barre."""
    return signal_time + timedelta(seconds=BAR_SECONDS[tf])


def age_minutes(signal_time: datetime, tf: str, scan_time: datetime) -> int:
    delta = (scan_time - confirmation_time(signal_time, tf)).total_seconds()
    return int(max(0.0, delta) // 60)


# ---- 4.2 parsing des bougies ----------------------------------------------

def _parse_candle_row(c: Mapping[str, Any], inst: str,
                      gran: str) -> Optional[dict[str, Any]]:
    try:
        mid = c["mid"]
        open_v = float(mid["o"])
        high_v = float(mid["h"])
        low_v = float(mid["l"])
        close_v = float(mid["c"])
        t = c["time"]
    except (KeyError, ValueError, TypeError) as exc:
        _log(logging.WARNING, "candle_malformed", instrument=inst,
             granularity=gran, err=str(exc))
        return None
    if not all(math.isfinite(v) for v in (open_v, high_v, low_v, close_v)):
        return None
    if not (high_v >= low_v and high_v >= max(open_v, close_v)
            and low_v <= min(open_v, close_v)):
        _log(logging.WARNING, "candle_inconsistent", instrument=inst,
             granularity=gran, t=str(t))
        return None
    return {"time": pd.to_datetime(t, utc=True), "open": open_v,
            "high": high_v, "low": low_v, "close": close_v}


# ---- 4.3 pivots & tendance -------------------------------------------------

def _classify_swings(
    pivots: Sequence[tuple[int, float, str]],
) -> list[SwingDict]:
    swings: list[SwingDict] = []
    prev_h: Optional[float] = None
    prev_l: Optional[float] = None
    for idx, price, k in pivots:
        if k == "H":
            kind: Literal["HH", "LH", "HL", "LL"] = (
                "HH" if prev_h is None or price > prev_h else "LH")
            prev_h = price
        else:
            kind = "HL" if prev_l is None or price > prev_l else "LL"
            prev_l = price
        swings.append({"idx": idx, "price": price, "kind": kind})
    return swings


def detect_swing_points(data: pd.DataFrame, tf: str) -> list[SwingDict]:
    """Pivots sur fenetre centree ; dedup par INDEX (jamais par prix).

    Residu documente (C-5) : ``start`` est ancre sur n, pas sur l'idx du
    signal ; les etiquettes HH/HL/LL de pivots anciens peuvent differer
    entre un prefixe et la serie complete (mesure LOT B : 0 signal emis
    modifie). Non corrige volontairement (cout x40).
    """
    lookback = SWING_LOOKBACK[tf]
    history = SWING_HISTORY[tf]
    n = len(data)
    if n < 2 * lookback + 1:
        return []
    win = 2 * lookback + 1
    high_s = data["high"].reset_index(drop=True)
    low_s = data["low"].reset_index(drop=True)
    h_mask = (high_s == high_s.rolling(win, center=True,
                                       min_periods=win).max()).to_numpy()
    l_mask = (low_s == low_s.rolling(win, center=True,
                                     min_periods=win).min()).to_numpy()
    high_arr = high_s.to_numpy()
    low_arr = low_s.to_numpy()
    pivots: list[tuple[int, float, str]] = []
    # dernier pivot confirmable : i = n-1-lookback (range exclut end)
    for i in range(max(lookback, n - history - lookback), n - lookback):
        if h_mask[i]:
            pivots.append((i, float(high_arr[i]), "H"))
        elif l_mask[i]:
            # un index a la fois max ET min (plat degenere) -> high seul
            pivots.append((i, float(low_arr[i]), "L"))
    return _classify_swings(pivots)


def get_structural_trend(swings: Sequence[SwingDict]) -> TrendT:
    if len(swings) < 4:
        return "Range"
    recent = swings[-6:]
    highs = [s["kind"] for s in recent if s["kind"] in ("HH", "LH")]
    lows = [s["kind"] for s in recent if s["kind"] in ("HL", "LL")]
    if not highs or not lows:
        return "Range"
    if highs[-1] == "HH" and lows[-1] == "HL":
        return "Bullish"
    if highs[-1] == "LH" and lows[-1] == "LL":
        return "Bearish"
    return "Range"


# ---- 4.4 resolution du signal ---------------------------------------------

_SigResult = tuple[Optional[SigTypeT], Optional[DirectionT], Optional[float]]
_NONE_SIG: Final[_SigResult] = (None, None, None)


def _last_of_kind(swings: Sequence[SwingDict],
                  kind: str) -> Optional[SwingDict]:
    """Dernier pivot du type demande.

    NB (B4, OUVERT) : le constat B4 — un pivot plus recent d'un AUTRE type
    existe apres le pivot choisi — n'est PAS traite par cette fonction. La
    selection est inchangee depuis r8 ; la corriger est une decision de
    regle en attente.
    """
    for s in reversed(swings):
        if s["kind"] == kind:
            return s
    return None


def _resolve_signal(trend: TrendT, close_arr: np.ndarray, idx: int,
                    swings: Sequence[SwingDict]) -> _SigResult:
    c0, c1 = close_arr[idx], close_arr[idx - 1]
    if trend == "Bullish":
        hl = _last_of_kind(swings, "HL")
        if hl is not None and c0 < hl["price"] <= c1:
            return "CHoCH", "Bearish", hl["price"]
        hh = _last_of_kind(swings, "HH")
        if hh is not None and c1 <= hh["price"] < c0:
            return "BOS", "Bullish", hh["price"]
    elif trend == "Bearish":
        lh = _last_of_kind(swings, "LH")
        if lh is not None and c1 <= lh["price"] < c0:
            return "CHoCH", "Bullish", lh["price"]
        ll = _last_of_kind(swings, "LL")
        if ll is not None and c0 < ll["price"] <= c1:
            return "BOS", "Bearish", ll["price"]
    return _NONE_SIG


def _detect_liquidity_sweep(high_arr: np.ndarray, low_arr: np.ndarray,
                            idx: int, swings: Sequence[SwingDict],
                            atr_val: float, direction: DirectionT) -> bool:
    """Seuls les 3 pivots les plus recents constituent des pools actifs."""
    margin = atr_val * 0.25
    if direction == "Bearish":
        pools = [s for s in swings if s["kind"] in ("HH", "LH")][-3:]
        return any(high_arr[idx] - s["price"] > margin for s in pools)
    pools = [s for s in swings if s["kind"] in ("HL", "LL")][-3:]
    return any(s["price"] - low_arr[idx] > margin for s in pools)


# ---- 4.5 largeur des Bollinger --------------------------------------------

def compute_bb_width(data: pd.DataFrame, length: int = 20,
                     std: int = 2) -> tuple[Optional[float], str]:
    close = data["close"]
    if len(close) < length * 2:
        return None, "N/A"
    sma = close.rolling(length).mean()
    std_dev = close.rolling(length).std()
    bb_w = (2 * std * std_dev) / sma.where(sma.abs() > 1e-12)
    bb_avg = bb_w.rolling(length).mean()
    pct_val = ((bb_w - bb_avg) / bb_avg.where(bb_avg.abs() > 1e-12)
               * 100.0).iloc[-1]
    if pd.isna(pct_val) or not math.isfinite(pct_val):
        return None, "N/A"
    if pct_val <= -25:
        return float(pct_val), "Squeeze"
    if pct_val >= 25:
        return float(pct_val), "Expansion"
    return float(pct_val), "Normal"


def format_bb_width(pct: Optional[float], regime: str) -> str:
    if pct is None:
        return "N/A"
    return f"{'+' if pct >= 0 else ''}{pct:.0f}%_{regime}"


# ---- 4.6 construction du SignalCore ---------------------------------------

@dataclass(frozen=True)
class _Ohlc:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


def _evaluate_candle(*, idx: int, df: pd.DataFrame, ohlc: _Ohlc,
                     swings: Sequence[SwingDict], atr_val: float,
                     atr_regime: str, trend: TrendT,
                     tf: str) -> Optional[SignalCore]:
    sig_type, direction, level = _resolve_signal(trend, ohlc.close, idx,
                                                 swings)
    if sig_type is None or direction is None or level is None:
        return None

    rng_v = ohlc.high[idx] - ohlc.low[idx]
    if rng_v <= 0:
        return None
    body_ratio = abs(ohlc.close[idx] - ohlc.open[idx]) / rng_v
    if body_ratio < 0.40:
        return None
    force: Literal["Fort", "Moyen"] = "Fort" if body_ratio >= 0.60 else "Moyen"

    # R11-3 : arrondi AVANT filtre et score -> recalculable depuis le payload
    dist_atr = round(abs(ohlc.close[idx] - level) / atr_val, 2)
    if dist_atr > ATR_DIST_MULT:
        return None

    candle_time = df.index[idx].to_pydatetime()
    session = get_session(candle_time, tf)
    has_sweep = sig_type == "CHoCH" and _detect_liquidity_sweep(
        ohlc.high, ohlc.low, idx, swings, atr_val, direction)
    score = compute_confluence_score(dist_atr, session, has_sweep)
    if score < MIN_SCORE:
        return None

    n = ohlc.close.size
    elapsed = (n - 1) - idx
    # Invalidation : niveau franchi APRES la confirmation (bougies idx+1..).
    #  Bullish : ancienne resistance cassee ; echec si low < level.
    #  Bearish : ancien support casse ; echec si high > level.
    if direction == "Bullish":
        invalidated = bool((ohlc.low[idx + 1:] < level).any())
    else:
        invalidated = bool((ohlc.high[idx + 1:] > level).any())
    base_status = status_for_elapsed(elapsed, tf)
    # R11-2 : Stale prioritaire (un signal expire n'est jamais emis)
    statut: StatusT = ("Invalidated"
                       if invalidated and base_status != "Stale"
                       else base_status)

    bb_pct, bb_regime = compute_bb_width(
        df.iloc[max(0, idx + 1 - 40): idx + 1])

    return SignalCore(
        sig_type=sig_type, direction=direction, level=float(level),
        close_price=float(ohlc.close[idx]),
        current_price=float(ohlc.close[-1]),
        has_sweep=bool(has_sweep), atr_val=float(atr_val),
        volatilite=atr_regime, trend=trend, force=force,
        dist_atr=float(dist_atr), score=int(score),
        bb_width_pct=bb_pct, bb_regime=bb_regime,
        signal_time_utc=candle_time, session=session, statut=statut,
        candles_elapsed=int(elapsed),
    )


def detect_choch(df: pd.DataFrame, tf: str, inst: str) -> Optional[SignalCore]:
    """Detection causale par offset : pour chaque idx, pivots, tendance et
    ATR ne voient que les donnees confirmees a idx. Le premier match (le
    plus recent) gagne."""
    swings = detect_swing_points(df, tf)
    if not swings:
        return None
    n = len(df)
    lookback = SWING_LOOKBACK[tf]
    ohlc = _Ohlc(
        open=df["open"].to_numpy(dtype=np.float64),
        high=df["high"].to_numpy(dtype=np.float64),
        low=df["low"].to_numpy(dtype=np.float64),
        close=df["close"].to_numpy(dtype=np.float64),
    )
    for offset in range(DETECTION_LOOKBACK[tf]):
        idx = n - 1 - offset
        if idx < 1:
            break
        # pivot i confirme a la bougie i+lookback, qui doit etre <= idx-1
        prev_swings = [s for s in swings if s["idx"] <= idx - (lookback + 1)]
        trend = get_structural_trend(prev_swings)
        if trend == "Range":
            continue
        atr_val, atr_regime = calc_atr_bundle(df.iloc[: idx + 1], inst)
        if not (math.isfinite(atr_val) and atr_val > 0):
            continue
        sig = _evaluate_candle(idx=idx, df=df, ohlc=ohlc,
                               swings=prev_swings, atr_val=atr_val,
                               atr_regime=atr_regime, trend=trend, tf=tf)
        if sig is not None:
            return sig
    return None


# ---- 4.7 projections (ligne UI / payload JSON) ----------------------------

def _signal_id(inst: str, tf: str, sig: SignalCore) -> str:
    stamp = sig.signal_time_utc.strftime("%Y%m%dT%H%M")
    raw = (f"{inst}|{tf}|{stamp}Z|{sig.sig_type}|{sig.direction}"
           f"|{RULE_VERSION}")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{inst}__{tf}__{stamp}__{digest}"


@dataclass(frozen=True)
class _Published:
    """Prix publies (arrondis a displayPrecision) et distances derivees."""
    prec: int
    level: float
    close_price: float
    current_price: float
    distance_pct: Optional[float]
    current_distance_pct: Optional[float]


def _published(inst: str, sig: SignalCore) -> _Published:
    prec = instrument_precision(inst)
    level = round(sig.level, prec)
    close_p = round(sig.close_price, prec)
    current_p = round(sig.current_price, prec)
    return _Published(
        prec=prec, level=level, close_price=close_p, current_price=current_p,
        distance_pct=_round4(calc_distance_pct(level, close_p)),
        current_distance_pct=_round4(calc_distance_pct(level, current_p)),
    )


def signal_to_row(inst: str, tf: str, sig: SignalCore) -> dict[str, Any]:
    pub = _published(inst, sig)
    return {
        "Instrument": inst.replace("_", "/"),
        "Timeframe": tf,
        "Type": sig.sig_type,
        "Ordre": "Achat" if sig.direction == "Bullish" else "Vente",
        "Signal": f"{sig.direction} {sig.sig_type}",
        "Niveau": f"{pub.level:.{pub.prec}f}",
        "Distance%": format_distance(pub.distance_pct),
        "Distance actuelle %": format_distance(pub.current_distance_pct),
        "Score": int(sig.score),
        "Volatilité": sig.volatilite,
        "Force": sig.force,
        "BB_Width": format_bb_width(sig.bb_width_pct, sig.bb_regime),
        "Statut": sig.statut,
        "Ouverture bougie (UTC)": sig.signal_time_utc.strftime(
            "%Y-%m-%d %H:%M"),
        "Confirmation (UTC)": confirmation_time(
            sig.signal_time_utc, tf).strftime("%Y-%m-%d %H:%M"),
        "signal_id": _signal_id(inst, tf, sig),
        "_time_sort": sig.signal_time_utc,
        "_status_rank": _STATUS_RANK[sig.statut],
    }


def signal_to_payload(inst: str, tf: str, sig: SignalCore,
                      scan_time: datetime) -> dict[str, Any]:
    pub = _published(inst, sig)
    return {
        "signal_id": _signal_id(inst, tf, sig),
        "scanner_version": SCANNER_VERSION,
        "rule_version": RULE_VERSION,
        "generated_at": scan_time.isoformat(),
        "pair": inst.replace("_", "/"),
        "pair_oanda": inst,
        "timeframe": tf,
        "type": sig.sig_type,
        "direction": sig.direction,
        "is_bullish": sig.direction == "Bullish",
        "order": "buy" if sig.direction == "Bullish" else "sell",
        "trend": sig.trend,
        "is_choch": sig.sig_type == "CHoCH",
        "status": sig.statut,
        "confluence_score": int(sig.score),
        "level": pub.level,
        "close_price": pub.close_price,
        "current_price": pub.current_price,
        "distance_pct": pub.distance_pct,
        "current_distance_pct": pub.current_distance_pct,
        "distance_atr_multiple": sig.dist_atr,
        "volatility": sig.volatilite,
        "force": sig.force,
        "bb_width_pct": (None if sig.bb_width_pct is None
                         else round(sig.bb_width_pct, 2)),
        "bb_regime": sig.bb_regime,
        "session": sig.session,
        "signal_time": sig.signal_time_utc.isoformat(),
        "candles_elapsed": int(sig.candles_elapsed),
        "has_sweep": bool(sig.has_sweep),
        "atr": round(sig.atr_val, pub.prec + 2),
        "confirmation_time": confirmation_time(
            sig.signal_time_utc, tf).isoformat(),
        "age_minutes": age_minutes(sig.signal_time_utc, tf, scan_time),
    }


# =====================================================================
# SECTION 5 — COUCHE I/O (OANDA)
# =====================================================================

class InsufficientDataError(Exception):
    """Reponse OANDA valide mais < MIN_CANDLES bougies exploitables
    (comptee en no_data, jamais en failed)."""


class ScanCancelled(Exception):
    """Annulation demandee (timeout global ou abandon auth) avant retry."""


def _fetch_candles_raw(inst: str, gran: str,
                       creds: OandaCredentials) -> list[dict[str, Any]]:
    req = instruments.InstrumentsCandles(
        instrument=inst,
        params={"count": GRAN_COUNT[gran], "granularity": gran, "price": "M"},
    )
    _get_oanda_api(creds).request(req)
    return [c for c in req.response.get("candles", []) if c.get("complete")]


def _fetch_with_retry(inst: str, gran: str, creds: OandaCredentials,
                      cancel_event: threading.Event) -> list[dict[str, Any]]:
    """Retry sur 429, 5xx et erreurs reseau ; 401 et 4xx remontent tout de
    suite. L'annulation est testee avant chaque backoff."""
    attempt = 0
    while True:
        try:
            return _fetch_candles_raw(inst, gran, creds)
        except V20Error as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt >= OANDA_MAX_RETRIES:
                if exc.code != 401:
                    _log(logging.WARNING, "oanda_v20_error", instrument=inst,
                         granularity=gran, code=exc.code, err=str(exc))
                raise
        except requests.RequestException as exc:
            if attempt >= OANDA_MAX_RETRIES:
                _log(logging.WARNING, "oanda_network_error", instrument=inst,
                     granularity=gran, err=str(exc))
                raise
        if cancel_event.is_set():
            raise ScanCancelled(f"annulation demandee pour {inst} {gran}")
        time.sleep(OANDA_BACKOFF_BASE * (2 ** attempt))
        attempt += 1


@st.cache_data(ttl=CANDLES_CACHE_TTL_SECONDS, show_spinner=False,
               max_entries=512)
def get_candles_cached(inst: str, gran: str, env: str, cache_bust: int,
                       _creds: OandaCredentials,
                       _cancel_event: threading.Event) -> pd.DataFrame:
    """Cle de cache : (inst, gran, env, cache_bust). Les arguments prefixes
    par _ ne sont pas hashes (credentials, Event). Ne renvoie jamais None :
    toute panne ou reponse insuffisante LEVE (les exceptions ne sont pas
    mises en cache)."""
    raw = _fetch_with_retry(inst, gran, _creds, _cancel_event)
    rows = [r for c in raw
            if (r := _parse_candle_row(c, inst, gran)) is not None]
    if len(rows) < MIN_CANDLES:
        raise InsufficientDataError(
            f"{inst} {gran} : {len(rows)} bougies exploitables "
            f"(< {MIN_CANDLES}, {len(raw)} recues)")
    df = pd.DataFrame(rows).set_index("time").sort_index()
    return df[~df.index.duplicated(keep="last")]


# =====================================================================
# SECTION 6 — ORCHESTRATION
# =====================================================================

class ScanControl:
    """Etat d'annulation PROPRE A UN SCAN (jamais partage entre scans :
    des workers attardes d'un scan expire ne voient pas le scan suivant)."""

    def __init__(self) -> None:
        self.cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._auth_failures = 0
        self._auth_aborted = False

    def record_auth_failure(self) -> int:
        with self._lock:
            self._auth_failures += 1
            if self._auth_failures >= MAX_AUTH_FAILURES:
                self._auth_aborted = True
                self.cancel_event.set()
            return self._auth_failures

    @property
    def auth_aborted(self) -> bool:
        with self._lock:
            return self._auth_aborted

    def cancel(self) -> None:
        self.cancel_event.set()

    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()


@dataclass(frozen=True)
class UnitOutcome:
    inst: str
    tf: str
    sig: Optional[SignalCore] = None
    kind: OutcomeT = "no_signal"    # ignore si sig is not None
    detail: str = ""


@dataclass
class ScanResult:
    rows: list[dict[str, Any]]
    payloads: list[dict[str, Any]]
    errors: list[str]
    scan_time: datetime
    auth_aborted: bool
    # Compteurs EXCLUSIFS ; somme == 132 (verifiee par serialize_pipeline)
    coverage_counts: dict[str, int]


def _scan_one_inner(inst: str, tf: str, gran: str, cache_bust: int,
                    creds: OandaCredentials,
                    control: ScanControl) -> UnitOutcome:
    if control.is_cancelled():
        return UnitOutcome(inst, tf, kind="aborted")
    try:
        df = get_candles_cached(inst, gran, creds.env, cache_bust,
                                _creds=creds,
                                _cancel_event=control.cancel_event)
    except V20Error as exc:
        if exc.code == 401:
            n = control.record_auth_failure()
            _log(logging.ERROR, "oanda_auth_failure", instrument=inst,
                 granularity=tf, count=n)
            return UnitOutcome(inst, tf, kind="aborted",
                               detail=f"auth 401 #{n}")
        return UnitOutcome(inst, tf, kind="failed",
                           detail=f"failed:v20:{exc.code}")
    except ScanCancelled:
        return UnitOutcome(inst, tf, kind="aborted")
    except requests.RequestException as exc:
        _log(logging.ERROR, "oanda_network_failure", instrument=inst,
             granularity=tf, err=str(exc))
        return UnitOutcome(inst, tf, kind="failed",
                           detail=f"failed:net:{type(exc).__name__}")
    except InsufficientDataError as exc:
        _log(logging.WARNING, "oanda_insufficient_data", instrument=inst,
             granularity=tf, err=str(exc))
        return UnitOutcome(inst, tf, kind="no_data", detail="no_data")
    if control.is_cancelled():
        return UnitOutcome(inst, tf, kind="aborted")
    sig = detect_choch(df, tf, inst)
    return UnitOutcome(inst, tf, sig=sig)


def _scan_one(inst: str, tf: str, gran: str, cache_bust: int,
              creds: OandaCredentials, control: ScanControl) -> UnitOutcome:
    """Frontiere du worker : aucune exception ne remonte a fut.result()."""
    try:
        return _scan_one_inner(inst, tf, gran, cache_bust, creds, control)
    except Exception as exc:  # noqa: BLE001 — frontiere defensive
        _log(logging.ERROR, "scan_one_unexpected", instrument=inst,
             granularity=tf, err=repr(exc))
        return UnitOutcome(inst, tf, kind="failed",
                           detail=f"failed:unexpected:{type(exc).__name__}")


def run_scan(creds: OandaCredentials, cache_bust: int,
             progress_callback: Optional[Callable[[str, str], None]] = None,
             ) -> ScanResult:
    correlation_id = uuid.uuid4().hex[:12]
    scan_time = datetime.now(timezone.utc)
    control = ScanControl()
    t0 = time.perf_counter()
    _log(logging.INFO, "scan_start", correlation_id=correlation_id,
         instruments=len(INSTRUMENTS), timeframes=len(TIMEFRAMES))

    executor = _get_scan_executor()
    futures: dict[Future[UnitOutcome], tuple[str, str]] = {
        executor.submit(_scan_one, inst, tf, gran, cache_bust, creds,
                        control): (inst, tf)
        for inst in INSTRUMENTS
        for tf, gran in TIMEFRAMES.items()
    }

    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    errors: list[str] = []
    counts: dict[str, int] = {
        "ok_signal": 0, "ok_no_signal": 0, "ok_not_emitted": 0,
        "invalid_contract": 0, "no_data": 0, "failed": 0, "aborted": 0,
        "timed_out": 0,
    }
    handled: set[Future[UnitOutcome]] = set()

    def _handle(fut: Future[UnitOutcome]) -> None:
        handled.add(fut)
        o = fut.result()
        try:
            if o.sig is not None:
                rows.append(signal_to_row(o.inst, o.tf, o.sig))
                if o.sig.statut in EMITTED_STATUSES:
                    payloads.append(
                        signal_to_payload(o.inst, o.tf, o.sig, scan_time))
                    counts["ok_signal"] += 1
                else:
                    counts["ok_not_emitted"] += 1  # Stale : visible, non emis
            elif o.kind == "no_signal":
                counts["ok_no_signal"] += 1
            else:
                counts[o.kind] += 1
                if o.detail:
                    errors.append(f"{o.inst}/{o.tf}: {o.detail}")
        finally:
            if progress_callback is not None:
                progress_callback(o.inst, o.tf)

    try:
        for fut in as_completed(futures, timeout=SCAN_GLOBAL_TIMEOUT):
            _handle(fut)
    except TimeoutError:
        control.cancel()
    # Race : des futures peuvent finir entre le TimeoutError et ce balayage.
    for fut in futures:
        if fut in handled:
            continue
        if fut.done():
            _handle(fut)
        else:
            fut.cancel()
            counts["timed_out"] += 1

    _log(logging.INFO, "scan_end", correlation_id=correlation_id,
         signals=len(rows), pipeline=len(payloads), errors=len(errors),
         timed_out=counts["timed_out"], auth_aborted=control.auth_aborted,
         total_ms=round((time.perf_counter() - t0) * 1000, 1))
    return ScanResult(rows=rows, payloads=payloads, errors=errors,
                      scan_time=scan_time, auth_aborted=control.auth_aborted,
                      coverage_counts=counts)


# =====================================================================
# SECTION 7 — CONTRAT JSON & EXPORTS
# =====================================================================

_RE_SIGNAL_ID = re.compile(
    r"^[A-Z0-9_]{2,20}__(H1|H4|D1|Weekly)__\d{8}T\d{4}__[0-9a-f]{12}$")
_RE_PAIR = re.compile(r"^[A-Z0-9]{2,8}/[A-Z0-9]{2,8}$")
_RE_PAIR_OANDA = re.compile(r"^[A-Z0-9]{2,8}_[A-Z0-9]{2,8}$")
_RE_ISO_UTC = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$")
_SIGNAL_ENUMS: Final[Mapping[str, frozenset[str]]] = {
    "timeframe": frozenset(TIMEFRAMES),
    "type": frozenset({"CHoCH", "BOS"}),
    "direction": frozenset({"Bullish", "Bearish"}),
    "order": frozenset({"buy", "sell"}),
    "trend": frozenset({"Bullish", "Bearish"}),
    "status": frozenset(EMITTED_STATUSES),
    "volatility": frozenset({"Très Haute", "Haute", "Moyenne", "Basse"}),
    "force": frozenset({"Fort", "Moyen"}),
    "bb_regime": frozenset({"Squeeze", "Expansion", "Normal", "N/A"}),
    "session": frozenset(SESSION_BONUS),
}
_SIGNAL_REQUIRED: Final[tuple[str, ...]] = (
    "signal_id", "scanner_version", "rule_version", "generated_at", "pair",
    "pair_oanda", "timeframe", "type", "direction", "is_bullish", "order",
    "trend", "is_choch", "status", "confluence_score", "level", "close_price",
    "current_price", "distance_pct", "current_distance_pct",
    "distance_atr_multiple", "volatility", "force", "bb_width_pct",
    "bb_regime", "session", "signal_time", "candles_elapsed", "has_sweep",
    "atr", "confirmation_time", "age_minutes",
)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_signal(s: Mapping[str, Any]) -> Optional[str]:
    """Contrat d'un signal. Renvoie la raison du rejet, ou None."""
    try:
        extra = set(s) - set(_SIGNAL_REQUIRED)
        if extra:
            return f"cles en trop: {sorted(extra)}"
        missing = [k for k in _SIGNAL_REQUIRED if k not in s]
        if missing:
            return f"cles requises manquantes: {missing}"
        if s["scanner_version"] != SCANNER_VERSION:
            return "scanner_version incoherent"
        if s["rule_version"] != RULE_VERSION:
            return "rule_version incoherent"
        if not _RE_SIGNAL_ID.match(s["signal_id"]):
            return "signal_id: pattern invalide"
        if not (_RE_PAIR.match(s["pair"])
                and _RE_PAIR_OANDA.match(s["pair_oanda"])):
            return "pair/pair_oanda: pattern invalide"
        if s["pair"] != s["pair_oanda"].replace("_", "/"):
            return "pair != pair_oanda"
        for k, allowed in _SIGNAL_ENUMS.items():
            if s[k] not in allowed:
                return f"{k}: valeur hors enum ({s[k]!r})"
        tf = s["timeframe"]
        if not s["signal_id"].startswith(f"{s['pair_oanda']}__{tf}__"):
            return "signal_id incoherent avec pair_oanda/timeframe"
        for k in ("is_bullish", "is_choch", "has_sweep"):
            if not isinstance(s[k], bool):
                return f"{k}: pas un booleen"
        if s["is_bullish"] != (s["direction"] == "Bullish"):
            return "is_bullish != (direction == 'Bullish')"
        if s["order"] != ("buy" if s["is_bullish"] else "sell"):
            return "order != is_bullish"
        if s["is_choch"] != (s["type"] == "CHoCH"):
            return "is_choch != (type == 'CHoCH')"
        if s["type"] == "CHoCH" and s["trend"] == s["direction"]:
            return "CHoCH : trend == direction"
        if s["type"] == "BOS" and s["trend"] != s["direction"]:
            return "BOS : trend != direction"
        if s["type"] == "BOS" and s["has_sweep"]:
            return "BOS : has_sweep interdit"
        for k in ("level", "close_price", "current_price", "atr"):
            if not _is_num(s[k]) or not (math.isfinite(s[k]) and s[k] > 0):
                return f"{k}: valeur invalide"
        dam = s["distance_atr_multiple"]
        if not _is_num(dam) or not (0.0 <= dam <= ATR_DIST_MULT):
            return "distance_atr_multiple invalide ou hors borne"
        if round(dam, 2) != dam:
            return "distance_atr_multiple non arrondi a 2 decimales"
        if s["bb_width_pct"] is not None and not (
                _is_num(s["bb_width_pct"])
                and math.isfinite(s["bb_width_pct"])):
            return "bb_width_pct invalide"
        # Distances : recalcul EXACT depuis les prix publies (R11-4)
        for k_dist, k_px in (("distance_pct", "close_price"),
                             ("current_distance_pct", "current_price")):
            if _round4(calc_distance_pct(s["level"], s[k_px])) != s[k_dist]:
                return f"{k_dist} non recalculable depuis level/{k_px}"
        # Score : recalcul EXACT (R11-3)
        if not _is_int(s["confluence_score"]):
            return "confluence_score: pas un entier"
        expected_score = compute_confluence_score(
            dam, s["session"], s["has_sweep"])
        if s["confluence_score"] != expected_score:
            return (f"confluence_score {s['confluence_score']} != "
                    f"bareme {expected_score}")
        if s["confluence_score"] < MIN_SCORE:
            return "confluence_score < MIN_SCORE"
        # Horodatages
        for k in ("generated_at", "signal_time", "confirmation_time"):
            if not _RE_ISO_UTC.match(str(s[k])):
                return f"{k}: pas ISO 8601 UTC"
        sig_t = datetime.fromisoformat(s["signal_time"])
        conf_t = datetime.fromisoformat(s["confirmation_time"])
        gen_t = datetime.fromisoformat(s["generated_at"])
        if conf_t != confirmation_time(sig_t, tf):
            return "confirmation_time != signal_time + 1 barre"
        if s["session"] != get_session(sig_t, tf):
            return "session incoherente avec signal_time/timeframe"
        if not _is_int(s["age_minutes"]) or s["age_minutes"] != int(
                max(0.0, (gen_t - conf_t).total_seconds()) // 60):
            return "age_minutes incoherent avec confirmation_time"
        # Statut <-> candles_elapsed
        elapsed = s["candles_elapsed"]
        if not _is_int(elapsed) or elapsed < 0:
            return "candles_elapsed: entier >= 0 attendu"
        base = status_for_elapsed(elapsed, tf)
        if base == "Stale":
            return "signal expire (Stale) emis"
        if s["status"] == "Invalidated":
            if elapsed < 1:
                return "Invalidated sans bougie posterieure"
        elif s["status"] != base:
            return f"status {s['status']} != {base} (candles_elapsed)"
    except Exception as exc:  # noqa: BLE001 — frontiere fail-closed
        return f"validateur: {type(exc).__name__}"
    return None


def _json_default(obj: Any) -> Any:
    """Convertisseur strict : tout type inattendu leve (fail-loud)."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"objet non serialisable : {type(obj).__name__}")


def _sanitize_json(obj: Any) -> Any:
    """Float non fini -> None AVANT dumps (allow_nan=False leverait)."""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not math.isfinite(obj):
        return None
    return obj


_COVERAGE_KEYS: Final[tuple[str, ...]] = (
    "ok_signal", "ok_no_signal", "ok_not_emitted", "invalid_contract",
    "no_data", "failed", "aborted", "timed_out",
)


def serialize_pipeline(payloads: Sequence[Mapping[str, Any]],
                       scan_time: datetime, errors: Sequence[str],
                       coverage_counts: Mapping[str, int]) -> bytes:
    """Document JSON du pipeline. Fail-closed par signal (rejete + trace),
    fail-loud sur les invariants du document (RuntimeError)."""
    valid: list[Mapping[str, Any]] = []
    invalid: list[str] = []
    for p in payloads:
        reason = _validate_signal(p)
        if reason is None:
            valid.append(p)
        else:
            sid = p.get("signal_id", "<sans-id>")
            invalid.append(f"{sid}: {reason}")
            _log(logging.ERROR, "signal_rejete_contrat", signal_id=sid,
                 raison=reason)

    ids = [p["signal_id"] for p in valid]
    if len(ids) != len(set(ids)):
        raise RuntimeError("invariant viole : signal_id duplique")

    cov: dict[str, Any] = {
        "pairs_requested": len(INSTRUMENTS) * len(TIMEFRAMES)}
    for k in _COVERAGE_KEYS:
        cov[k] = int(coverage_counts[k])
    cov["ok_signal"] -= len(invalid)
    cov["invalid_contract"] += len(invalid)
    total = sum(cov[k] for k in _COVERAGE_KEYS)
    if total != cov["pairs_requested"]:
        raise RuntimeError(
            f"invariant de couverture viole : {total} != "
            f"{cov['pairs_requested']} "
            f"({ {k: cov[k] for k in _COVERAGE_KEYS} })")
    if cov["ok_signal"] != len(valid):
        raise RuntimeError(
            f"invariant viole : ok_signal={cov['ok_signal']} != {len(valid)}")
    cov["pairs_failed"] = len(errors)
    cov["pairs_timed_out"] = cov["timed_out"]
    cov["failures"] = list(errors[:50])

    doc = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "scanner_version": SCANNER_VERSION,
            "rule_version": RULE_VERSION,
            "generated_at": scan_time.isoformat(),
            "signal_count": len(valid),
            "status_counts": {
                st_: sum(1 for p in valid if p["status"] == st_)
                for st_ in EMITTED_STATUSES
            },
            "coverage": cov,
            "invalid_signals": invalid[:50],
        },
        "signals": sorted(valid, key=lambda p: p["signal_id"]),
    }
    return json.dumps(_sanitize_json(doc), ensure_ascii=False, indent=2,
                      default=_json_default, allow_nan=False).encode("utf-8")


# PDF : largeurs calibrees pour 802 pt (A4 paysage 842 - 2 x 20 de marge),
# police 6.5 pt, toutes les cellules en Paragraph (retour a la ligne,
# splitLongWords pour signal_id).
_PDF_WIDTHS: Final[Mapping[str, float]] = {
    "Instrument": 44, "Timeframe": 34, "Type": 32, "Ordre": 32,
    "Signal": 44, "Niveau": 44, "Distance%": 36, "Distance actuelle %": 40,
    "Score": 28, "Volatilité": 40, "Force": 32, "BB_Width": 58,
    "Statut": 46, "Ouverture bougie (UTC)": 50, "Confirmation (UTC)": 50,
    "signal_id": 192,
}
_PDF_USABLE_WIDTH: Final[float] = A4[1] - 40


def create_pdf(df_export: pd.DataFrame, scan_time: datetime) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=20,
                            rightMargin=20, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=6.5,
                          leading=8, alignment=1)
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold",
                          textColor=colors.white)
    cols = [c for c in DISPLAY_COLS if c in df_export.columns]
    widths = [_PDF_WIDTHS[c] for c in cols]
    total = sum(widths)
    if total > _PDF_USABLE_WIDTH:
        widths = [w * _PDF_USABLE_WIDTH / total for w in widths]
    data: list[list[Any]] = [[Paragraph(_xml_escape(c), head) for c in cols]]
    for row in df_export[cols].itertuples(index=False):
        data.append([Paragraph(_xml_escape(str(v)), cell) for v in row])
    if len(data) == 1:
        data.append([Paragraph("(aucun signal)", cell)]
                    + [Paragraph("", cell)] * (len(cols) - 1))
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.beige]),
    ]))
    doc.build([
        Paragraph(f"Rapport des Signaux CHoCH v{SCANNER_VERSION} "
                  f"({RULE_VERSION})", styles["Title"]),
        Paragraph(f"Généré le {scan_time.strftime('%d/%m/%Y à %H:%M')} UTC",
                  styles["Normal"]),
        Spacer(1, 20),
        table,
    ])
    return buffer.getvalue()


def generate_png(data: pd.DataFrame) -> bytes:
    """PNG pagine (PNG_ROWS_PER_PAGE lignes/page), pages empilees."""
    cols = [c for c in DISPLAY_COLS if c in data.columns]
    disp = data[cols]
    if disp.empty:
        disp = pd.DataFrame([["(aucun signal)"] + [""] * (len(cols) - 1)],
                            columns=cols)
    images: list[bytes] = []
    for start in range(0, len(disp), PNG_ROWS_PER_PAGE):
        page = disp.iloc[start:start + PNG_ROWS_PER_PAGE]
        fig = Figure(figsize=(22, min(max(5, len(page) * 0.35), 30)))
        ax = fig.add_subplot(111)
        ax.axis("off")
        tbl = ax.table(cellText=page.values, colLabels=page.columns,
                       cellLoc="center", loc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.2, 1.8)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        images.append(buf.getvalue())
    if len(images) == 1:
        return images[0]
    from PIL import Image  # dependance epinglee dans requirements.txt
    pages = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
    canvas = Image.new("RGB", (max(p.width for p in pages),
                               sum(p.height for p in pages)), "white")
    y = 0
    for p in pages:
        canvas.paste(p, (0, y))
        y += p.height
    out = io.BytesIO()
    canvas.save(out, format="png")
    return out.getvalue()


# =====================================================================
# SECTION 8 — COUCHE UI
# =====================================================================

@dataclass
class StoredScan:
    """Tout ce que l'UI affiche vient d'ici ; construit UNE fois par scan."""
    scan_time: datetime
    df: pd.DataFrame
    json_bytes: Optional[bytes]
    doc: Optional[dict[str, Any]]
    json_error: Optional[str]
    errors: list[str]
    timed_out: int
    auth_aborted: bool
    exports: dict[str, bytes] = field(default_factory=dict)


def _build_stored_scan(result: ScanResult) -> StoredScan:
    if result.rows:
        df = (pd.DataFrame(result.rows)
              .sort_values(["_status_rank", "_time_sort", "Instrument",
                            "Timeframe"],
                           ascending=[True, False, True, True],
                           kind="mergesort")
              .loc[:, list(DISPLAY_COLS)]
              .reset_index(drop=True))
    else:
        df = pd.DataFrame(columns=list(DISPLAY_COLS))
    json_bytes: Optional[bytes] = None
    doc: Optional[dict[str, Any]] = None
    json_error: Optional[str] = None
    try:
        json_bytes = serialize_pipeline(result.payloads, result.scan_time,
                                        result.errors,
                                        result.coverage_counts)
        doc = json.loads(json_bytes)
    except (RuntimeError, TypeError, ValueError) as exc:
        json_error = str(exc)
        _log(logging.ERROR, "pipeline_json_failed", err=json_error)
    return StoredScan(
        scan_time=result.scan_time, df=df, json_bytes=json_bytes, doc=doc,
        json_error=json_error, errors=list(result.errors),
        timed_out=result.coverage_counts["timed_out"],
        auth_aborted=result.auth_aborted,
    )


def _style_bb(val: object) -> str:
    s = str(val)
    if "Squeeze" in s:
        return "color:#ff9800;font-weight:bold"
    if "Expansion" in s:
        return "color:#ab47bc;font-weight:bold"
    return "color:#90a4ae"


def _style_distance(val: object) -> str:
    try:
        v = float(str(val).replace("%", ""))
    except ValueError:
        return "color:#90a4ae"
    if v <= 0.15:
        return "color:#00c853;font-weight:bold"
    if v <= 0.40:
        return "color:#ff9800;font-weight:bold"
    return "color:#ff5252;font-weight:bold"


_STATUS_STYLE: Final[Mapping[str, str]] = {
    "Fresh": "color:#00c853;font-weight:bold",
    "Aged": "color:#ff9800;font-weight:bold",
    "Stale": "color:#ff5252;font-weight:bold",
    "Invalidated": "color:#9e9e9e;text-decoration:line-through",
}


def _render_dataframe(df_all: pd.DataFrame) -> None:
    styled = (
        df_all.style
        .map(lambda x: "color:#e879f9;font-weight:bold" if x == "CHoCH"
             else "color:#94a3b8", subset=["Type"])
        .map(lambda x: "color:#00c853;font-weight:bold" if x == "Achat"
             else "color:#ff5252;font-weight:bold", subset=["Ordre"])
        .map(lambda x: "color:#00c853" if str(x).startswith("Bull")
             else "color:#ff5252", subset=["Signal"])
        .map(lambda x: "color:#00c853;font-weight:bold" if x == "Fort"
             else "color:#ff9800", subset=["Force"])
        .map(_style_bb, subset=["BB_Width"])
        .map(_style_distance, subset=["Distance%"])
        .map(lambda x: _STATUS_STYLE.get(str(x), ""), subset=["Statut"])
    )
    st.dataframe(styled, hide_index=True, width="stretch")


def _render_downloads(scan: StoredScan, df_export: pd.DataFrame) -> None:
    ts = scan.scan_time.strftime("%Y%m%d_%H%M%S")
    if "csv" not in scan.exports:
        # to_csv() sans chemin renvoie une chaine : on encode en utf-8-sig
        # pour que la BOM soit presente (Excel, accents).
        scan.exports["csv"] = df_export.to_csv(index=False).encode(
            "utf-8-sig")
    if "png" not in scan.exports:
        scan.exports["png"] = generate_png(df_export)
    if "pdf" not in scan.exports:
        scan.exports["pdf"] = create_pdf(df_export, scan.scan_time)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("CSV", scan.exports["csv"], f"choch_{ts}.csv",
                           "text/csv", key=f"dl_csv_{ts}")
    with c2:
        st.download_button("PNG", scan.exports["png"], f"choch_{ts}.png",
                           "image/png", key=f"dl_png_{ts}")
    with c3:
        st.download_button("PDF", scan.exports["pdf"],
                           f"choch_signaux_{ts}.pdf", "application/pdf",
                           key=f"dl_pdf_{ts}")
    with c4:
        if scan.json_bytes is not None:
            st.download_button("JSON", scan.json_bytes,
                               f"choch_pipeline_{ts}.json",
                               "application/json", key=f"dl_json_{ts}")
        else:
            st.error(f"JSON non généré (invariant violé) : {scan.json_error}")


def _render_results(scan: StoredScan) -> None:
    if scan.auth_aborted:
        st.error("Scan interrompu — trop d'erreurs d'authentification "
                 "OANDA. Vérifiez le token.")
    if scan.timed_out:
        st.warning(f"Timeout global — {scan.timed_out} requête(s) "
                   "non terminée(s), résultats partiels.")
    if scan.errors:
        st.warning(f"{len(scan.errors)} erreur(s) : "
                   f"{'; '.join(scan.errors[:5])}")

    df_all = scan.df
    df_export = df_all[df_all["Statut"].isin(EMITTED_STATUSES)]
    _render_downloads(scan, df_export)

    if scan.doc is not None:
        meta = scan.doc["meta"]
        cov = meta["coverage"]
        sc = meta["status_counts"]
        n_invalid = len(meta["invalid_signals"])
        st.caption(
            f"Pipeline JSON : {meta['signal_count']} signal(s) "
            f"(Fresh {sc['Fresh']}, Aged {sc['Aged']}, "
            f"Invalidated {sc['Invalidated']})"
            + (f", {n_invalid} rejeté(s) par le contrat" if n_invalid else "")
            + f" | schema {meta['schema_version']} | rule "
            f"{meta['rule_version']} | couverture {cov['pairs_requested']} = "
            f"{cov['ok_signal']} signal + {cov['ok_no_signal']} sans signal"
            f" + {cov['ok_not_emitted']} Stale + "
            f"{cov['invalid_contract']} rejetés + {cov['no_data']} no_data"
            f" + {cov['failed']} failed + {cov['aborted']} aborted + "
            f"{cov['timed_out']} timed_out"
        )

    if df_all.empty:
        st.info(f"Aucun signal CHoCH/BOS qualifié (Score ≥ {MIN_SCORE}).")
        return
    n_stale = int((df_all["Statut"] == "Stale").sum())
    if n_stale:
        st.info(f"{n_stale} signal(s) Stale visible(s) dans le tableau — "
                "exclus des exports et du pipeline JSON.")
    _render_dataframe(df_all)
    if scan.doc is not None and scan.doc["signals"]:
        with st.expander("Aperçu JSON Pipeline (premier signal)"):
            st.json(scan.doc["signals"][0])


def _init_session_state() -> None:
    for k, v in {"scanning": False, "cache_bust": 0, "scan": None}.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _trigger_scan() -> None:
    if st.session_state.scanning:
        return
    st.session_state.scanning = True
    try:
        try:
            creds = resolve_credentials()
        except ConfigError as exc:
            st.error(str(exc))
            return
        pb = st.progress(0.0, text="Initialisation du scan…")
        info = st.empty()
        t0 = time.monotonic()
        total = len(INSTRUMENTS) * len(TIMEFRAMES)
        done = 0

        def _tick(inst: str, tf: str) -> None:
            nonlocal done
            done += 1
            elapsed = time.monotonic() - t0
            eta = elapsed / done * (total - done)
            pb.progress(done / total,
                        text=f"[{done}/{total}] {inst.replace('_', '/')} "
                             f"({tf}) — ETA {int(eta)}s")
            info.caption(f"Workers: {SCAN_MAX_WORKERS} | Timeout: "
                         f"{SCAN_GLOBAL_TIMEOUT}s | Écoulé: {elapsed:.1f}s")

        try:
            result = run_scan(creds, st.session_state.cache_bust,
                              progress_callback=_tick)
        except Exception as exc:  # noqa: BLE001 — barriere finale
            _log(logging.ERROR, "scan_fatal", err=repr(exc))
            st.error(f"Erreur critique du scan : {exc}")
            return
        finally:
            pb.empty()
            info.empty()
        st.session_state.scan = _build_stored_scan(result)
    finally:
        st.session_state.scanning = False


# =====================================================================
# SECTION 9 — POINT D'ENTREE STREAMLIT
# =====================================================================

st.set_page_config(page_title=f"CHoCH Scanner v{SCANNER_VERSION}",
                   layout="wide")
st.title(f"Scanner Change of Character (CHoCH) — v{SCANNER_VERSION} "
         f"({RULE_VERSION})")
_init_session_state()

col_a, col_b = st.columns([3, 1])
with col_a:
    scan_clicked = st.button("Lancer le Scan", type="primary",
                             width="stretch", key="btn_scan")
with col_b:
    force_refresh = st.button("Force refresh (bust cache)", width="stretch",
                              key="btn_force_refresh")

if force_refresh:
    # Nouvelle cle de cache pour CETTE session ; n'efface pas le cache des
    # autres sessions (pas de get_candles_cached.clear() global).
    st.session_state.cache_bust += 1
    st.toast("Le prochain scan rechargera les bougies depuis OANDA.")

if scan_clicked:
    _trigger_scan()

if st.session_state.scan is not None:
    _render_results(st.session_state.scan)
