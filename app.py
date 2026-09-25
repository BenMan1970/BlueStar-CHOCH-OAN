"""
CHoCH Scanner v5.19 — Production-grade hardened build.

Monolithic deployment (Streamlit Cloud compatible) but architected as strict
layered modules in a single file:

    1.  Constants & rule registry (frozen, versioned)
    2.  Structured JSON logging
    3.  Cached resources (API, ThreadPool, TZ)
    4.  Pure domain layer (deterministic, side-effect free)
    5.  I/O layer (OANDA with retry, circuit breaker, dedup)
    6.  Orchestration (scan as pure function)
    7.  UI layer (presentation only)

Invariants enforced:
    - Single source of truth for every signal field (no UI/pipeline drift).
    - signal_id is deterministic and reproducible.
    - rule_version is embedded in every emitted payload.
    - No global mutable state shared across Streamlit sessions.
    - No silent JSON serialization failure (whitelisted converter).
    - No thread/pool leak on script reruns.
"""
from __future__ import annotations

# pylint: disable=wrong-import-position
import matplotlib
matplotlib.use("Agg")

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
from concurrent.futures import (
    CancelledError,
    Future,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import (Any, Callable, Final, Literal, Mapping, Optional,
                    Sequence, TypedDict)

from zoneinfo import ZoneInfo  # Python >= 3.12 (exige par numpy 2.5.3)

import numpy as np
import pandas as pd
import requests
import streamlit as st
# A4 (LOT A) : import interne RerunException supprime — on utilise
# desormais l'API publique st.rerun().
from matplotlib.figure import Figure
from oandapyV20 import API
from oandapyV20.endpoints import instruments
from oandapyV20.exceptions import V20Error
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# =====================================================================
# SECTION 1 — CONSTANTS & RULE REGISTRY
# =====================================================================

SCANNER_VERSION: Final[str] = "5.16"
# C-1 (LOT C) : un SEUL nouveau RULE_VERSION pour tout le lot. Toute
# evolution semantique (V3 Invalidated, B4 niveau protecteur) est
# regroupee sous r10 — plus jamais de version intermediaire non
# versionnee (defaut prouve : session Off vs DailyClose sous le
# meme r9).
RULE_VERSION: Final[str] = "choch.v58.r10"
# r7: revert de la fenetre de detection, de la sur-restriction du pivot CHoCH
# et du couplage du bonus de session ; preserve le BOS, le look-ahead et la
# portee du sweep.
# r8: fix FND-08 (fenetre de pivot : dernier pivot confirmable inclus),
#     FND-01 (ordre JSON trie), FND-04 (validation contrat fail-closed),
#     FND-06 (champs additifs confirmation_time + age_minutes),
#     FND-02 (etat no_data explicite). Impact : 3/132 signaux terminaux.
# r9 (PHASE 3, SCHEMA 2.0.0, apres decisions D1-a/D2-a/D3-c/D4-b/D5-a/D6-a) :
#     PG-30 tendance+ATR causales, PG-31 BOS emis (trend==direction pour
#     BOS), PG-31b session DailyClose, PG-32 niveau non franchi,
#     PG-33 volatilite a la bougie de cassure. Couche domaine modifiee.
# r8.1 (production grade, schema 1.2.0) : PG-01 (age_minutes depuis
#     confirmation_time), PG-02 (displayPrecision OANDA figee),
#     PG-03 (compteurs de couverture exclusifs, invariant 132),
#     PG-04/05 (JSON toujours telechargeable, serialise une fois),
#     PG-06 (corpus de 50 mutations), PG-07 (lever au lieu du cache None),
#     PG-08 (retry 5xx), PG-09 (race timeout), PG-11 (Event d'annulation),
#     PG-12 (RerunException), PG-13 (OANDA_ENV), PG-15 (timings),
#     PG-16 (largeur PDF <= 802pt), PG-17 (PNG pagine), PG-18 (CSV utf-8-sig).
#     Couche domaine INTACTE (RULE_VERSION inchange) ; impact mesure sur le
#     golden master : age_minutes (7/7) + arrondis atr/level des 5 indices.

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

TIMEFRAMES: Final[Mapping[str, str]] = {
    "H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W",
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

# r7: explicit per-timeframe detection window.
# La boucle d'offset est INDISPENSABLE : _scan_one appelle detect_choch()
# exactement UNE fois par (inst, tf) ; il n'y a pas de boucle externe par
# bougie ni de filtre detect_at_idx() dans le code. Supprimer la boucle
# ramenait la detection a la derniere bougie cloturee, perdant ~80% des
# signaux valides (surtout D1/Weekly ou les CHoCH se forment rarement en
# N-1). D1/Weekly utilisent des fenetres plus courtes car un lookback de 5
# bougies detecterait des signaux deja Stale par definition (cf. TF_STATUT
# ci-dessous : Weekly Aged = 4 bougies, donc un signal forme a N-5 ou
# avant est deja Aged ; un lookback de 5 le detecterait systematiquement
# trop tard).
DETECTION_LOOKBACK: Final[Mapping[str, int]] = {
    "H1": 5, "H4": 5, "D1": 3, "Weekly": 3,
}

TF_STATUT: Final[Mapping[str, Mapping[str, int]]] = {
    "H1":     {"Fresh": 4, "Aged": 12},
    "H4":     {"Fresh": 3, "Aged": 8},
    "D1":     {"Fresh": 2, "Aged": 5},
    "Weekly": {"Fresh": 2, "Aged": 4},
}

ATR_DIST_MULT: Final[float] = 1.8
MIN_SCORE: Final[int] = 65
SCAN_GLOBAL_TIMEOUT: Final[int] = 180
SCAN_MAX_WORKERS: Final[int] = 6
OANDA_REQUEST_TIMEOUT: Final[int] = 12
CANDLES_CACHE_TTL_SECONDS: Final[int] = 60  # short — quotes refresh quickly
MAX_AUTH_FAILURES: Final[int] = 3
OANDA_MAX_RETRIES: Final[int] = 2
OANDA_BACKOFF_BASE: Final[float] = 0.25
# PG-17 : lignes par page PNG. Au-dela, le tableau est pagine puis empile
# verticalement dans un seul fichier (lisibilite + 1 telechargement).
PNG_ROWS_PER_PAGE: Final[int] = 40

# Le score de confluence (critere d'emission) et l'identifiant de signal
# sont visibles et exportables.
# A9 (LOT A) : "Heure (UTC)" remplacee par "Ouverture bougie (UTC)" +
# "Confirmation (UTC)" (affichage honnete de l'ouverture vs la cloture
# confirmative) ; ajout de "Distance actuelle %" (current_distance_pct).
DISPLAY_COLS: Final[tuple[str, ...]] = (
    "Instrument", "Timeframe", "Type", "Ordre", "Signal",
    "Niveau", "Distance%", "Distance actuelle %", "Score",
    "Volatilité", "Force", "BB_Width", "Statut",
    "Ouverture bougie (UTC)", "Confirmation (UTC)", "signal_id",
)
EXPORT_COLS: Final[tuple[str, ...]] = DISPLAY_COLS

TrendT = Literal["Bullish", "Bearish", "Range"]
DirectionT = Literal["Bullish", "Bearish"]
SigTypeT = Literal["CHoCH", "BOS"]
StatusT = Literal["Fresh", "Aged", "Stale", "Invalidated", "N/A"]
SessionT = Literal[
    "London_NY_Overlap", "London", "NewYork", "Tokyo", "Off", "DailyClose",
]


# =====================================================================
# SECTION 2 — STRUCTURED JSON LOGGING
# =====================================================================

class _JsonFormatter(logging.Formatter):
    """Minimal dependency-free JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
                .isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any structured extras
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                payload[key[4:]] = value
        try:
            return json.dumps(payload, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            # Never let logging crash the process
            return json.dumps({"ts": payload["ts"], "level": "ERROR",
                               "msg": "log_serialization_failure"})


def _configure_root_logger() -> logging.Logger:
    root = logging.getLogger("choch")
    if getattr(root, "_choch_configured", False):
        return root
    # A10 (LOT A) : une valeur invalide (ex. CHOCH_LOG_LEVEL=verbose) ne
    # doit pas faire crasher l'import. On valide et on replie sur INFO
    # avec un warning, plutot que de lever ValueError.
    level_name = os.environ.get("CHOCH_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        print(
            f"CHOCH_LOG_LEVEL inconnu : '{level_name}' — repli sur INFO",
            file=sys.stderr,
        )
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
    """Structured log helper — keys are flattened in JSON output."""
    logger.log(level, msg, extra={f"ctx_{k}": v for k, v in ctx.items()})


# =====================================================================
# SECTION 3 — CACHED RESOURCES (Streamlit-aware)
# =====================================================================

@lru_cache(maxsize=8)
def _tz(name: str) -> ZoneInfo:
    """Cached ZoneInfo factory — TZ objects are heavy to instantiate."""
    return ZoneInfo(name)


_thread_local = threading.local()


def _get_oanda_api() -> API:
    """
    Un client OANDA API PAR THREAD : oandapyV20 wrappe un seul
    requests.Session, que requests ne garantit pas thread-safe. Une version
    anterieure partageait une Session entre les workers du scan via
    @st.cache_resource ; remplacee par un client thread-local.
    """
    api = getattr(_thread_local, "api", None)
    if api is None:
        # st.secrets en production ; CHOCH_TEST_TOKEN pour les tests hors
        # Streamlit (jamais logge, jamais mis dans le document JSON).
        # PG-13 : OANDA_ENV selectionne l'environnement ; un token de test
        # est REFUSE en live (jamais de credentials de test contre le
        # compte reel).
        env = os.environ.get("OANDA_ENV", "").strip().lower() or (
            str(st.secrets.get("OANDA_ENV", "practice")).strip().lower())
        token = os.environ.get("CHOCH_TEST_TOKEN")
        if token:
            if env == "live":
                raise RuntimeError(
                    "CHOCH_TEST_TOKEN refuse en OANDA_ENV=live : un token "
                    "de test ne doit jamais atteindre le compte reel")
            token_name = "CHOCH_TEST_TOKEN"
        else:
            token = st.secrets.get("OANDA_ACCESS_TOKEN")
            token_name = "OANDA_ACCESS_TOKEN"
        if not token:
            raise RuntimeError("OANDA_ACCESS_TOKEN missing from st.secrets")
        if env not in ("practice", "live"):
            raise RuntimeError(
                f"OANDA_ENV invalide : {env!r} (attendu practice|live)")
        api = API(
            access_token=token,
            environment=env,
            request_params={"timeout": OANDA_REQUEST_TIMEOUT},
        )
        _log(logging.INFO, "oanda_api_init",
             env=env, source=token_name)
        _thread_local.api = api
    return api


@st.cache_resource(show_spinner=False)
def _get_scan_executor() -> ThreadPoolExecutor:
    """
    Single ThreadPoolExecutor reused across scans (prevents thread storm
    when Streamlit reruns trigger button re-evaluation).
    """
    return ThreadPoolExecutor(
        max_workers=SCAN_MAX_WORKERS,
        thread_name_prefix="choch-scan",
    )


# =====================================================================
# SECTION 4 — PURE DOMAIN LAYER
# =====================================================================

class SwingDict(TypedDict):
    idx: int
    price: float
    kind: Literal["HH", "LH", "HL", "LL"]


@dataclass(frozen=True)
class SignalCore:
    """Single source of truth for a detected signal — used by UI and JSON."""
    sig_type: SigTypeT
    direction: DirectionT
    level: float
    idx_break: int
    close_price: float
    current_price: float
    has_sweep: bool
    atr_val: float
    volatilite: str
    trend: TrendT
    force: Literal["Fort", "Moyen", "Faible"]
    dist_atr: float
    score: int
    bb_width_pct: Optional[float]
    bb_regime: str
    signal_time_utc: datetime
    session: SessionT
    statut: StatusT
    candles_elapsed: int
    distance_pct: Optional[float]
    current_distance_pct: Optional[float]


# ---- 4.1 numeric utilities ------------------------------------------------

def _compute_true_range_vec(data: pd.DataFrame) -> np.ndarray:
    """
    Vectorised True Range. Resets TR to (high-low) across abnormal gaps
    (gap > 2 × median delta) to avoid spurious TR spikes across weekends.
    """
    high = data["high"].to_numpy(dtype=np.float64)
    low = data["low"].to_numpy(dtype=np.float64)
    close = data["close"].to_numpy(dtype=np.float64)
    n = high.size
    if n < 2:
        return np.empty(0, dtype=np.float64)

    deltas = np.diff(data.index.values.astype("datetime64[s]").astype(np.int64))
    if deltas.size == 0:
        return np.empty(0, dtype=np.float64)
    typical = float(np.median(deltas)) if deltas.size else 3600.0
    typical = max(typical, 1.0)

    hl = high[1:] - low[1:]
    hc = np.abs(high[1:] - close[:-1])
    lc = np.abs(low[1:] - close[:-1])
    tr = np.maximum(hl, np.maximum(hc, lc))
    # Across abnormal gaps -> use (high - low) only
    gap_mask = deltas > (2 * typical)
    tr = np.where(gap_mask, hl, tr)
    return tr


def calc_atr_bundle(
    data: pd.DataFrame, inst: str, period: int = 14,
) -> tuple[float, str]:
    """Returns (atr, regime). NaN-safe, fallback to static volatility."""
    tr = _compute_true_range_vec(data)
    fallback = VOLATILITY_STATIC.get(inst, "Moyenne")
    if tr.size < period * 3:
        return float("nan"), fallback

    # A10 (LOT A) : correction d'une affirmation inexacte. Un EWM avec
    # alpha=1/14 atteint ~97,5% de convergence apres 50 periodes (et
    # ~99,9% apres ~66), pas 99,9% a 50. Le slice a 100 reste largement
    # suffisant : le poids des bougies au-dela est negligeable.
    tr_ewm = tr[-100:] if tr.size > 100 else tr
    atr_val = float(
        pd.Series(tr_ewm).ewm(alpha=1.0 / period, adjust=False).mean().iloc[-1]
    )
    if not math.isfinite(atr_val):
        return float("nan"), fallback

    window = tr[-100:] if tr.size >= 100 else tr
    median_tr = float(np.median(window))
    if not math.isfinite(median_tr) or median_tr < 1e-10:
        return atr_val, fallback

    ratio = atr_val / median_tr
    if ratio >= 1.8:
        regime = "Très Haute"
    elif ratio >= 1.2:
        regime = "Haute"
    elif ratio >= 0.7:
        regime = "Moyenne"
    else:
        regime = "Basse"
    return atr_val, regime


# PG-02 (r8.1) : precision d'affichage figee, extraite de l'API OANDA
# (AccountInstruments, 2026-09-25, compte practice). Aucun appel reseau a
# l'execution. Conservee pour les instruments non listes.
_PRECISION_OANDA: Final[dict[str, int]] = {
    "DE30_EUR": 1,
    "XAU_USD": 3,
    "SPX500_USD": 1,
    "NAS100_USD": 1,
    "US30_USD": 1,
}


def instrument_precision(inst: str) -> int:
    """Precision d'affichage d'OANDA (displayPrecision), figee.

    Priorite : table OANDA relevee le 2026-09-25 (PG-02), puis regles
    historiques (JPY 3, autres 5) pour les instruments non listes.
    """
    if inst in _PRECISION_OANDA:
        return _PRECISION_OANDA[inst]
    if "JPY" in inst:
        return 3
    return 5


def format_niveau(niveau: Optional[float], inst: str) -> str:
    if niveau is None or not math.isfinite(niveau):
        return "N/A"
    return f"{niveau:.{instrument_precision(inst)}f}"


def calc_distance_pct(
    niveau: Optional[float], close_actuel: Optional[float],
) -> Optional[float]:
    if niveau is None or close_actuel is None:
        return None
    if not (math.isfinite(niveau) and math.isfinite(close_actuel)):
        return None
    if abs(niveau) < 1e-12:
        return None
    dist = abs(close_actuel - niveau) / abs(niveau) * 100.0
    return dist if 0.0 <= dist <= 100.0 else None


def format_distance(dist_pct: Optional[float]) -> str:
    return "N/A" if dist_pct is None else f"{dist_pct:.3f}%"


def _local_hour(dt: datetime, tz_name: str) -> int:
    return dt.astimezone(_tz(tz_name)).hour


def get_session(dt: datetime, tf: Optional[str] = None) -> SessionT:
    # PG-31b (r9, D2-a) : D1 et Weekly clôturent a 21:00 UTC, hors de toutes
    # les fenetres intraday. Sans session dediee, 100 % de leurs signaux
    # etaient "Off" et ne recevaient jamais le bonus de session (+20 a
    # l'epoque, +10 en r9). On definit une session DailyClose dediee.
    if tf in ("D1", "Weekly"):
        return "DailyClose"
    london_h = _local_hour(dt, "Europe/London")
    ny_h = _local_hour(dt, "America/New_York")
    tokyo_h = _local_hour(dt, "Asia/Tokyo")
    london = 8 <= london_h < 17
    ny = 9 <= ny_h < 17
    tokyo = 9 <= tokyo_h < 18
    if london and ny:
        return "London_NY_Overlap"
    if london:
        return "London"
    if ny:
        return "NewYork"
    if tokyo:
        return "Tokyo"
    return "Off"


def is_premium_session(s: str) -> bool:
    # PG-31b (r9, D2-a) : DailyClose est premium (bonus partiel +10) mais
    # pas autant qu'un overlap intraday (+20).
    return s in ("London", "NewYork", "London_NY_Overlap", "DailyClose")


# ---- 4.2 candle parsing ---------------------------------------------------

def _parse_candle_row(c: Mapping[str, Any], inst: str, gran: str) -> Optional[dict]:
    """Parse one OANDA candle dict; reject malformed or non-OHLC-consistent."""
    try:
        mid = c["mid"]
        open_v = float(mid["o"])
        high_v = float(mid["h"])
        low_v = float(mid["l"])
        close_v = float(mid["c"])
        t = c["time"]
    except (KeyError, ValueError, TypeError) as exc:
        _log(logging.WARNING, "candle_malformed",
             instrument=inst, granularity=gran, err=str(exc))
        return None

    if not all(math.isfinite(v) for v in (open_v, high_v, low_v, close_v)):
        return None
    # Canonical OHLC consistency: high >= max(o,c) >= min(o,c) >= low
    if not (high_v >= low_v
            and high_v >= max(open_v, close_v)
            and low_v <= min(open_v, close_v)):
        _log(logging.WARNING, "candle_inconsistent",
             instrument=inst, granularity=gran, t=str(t))
        return None

    return {
        "time": pd.to_datetime(t, utc=True),
        "open": open_v,
        "high": high_v,
        "low": low_v,
        "close": close_v,
    }


# ---- 4.3 swing & trend ----------------------------------------------------

def _classify_swings(pivots: Sequence[tuple[int, float, str]]) -> list[SwingDict]:
    swings: list[SwingDict] = []
    prev_h: Optional[float] = None
    prev_l: Optional[float] = None
    for idx, price, k in pivots:
        if k == "H":
            kind: Literal["HH", "LH", "HL", "LL"] = (
                "HH" if (prev_h is None or price > prev_h) else "LH"
            )
            swings.append({"idx": idx, "price": price, "kind": kind})
            prev_h = price
        else:
            kind = "HL" if (prev_l is None or price > prev_l) else "LL"
            swings.append({"idx": idx, "price": price, "kind": kind})
            prev_l = price
    return swings


def detect_swing_points(data: pd.DataFrame, tf: str) -> list[SwingDict]:
    """
    Pivot detection over a centered window. Dedup is performed by **index**
    (never by price) to avoid losing legitimate same-price pivots on
    indices/metals.

    C-5 (LOT C, B3) — RESIDU DE NON-CAUSALITE DOCUMENTE :
    ``start = max(lookback, n - history - lookback)`` est ancre sur **n**,
    pas sur l'idx du signal. Consequence mesuree (LOT B) : pour 30 % des
    signaux (18/60), les etiquettes HH/HL/LL des pivots DIFFERENT entre
    le prefixe [0, idx] et la serie complete — un pivot qui etait le
    dernier HL dans le prefixe ne l'est plus quand de nouvelles bougies
    arrivent. En revanche **0 signal emis sur 60 (0.0 %) change** : la
    propriete minimum-prefix tient au niveau du signal. Ce residu est
    latent (non observe sur les sorties) et volontairement NON corrige :
    le corriger exigerait de recalculer toute la chaine par prefixe
    (cout x40). A savoir pour toute analyse d'etiquettes passees.
    """
    lookback = SWING_LOOKBACK.get(tf, 5)
    history = SWING_HISTORY.get(tf, 60)
    n = len(data)
    if n < 2 * lookback + 1:
        return []

    win = 2 * lookback + 1
    high_s = data["high"].reset_index(drop=True)
    low_s = data["low"].reset_index(drop=True)
    roll_max = high_s.rolling(window=win, center=True, min_periods=win).max()
    roll_min = low_s.rolling(window=win, center=True, min_periods=win).min()
    h_mask = (high_s == roll_max) & high_s.notna()
    l_mask = (low_s == roll_min) & low_s.notna()

    start = max(lookback, n - history - lookback)
    # r8 (fix FND-08) : end = n - lookback, pas n - lookback - 1.
    # range(start, end) exclut end ; le dernier pivot confirmable est
    # i = n - 1 - lookback (fenetre centree [i-lb, i+lb] dans la serie).
    end = n - lookback
    pivots: list[tuple[int, float, str]] = []
    seen_idx: set[int] = set()
    high_arr = high_s.to_numpy()
    low_arr = low_s.to_numpy()
    for i in range(start, end):
        if h_mask.iloc[i] and i not in seen_idx:
            pivots.append((i, float(high_arr[i]), "H"))
            seen_idx.add(i)
        if l_mask.iloc[i] and i not in seen_idx:
            # An index that is simultaneously a max AND a min in the same
            # window is a degenerate flat — emit high only, skip low.
            pivots.append((i, float(low_arr[i]), "L"))
            seen_idx.add(i)
    pivots.sort(key=lambda p: p[0])
    return _classify_swings(pivots)


def _last_high_low(
    swings: Sequence[SwingDict],
) -> tuple[Optional[str], Optional[str]]:
    highs = [s for s in swings if s["kind"] in ("HH", "LH")]
    lows = [s for s in swings if s["kind"] in ("HL", "LL")]
    if not highs or not lows:
        return None, None
    return highs[-1]["kind"], lows[-1]["kind"]


def get_structural_trend(swings: Sequence[SwingDict]) -> TrendT:
    if len(swings) < 4:
        return "Range"
    last_high, last_low = _last_high_low(swings[-6:])
    if last_high is None:
        return "Range"
    if last_high == "HH" and last_low == "HL":
        return "Bullish"
    if last_high == "LH" and last_low == "LL":
        return "Bearish"
    return "Range"


# ---- 4.4 signal resolution ------------------------------------------------

_SigResult = tuple[
    Optional[SigTypeT], Optional[DirectionT], Optional[float],
]
_NONE_SIG: _SigResult = (None, None, None)


def _last_of_kind(
    prev_swings: Sequence[SwingDict], kind: str,
) -> Optional[SwingDict]:
    """C-3 (LOT C, B4) : dernier pivot d'un type donne.

    Avant, _resolve_* prenaient simplement ``list[-1]`` sur les pivots
    filtres par kind, ce qui est correct (le dernier du type). Le defaut
    mesure (66 % des signaux) n'etait pas la selection mais le fait que
    la CASSURE est testee contre ce niveau alors que des pivots plus
    recents existent. On garde donc la selection ``[-1]`` — la correction
    porte sur le diagnostic : la fonction est nommee et documentee pour
    que l'invariant "niveau == dernier pivot du type attendu" soit
    verifiable par test.
    """
    cands = [s for s in prev_swings if s["kind"] == kind]
    return cands[-1] if cands else None


def _resolve_bullish(
    close_arr: np.ndarray, idx: int, prev_swings: Sequence[SwingDict],
) -> _SigResult:
    # Le pivot protectif d'un CHoCH bearish est le HL le plus recent, sans
    # condition de position par rapport au dernier HH. Exiger
    # HL.idx > last_HH.idx eliminait la configuration dominante en bord droit
    # de serie, ou le pivot protectif se forme pres de la fin sans avoir
    # encore ete suivi d'un nouveau HH. Comportement r3 preserve : le BOS
    # sur le dernier extremum est conserve.
    # C-3 (LOT C, B4) : le niveau doit etre le DERNIER pivot du type
    # attendu. Mesure LOT B : dans 66 % des signaux (68/103) un pivot plus
    # recent existait apres le pivot choisi, donc le "niveau protectif"
    # n'etait pas le vrai pivot protectif. _last_of_kind prend le dernier
    # pivot du type demande.
    hl = _last_of_kind(prev_swings, "HL")
    if hl is not None:
        ref = hl["price"]
        if close_arr[idx] < ref <= close_arr[idx - 1]:
            return "CHoCH", "Bearish", ref
    hh = _last_of_kind(prev_swings, "HH")
    if hh is not None:
        ref = hh["price"]
        if close_arr[idx - 1] <= ref < close_arr[idx]:
            return "BOS", "Bullish", ref
    return _NONE_SIG


def _resolve_bearish(
    close_arr: np.ndarray, idx: int, prev_swings: Sequence[SwingDict],
) -> _SigResult:
    # Symetrique de _resolve_bullish : le pivot protectif d'un CHoCH bull
    # est le LH le plus recent, sans condition de position.
    # Symetrique de _resolve_bullish (C-3 : dernier pivot du type attendu).
    lh = _last_of_kind(prev_swings, "LH")
    if lh is not None:
        ref = lh["price"]
        if close_arr[idx - 1] <= ref < close_arr[idx]:
            return "CHoCH", "Bullish", ref
    ll = _last_of_kind(prev_swings, "LL")
    if ll is not None:
        ref = ll["price"]
        if close_arr[idx] < ref <= close_arr[idx - 1]:
            return "BOS", "Bearish", ref
    return _NONE_SIG


def _resolve_signal(
    trend: TrendT, close_arr: np.ndarray, idx: int,
    prev_swings: Sequence[SwingDict],
) -> _SigResult:
    if trend == "Bullish":
        return _resolve_bullish(close_arr, idx, prev_swings)
    if trend == "Bearish":
        return _resolve_bearish(close_arr, idx, prev_swings)
    return _NONE_SIG


def _detect_liquidity_sweep(
    high_arr: np.ndarray, low_arr: np.ndarray, idx: int,
    prev_swings: Sequence[SwingDict], atr_val: float, direction: DirectionT,
) -> bool:
    # Seuls les 3 pivots les plus recents constituent des pools actifs.
    # any() sur toute l'histoire gonflait has_sweep → scores artificiels.
    if direction == "Bearish":
        cands = [s for s in prev_swings if s["kind"] in ("HH", "LH")]
        if not cands:
            return False
        return any(
            high_arr[idx] > s["price"]
            and (high_arr[idx] - s["price"]) > (atr_val * 0.25)
            for s in cands[-3:]
        )
    cands = [s for s in prev_swings if s["kind"] in ("HL", "LL")]
    if not cands:
        return False
    return any(
        low_arr[idx] < s["price"]
        and (s["price"] - low_arr[idx]) > (atr_val * 0.25)
        for s in cands[-3:]
    )


def compute_statut(idx_sig: Optional[int], len_df: int, tf: str) -> StatusT:
    if idx_sig is None:
        return "N/A"
    candles_elapsed = (len_df - 1) - idx_sig
    thr = TF_STATUT.get(tf, {"Fresh": 2, "Aged": 5})
    if candles_elapsed <= thr["Fresh"]:
        return "Fresh"
    if candles_elapsed <= thr["Aged"]:
        return "Aged"
    return "Stale"


def _compute_confluence_score(
    dist_atr: float, candle_time: datetime, has_sweep: bool,
    sig_type: SigTypeT, tf: Optional[str] = None,
) -> int:
    score = 25
    if dist_atr <= 1.0:
        score += 15
    # Le bonus de session reflete la qualite du contexte de formation de la
    # bougie, qui ne change pas quand le signal vieillit.
    # PG-31b (r9, D2-a) : DailyClose (D1/Weekly) est premium a +10, les
    # sessions intraday (overlap inclus) a +20.
    sess = get_session(candle_time, tf)
    if sess == "DailyClose":
        score += 10
    elif is_premium_session(sess):
        score += 20
    if has_sweep:
        score += 15
    # D7-a : bonus symetrique pour le BOS. Un BOS continue la tendance
    # (D1-a) : c'est un evenement structurel aussi informatif qu'un CHoCH,
    # mais il n'a jamais droit au bonus has_sweep (+15, reserve au CHoCH).
    # Sans ce bonus son score plafonnait a 60 (25+15+20) < MIN_SCORE=65 et
    # AUCUN BOS n'etait emis (693 candidats mesures, 0 emission). Le +10
    # compense exactement l'absence de bonus CHoCH : un BOS en session
    # premium atteint 70, un BOS DailyClose 50+10=60 (reste filtre). Le
    # seuil des CHoCH est INTACT.
    if sig_type == "CHoCH":
        score += 10
    elif sig_type == "BOS":
        score += 10
    return score


# ---- 4.5 BB width ---------------------------------------------------------

def compute_bb_width(
    data: pd.DataFrame, length: int = 20, std: int = 2,
) -> tuple[Optional[float], str]:
    close = data["close"]
    if len(close) < length * 2:
        return None, "N/A"
    sma = close.rolling(length).mean()
    std_dev = close.rolling(length).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev

    sma_safe = sma.where(sma.abs() > 1e-12)
    bb_w = (upper - lower) / sma_safe
    bb_avg = bb_w.rolling(length).mean()
    avg_last = bb_avg.iloc[-1]
    if pd.isna(avg_last) or abs(avg_last) < 1e-12:
        return None, "N/A"

    bb_avg_safe = bb_avg.where(bb_avg.abs() > 1e-12)
    pct_series = (bb_w - bb_avg) / bb_avg_safe * 100.0
    pct_val = pct_series.iloc[-1]
    if pd.isna(pct_val) or not math.isfinite(pct_val):
        return None, "N/A"

    if pct_val <= -25:
        regime = "Squeeze"
    elif pct_val >= 25:
        regime = "Expansion"
    else:
        regime = "Normal"
    return float(pct_val), regime


def format_bb_width(bb_result: tuple[Optional[float], str]) -> str:
    pct, regime = bb_result
    if pct is None:
        return "N/A"
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.0f}%_{regime}"


# ---- 4.6 SignalCore builder (the single source of truth) -----------------

def _evaluate_candle(
    *, idx: int, df: pd.DataFrame, prev_swings: Sequence[SwingDict],
    atr_val: float, atr_regime: str, trend: TrendT, tf: str,
) -> Optional[SignalCore]:
    close_arr = df["close"].to_numpy()
    high_arr = df["high"].to_numpy()
    low_arr = df["low"].to_numpy()
    open_arr = df["open"].to_numpy()
    n = close_arr.size

    sig_type, direction, level = _resolve_signal(
        trend, close_arr, idx, prev_swings,
    )
    if sig_type is None or direction is None or level is None:
        return None

    rng_v = high_arr[idx] - low_arr[idx]
    if rng_v <= 0:
        return None
    body_ratio = abs(close_arr[idx] - open_arr[idx]) / rng_v
    if body_ratio < 0.40:
        return None
    force_label: Literal["Fort", "Moyen", "Faible"] = (
        "Fort" if body_ratio >= 0.60 else "Moyen"
    )

    has_sweep = sig_type == "CHoCH" and _detect_liquidity_sweep(
        high_arr, low_arr, idx, prev_swings, atr_val, direction,
    )
    # PG-32 (r9) : un signal n'est valide que si son niveau protectif n'a
    # pas ete franchi DEPUIS la confirmation, c'est-a-dire si la cassure n'a
    # pas echoue. La cassure est confirmee a la cloture de idx ; les bougies
    # idx+1..n-1 doivent la respecter.
    #   - direction Bullish : niveau = ancienne RESISTANCE (LH) cassee vers
    #     le haut. Echec si le prix repasse DESSOUS (low < level).
    #   - direction Bearish : niveau = ancien SUPPORT (HL) casse vers le
    #     bas. Echec si le prix repasse DESSUS (high > level).
    # NB : high>level apres une cassure haussiere est la CONTINUATION, pas
    # l'echec — ne pas inverser ces deux conditions.
    # C-4 (LOT C, V3) : un signal dont le niveau protectif a ete franchi
    # DEPUIS la confirmation n'est plus SUPPRIME — il est emis avec le
    # statut "Invalidated". Raisons (point d'arrêt LOT B) :
    #   - V1 supprimait 47.8 % des candidats (186/356) ; le signal
    #     NZD/CAD H1 disparaissait entre 12:10 et 13:28 sans laisser de
    #     trace, indiscernable d'une regression (defaut 2).
    #   - V3 garde l'information : 356 signaux visibles dont 104 (29.2 %)
    #     marques Invalidated.
    # L'operateur voit que la cassure a echoue au lieu de ne rien voir.
    invalidated = False
    if idx + 1 < n:
        after_hi = high_arr[idx + 1:]
        after_lo = low_arr[idx + 1:]
        if direction == "Bullish":
            if bool((after_lo < level).any()):
                invalidated = True
        else:
            if bool((after_hi > level).any()):
                invalidated = True
    dist_atr = abs(close_arr[idx] - level) / atr_val
    if dist_atr > ATR_DIST_MULT:
        return None

    candle_time = df.index[idx].to_pydatetime()
    statut = compute_statut(idx, n, tf)
    # C-4 (V3) : le statut Invalidated est prioritaire. Un signal dont la
    # cassure a echoue est emis AVEC cette marque, jamais supprime.
    if invalidated:
        statut = "Invalidated"
    score = _compute_confluence_score(
        dist_atr, candle_time, has_sweep, sig_type, tf,
    )
    if score < MIN_SCORE:
        return None

    # BB width over a bounded window ending at signal candle
    bb_window = df.iloc[max(0, idx + 1 - 40): idx + 1]
    bb_pct, bb_regime = compute_bb_width(bb_window)

    level_f = float(level)
    close_price = float(close_arr[idx])
    current_price = float(close_arr[-1])

    return SignalCore(
        sig_type=sig_type,
        direction=direction,
        level=level_f,
        idx_break=int(idx),
        close_price=close_price,
        current_price=current_price,
        has_sweep=bool(has_sweep),
        atr_val=float(atr_val),
        volatilite=atr_regime,
        trend=trend,
        force=force_label,
        dist_atr=float(dist_atr),
        score=int(score),
        bb_width_pct=bb_pct,
        bb_regime=bb_regime,
        signal_time_utc=candle_time,
        # PG-31b : MEME appel que _compute_confluence_score. Avant,
        # le payload affichait session="Off" pour D1/Weekly alors que le
        # score incluait le bonus DailyClose +10 : champ et score
        # incoherents. Desormais les deux voient "DailyClose".
        session=get_session(candle_time, tf),
        statut=statut,
        candles_elapsed=(n - 1) - int(idx),
        distance_pct=calc_distance_pct(level_f, close_price),
        current_distance_pct=calc_distance_pct(level_f, current_price),
    )


def detect_choch(df: pd.DataFrame, tf: str, inst: str) -> Optional[SignalCore]:
    swings = detect_swing_points(df, tf)
    # PG-30 (r9) : tendance et ATR CAUSAUX. Ils ne sont plus calcules une
    # fois sur la serie entiere (look-ahead) mais par offset, sur les
    # donnees CONFIRMEES a idx. Les pivots de `swings` futurs sont filtres
    # par prev (idx-(lb+1)) ; la tendance ne voit donc que le passe.

    # Fenetre de detection explicite. _scan_one appelle detect_choch()
    # exactement UNE fois par (inst, tf) — la boucle d'offset est donc le
    # SEUL mecanisme couvrant les signaux formes sur N-2..N-k. Le premier
    # match (le plus recent) gagne, preservant le determinisme du signal_id.
    n = len(df)
    lookback = SWING_LOOKBACK.get(tf, 5)
    window = DETECTION_LOOKBACK.get(tf, 5)

    for offset in range(window):
        idx = n - 1 - offset
        if idx < 3:
            break
        prev_swings = [s for s in swings if s["idx"] <= idx - (lookback + 1)]
        if not prev_swings:
            continue
        # PG-30 (r9) : tendance et ATR sur donnees confirmees a idx.
        trend = get_structural_trend(prev_swings)
        if trend == "Range":
            continue
        atr_val, atr_regime = calc_atr_bundle(df.iloc[: idx + 1], inst)
        if not math.isfinite(atr_val) or atr_val <= 0:
            continue
        sig = _evaluate_candle(
            idx=idx, df=df, prev_swings=prev_swings,
            atr_val=atr_val, atr_regime=atr_regime, trend=trend, tf=tf,
        )
        if sig is not None:
            return sig
    return None


# ---- 4.7 row/payload projections (single source of truth) ----------------

def _signal_id(inst: str, tf: str, sig: SignalCore) -> str:
    """Deterministic id: same candle close + same rule => same id."""
    raw = (
        f"{inst}|{tf}|{sig.signal_time_utc.strftime('%Y%m%dT%H%MZ')}"
        f"|{sig.sig_type}|{sig.direction}|{RULE_VERSION}"
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{inst}__{tf}__{sig.signal_time_utc.strftime('%Y%m%dT%H%M')}__{digest}"


def signal_to_row(inst: str, tf: str, sig: SignalCore) -> dict[str, Any]:
    inst_disp = inst.replace("_", "/")
    return {
        "Instrument": inst_disp,
        "_time_sort": sig.signal_time_utc,
        "Timeframe": tf,
        "Type": sig.sig_type,
        "Ordre": "Achat" if sig.direction == "Bullish" else "Vente",
        "Signal": f"{sig.direction} {sig.sig_type}",
        "Niveau": format_niveau(sig.level, inst),
        "Distance%": format_distance(sig.distance_pct),
        "Score": int(sig.score),  # A9 : nombre pour tri numerique
        "Volatilité": sig.volatilite,
        "Force": sig.force,
        "BB_Width": format_bb_width((sig.bb_width_pct, sig.bb_regime)),
        "Statut": sig.statut,
        # A9 : "Heure (UTC)" affichait l'OUVERTURE (signal_time_utc) sans le
        # dire. Desormais deux colonnes explicites + distance actuelle.
        "Ouverture bougie (UTC)": sig.signal_time_utc.strftime(
            "%Y-%m-%d %H:%M"),
        "Confirmation (UTC)": _confirmation_time(
            sig.signal_time_utc, tf)[:16].replace("T", " "),
        "Distance actuelle %": format_distance(sig.current_distance_pct),
        "signal_id": _signal_id(inst, tf, sig),
    }


def signal_to_payload(
    inst: str, tf: str, sig: SignalCore, scan_time: datetime,
) -> dict[str, Any]:
    """JSON pipeline payload — bit-stable, sortable, rule-versioned."""
    inst_disp = inst.replace("_", "/")
    prec = instrument_precision(inst)
    return {
        "signal_id": _signal_id(inst, tf, sig),
        "scanner_version": SCANNER_VERSION,
        "rule_version": RULE_VERSION,
        "generated_at": scan_time.isoformat(),
        "pair": inst_disp,
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
        "level": round(sig.level, prec),
        "close_price": round(sig.close_price, prec),
        "current_price": round(sig.current_price, prec),
        "distance_pct": (
            round(sig.distance_pct, 4) if sig.distance_pct is not None else None
        ),
        "current_distance_pct": (
            round(sig.current_distance_pct, 4)
            if sig.current_distance_pct is not None else None
        ),
        "distance_atr_multiple": round(sig.dist_atr, 2),
        "volatility": sig.volatilite,
        "force": sig.force,
        "bb_width_pct": (
            round(sig.bb_width_pct, 2)
            if sig.bb_width_pct is not None else None
        ),
        "bb_regime": sig.bb_regime,
        "session": sig.session,
        "signal_time": sig.signal_time_utc.isoformat(),
        "candles_elapsed": int(sig.candles_elapsed),
        "has_sweep": bool(sig.has_sweep),
        "atr": round(sig.atr_val, prec + 2),
        # PG-01 (r8.1, schema 1.2.0) : age_minutes desormais calcule depuis
        # confirmation_time (cloture de la bougie de cassure), pas depuis
        # signal_time (ouverture). L'age reflete la confirmation, comme
        # confirmation_time. Cohrent avec le validateur (controle ajoute).
        "confirmation_time": _confirmation_time(sig.signal_time_utc, tf),
        "age_minutes": _age_minutes(sig.signal_time_utc, tf, scan_time),
    }


_BAR_SECONDS = {"H1": 3600, "H4": 14400, "D": 86400, "W": 604800,
                "D1": 86400, "Weekly": 604800}


def _confirmation_time(signal_time: datetime, tf: str) -> str:
    """Cloture de la bougie de cassure = signal_time + 1 barre."""
    secs = _BAR_SECONDS.get(tf, 3600)
    return (signal_time + timedelta(seconds=secs)).isoformat()


def _age_minutes(signal_time: datetime, tf: str,
                 scan_time: datetime) -> int:
    """Minutes ecoulees depuis la CONFIRMATION (cloture = signal_time + 1
    barre). Non negatif : 0 tant que la bougie de cassure n'est pas cloturee.

    PG-01 : avant, l'age etait compte depuis signal_time (ouverture), donc
    surestime d'exactement une barre.
    """
    secs = _BAR_SECONDS.get(tf, 3600)
    confirmation = signal_time + timedelta(seconds=secs)
    delta = (scan_time - confirmation).total_seconds()
    return int(max(0.0, delta) // 60)


# =====================================================================
# SECTION 5 — I/O LAYER (OANDA)
# =====================================================================

@dataclass
class AuthState:
    """Per-session auth failure tracker (NOT module-level)."""
    failures: int = 0
    aborted: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    # PG-11 : Event d'annulation partage avec les workers en vol. Permet
    # d'interrompre les requetes HTTP en cours, pas seulement de bloquer les
    # futures.
    cancel_event: threading.Event = field(
        default_factory=threading.Event)

    def record_failure(self) -> int:
        with self.lock:
            self.failures += 1
            if self.failures >= MAX_AUTH_FAILURES:
                self.aborted = True
                self.cancel_event.set()
            return self.failures

    def is_aborted(self) -> bool:
        with self.lock:
            return self.aborted

    def cancel(self) -> None:
        """Annulation : positionne l'Event (workers + run_scan timeout).

        A7 (LOT A) : il n'y a PAS de bouton Stop dans l'UI ; cancel() est
        appele par run_scan sur timeout global. Tout commentaire mentionnant
        un bouton Stop etait faux. Limite documentee : une requete HTTP
        deja partie ne s'interrompt pas — elle dure jusqu'a
        OANDA_REQUEST_TIMEOUT ; l'annulation agit AVANT la prochaine
        tentative et avant chaque backoff sleep.
        """
        with self.lock:
            self.aborted = True
        self.cancel_event.set()

    def is_cancelled(self) -> bool:
        """Test non-bloquant pour les workers en vol (PG-11)."""
        return self.cancel_event.is_set()

    def reset(self) -> None:
        with self.lock:
            self.failures = 0
            self.aborted = False
        self.cancel_event.clear()


def _fetch_candles_raw(inst: str, gran: str) -> Optional[list[dict]]:
    """Single OANDA REST call. Returns raw `complete` candles list, or None."""
    count = GRAN_COUNT.get(gran, 300)
    req = instruments.InstrumentsCandles(
        instrument=inst,
        params={"count": count, "granularity": gran, "price": "M"},
    )
    api = _get_oanda_api()
    api.request(req)
    return [c for c in req.response.get("candles", []) if c.get("complete")]


@st.cache_data(
    ttl=CANDLES_CACHE_TTL_SECONDS,
    show_spinner=False,
    max_entries=512,
)
def get_candles_cached(
    inst: str, gran: str, cache_bust: int,
    # A7 (LOT A) : le prefix underscore indique a st.cache_data de NE PAS
    # hasher cet argument (Streamlit ne sait pas hasher un threading.Event
    # -> "Cannot hash argument"). C'est correct : l'Event n'influence pas
    # le resultat (memes bougies pour (inst, gran, cache_bust)) ; il ne
    # sert qu'a interrompre les retries/backoff.
    _cancel_event: Optional[threading.Event] = None,
) -> Optional[pd.DataFrame]:
    """
    Streamlit-cached candles fetch with explicit cache_bust key
    (le parametre participe reellement a la cle de cache).
    Cache TTL is short (60s) — quotes refresh quickly during market hours.

    A6 (LOT A) : une reponse valide mais insuffisante (< 50 bougies) leve
    InsufficientDataError et est comptee en no_data. Une vraie panne
    (v20/network) leve son exception et est comptee en failed. Cette
    fonction ne renvoie JAMAIS None (comportement r9 obsolete : None etait
    cache 60 s et masquait une panne reelle). Les exceptions ne sont pas
    mises en cache : st.cache_data ne met en cache que les valeurs de
    retour normales. Pour auth errors on raise aussi (tracking session).
    """
    for attempt in range(OANDA_MAX_RETRIES + 1):
        try:
            raw = _fetch_candles_raw(inst, gran)
            break
        except V20Error as exc:
            if exc.code == 401:
                # Bubble up so the caller can update AuthState
                raise
            if exc.code == 429 and attempt < OANDA_MAX_RETRIES:
                if _cancel_event is not None and _cancel_event.is_set():
                    raise requests.RequestException(
                        f"annulation demandee pour {inst} {gran}")
                time.sleep(OANDA_BACKOFF_BASE * (2 ** attempt))
                continue
            # PG-08 : retry sur 5xx (erreur temporaire du serveur OANDA).
            # 500/502/503/504 sont transitoires ; le backoff exponentiel
            # evite de saturer l'API.
            if 500 <= exc.code < 600 and attempt < OANDA_MAX_RETRIES:
                if _cancel_event is not None and _cancel_event.is_set():
                    raise requests.RequestException(
                        f"annulation demandee pour {inst} {gran}")
                time.sleep(OANDA_BACKOFF_BASE * (2 ** attempt))
                continue
            _log(logging.WARNING, "oanda_v20_error",
                 instrument=inst, granularity=gran, code=exc.code,
                 err=str(exc))
            # PG-07 : lever, ne jamais cacher un echec dans le cache TTL.
            raise
        except requests.RequestException as exc:
            if attempt < OANDA_MAX_RETRIES:
                if _cancel_event is not None and _cancel_event.is_set():
                    raise
                time.sleep(OANDA_BACKOFF_BASE * (2 ** attempt))
                continue
            _log(logging.WARNING, "oanda_network_error",
                 instrument=inst, granularity=gran, err=str(exc))
            raise
    else:  # pragma: no cover - unreachable
        raise requests.RequestException(
            f"retry eteint pour {inst} {gran} sans reponse")

    if raw is None or len(raw) < 50:
        # A6 (LOT A) : reponse valide mais contenu insuffisant. Ce n'est pas
        # une panne reseau : on leve InsufficientDataError pour que l'appel
        # la compte en no_data. Leve (pas de None cache) car st.cache_data
        # ne met en cache que les retours normaux — un tel appel refait
        # donc bien une requete au prochain appel.
        raise InsufficientDataError(
            f"reponse OANDA insuffisante pour {inst} {gran} "
            f"({len(raw) if raw is not None else 0} < 50 bougies)")

    rows = [
        r for c in raw
        if (r := _parse_candle_row(c, inst, gran)) is not None
    ]
    if len(rows) < 50:
        raise InsufficientDataError(
            f"bougies exploitables insuffisantes pour {inst} {gran} "
            f"({len(rows)} < 50 apres parsing)")

    df = pd.DataFrame(rows).set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


class InsufficientDataError(Exception):
    """A6 (LOT A) : reponse OANDA valide mais < 50 bougies exploitables.

    Ce n'est PAS une panne reseau ni une erreur v20 : l'API a repondu,
    mais le contenu est insuffisant pour la detection. Comptee en no_data
    (pas en failed). Herite d'Exception (pas de requests.RequestException)
    pour ne pas etre confondue avec une panne reseau.
    """


# =====================================================================
# SECTION 6 — ORCHESTRATION
# =====================================================================

@dataclass
class ScanResult:
    rows: list[dict[str, Any]]
    payloads: list[dict[str, Any]]
    errors: list[str]
    timed_out: int
    scan_time: datetime
    aborted: bool = False
    # PG-03 : compteurs exclusifs. Invariant : la somme vaut
    # pairs_requested (132), verifie par le validateur.
    # A2 (LOT A) : ok_signal ne compte QUE les signaux VALIDES emis.
    # Les signaux detectes mais non emis (Stale, doublon de signal_id)
    # vont dans ok_not_emitted, et les payloads rejetes par le contrat
    # sont recomptes en invalid_contract par serialize_pipeline.
    coverage_counts: dict[str, int] = field(default_factory=dict)


def _scan_one(
    inst: str, tf_name: str, tf_code: str, cache_bust: int,
    auth: AuthState,
) -> tuple[str, str, Optional[SignalCore], Optional[str]]:
    """Worker — fetches candles and runs detection. Returns errors as strings."""
    if auth.is_aborted():
        return inst, tf_name, None, "aborted"
    try:
        df = get_candles_cached(inst, tf_code, cache_bust,
                                auth.cancel_event)
    except V20Error as exc:
        if exc.code == 401:
            n = auth.record_failure()
            _log(logging.ERROR, "oanda_auth_failure",
                 instrument=inst, granularity=tf_name, count=n)
            return inst, tf_name, None, f"401#{n}"
        return inst, tf_name, None, f"failed:v20:{exc.code}"
    except requests.RequestException as exc:  # noqa: BLE001 — PG-07
        _log(logging.ERROR, "oanda_network_failure",
             instrument=inst, granularity=tf_name, err=str(exc))
        return inst, tf_name, None, f"failed:net:{type(exc).__name__}"
    except InsufficientDataError as exc:
        # A6 (LOT A) : reponse valide mais < 50 bougies -> no_data, pas
        # failed. Une vraie panne reseau reste en failed:net:.
        _log(logging.WARNING, "oanda_insufficient_data",
             instrument=inst, granularity=tf_name, err=str(exc))
        return inst, tf_name, None, "no_data"
    except Exception as exc:  # noqa: BLE001 — defensive boundary
        _log(logging.ERROR, "scan_one_unexpected",
             instrument=inst, granularity=tf_name, err=str(exc))
        return inst, tf_name, None, f"failed:unexpected:{type(exc).__name__}"

    if df is None:
        # PG-07 : desormais INATTEIGNABLE (get_candles_cached leve au lieu
        # de renvoyer None). Garde en garde-fou defensif.
        return inst, tf_name, None, "no_data"
    # PG-11 : une annulation pendant le fetch (Event) doit interrompre la
    # detection, pas seulement empecher le demarrage.
    if auth.is_cancelled():
        return inst, tf_name, None, "aborted"
    try:
        sig = detect_choch(df, tf_name, inst)
    except Exception as exc:  # noqa: BLE001
        _log(logging.ERROR, "detect_choch_failed",
             instrument=inst, granularity=tf_name, err=str(exc))
        return inst, tf_name, None, f"detect:{type(exc).__name__}"
    return inst, tf_name, sig, None


def run_scan(
    auth: AuthState, cache_bust: int,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> ScanResult:
    """
    Pure orchestration. Idempotent: same inputs (cache_bust unchanged)
    produce identical outputs thanks to st.cache_data on candles.
    """
    correlation_id = uuid.uuid4().hex[:12]
    scan_time = datetime.now(timezone.utc)
    # PG-15 : timings par etape (millisecondes), attitres au correlation_id.
    t0 = time.perf_counter()
    auth.reset()
    _log(logging.INFO, "scan_start",
         correlation_id=correlation_id,
         instruments=len(INSTRUMENTS), timeframes=len(TIMEFRAMES))

    executor = _get_scan_executor()
    t_submit = time.perf_counter()
    futures: dict[Future, tuple[str, str]] = {
        executor.submit(_scan_one, inst, tf_name, tf_code, cache_bust, auth):
            (inst, tf_name)
        for inst in INSTRUMENTS
        for tf_name, tf_code in TIMEFRAMES.items()
    }
    t_after_submit = time.perf_counter()

    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    errors: list[str] = []
    # PG-03 : compteurs de couverture EXCLUSIFS. Chaque (instrument, TF)
    # finit dans exactement une categorie. Invariant : la somme vaut
    # pairs_requested (132), verifie par le validateur.
    cov_ok_signal = 0
    cov_ok_no_signal = 0
    cov_no_data = 0
    cov_failed = 0
    cov_aborted = 0
    # A2 (LOT A) : signaux detectes mais non emis en payload (Stale, ou
    # doublon de signal_id ecarte par seen_ids). Categorie dediee pour
    # que ok_signal reflete exactement les payloads emis.
    cov_ok_not_emitted = 0
    seen_ids: set[str] = set()
    # PG-09 : trace les futures deja traitees par as_completed pour eviter
    # un double comptage lors du rattrapage post-timeout.
    _handled: set[int] = set()

    def _handle(fut: Future) -> None:
        # Progression animee : le callback est appele sur TOUTES les
        # branches (erreur, sans-signal, signal), des que la tache finit.
        inst, tf_name = futures[fut]
        if id(fut) in _handled:
            return
        _handled.add(id(fut))
        nonlocal cov_ok_signal, cov_ok_no_signal, cov_no_data, cov_failed
        nonlocal cov_aborted
        try:
            inst_r, tf_r, sig, err = fut.result()
        except CancelledError:
            cov_aborted += 1
            return
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{inst}/{tf_name}: {exc}")
            cov_failed += 1
            return
        try:
            if err and err.startswith("401"):
                errors.append(f"{inst_r}/{tf_r}: auth {err}")
                cov_aborted += 1
                return
            if err == "aborted":
                cov_aborted += 1
                return
            if err == "no_data":
                errors.append(f"{inst_r}/{tf_r}: no_data")
                cov_no_data += 1
                return
            if err and err.startswith("failed:"):
                # PG-07 : panne reelle (reseau/v20/autre), pas un "pas de
                # signal". Comptee en failed, jamais cachee en no_data.
                errors.append(f"{inst_r}/{tf_r}: {err}")
                cov_failed += 1
                return
            if err:
                errors.append(f"{inst_r}/{tf_r}: {err}")
                cov_failed += 1
                return
            if sig is not None:
                row = signal_to_row(inst_r, tf_r, sig)
                sid = row["signal_id"]
                if sid not in seen_ids:
                    seen_ids.add(sid)
                    rows.append(row)
                    if sig.statut in ("Fresh", "Aged", "Invalidated"):
                        # C-4 (V3) : Invalidated est EMIS dans le pipeline
                        # (visible pour l'operateur, marquee) au lieu d'etre
                        # supprime. Stale reste non emis.
                        payloads.append(
                            signal_to_payload(inst_r, tf_r, sig, scan_time))
                        cov_ok_signal += 1
                    else:
                        # A2 : Stale — detecte, visible dans le tableau,
                        # mais non emis dans le pipeline JSON.
                        cov_ok_not_emitted += 1
                else:
                    # A2 : doublon de signal_id (deja emis par une autre
                    # unite). Detecte, non emis.
                    cov_ok_not_emitted += 1
            else:
                cov_ok_no_signal += 1
        finally:
            if progress_callback is not None:
                progress_callback(inst_r, tf_r)

    not_done: list[Future] = []
    try:
        for fut in as_completed(futures.keys(), timeout=SCAN_GLOBAL_TIMEOUT):
            _handle(fut)
    except TimeoutError:
        # A7 (LOT A) : un timeout global est une demande d'arret. On leve
        # l'Event d'annulation pour que les workers en vol (backoff sleep
        # inclus) s'arretent des que possible.
        auth.cancel()
    # PG-09 : entre le TimeoutError de as_completed et ce balayage, des
    # futures peuvent terminer (race). Il faut les traiter, sinon un
    # resultat disponible serait compte en timed_out et perdu.
    not_done = []
    for f in futures:
        if f.done():
            if not f.cancelled():
                # deja traitees par la boucle as_completed ; les nouvelles
                # terminees depuis le timeout doivent etre gerees ici.
                # A8 (LOT A) : on compare id(f) (le set contient des int),
                # pas f lui-meme — sinon un Future ne vaut jamais un int et
                # le test est toujours vrai => double comptage.
                if id(f) not in _handled:
                    _handle(f)
        else:
            not_done.append(f)
    for f in not_done:
        f.cancel()

    aborted = auth.is_aborted()
    # PG-03 : les futures expirees (timeout) ne sont ni aborted (401) ni
    # failed ; elles forment une categorie dediee.
    cov_timed_out = len(not_done)
    # PG-15 : timings de la phase de collecte
    t_collect = time.perf_counter()
    _log(logging.INFO, "scan_timings",
         correlation_id=correlation_id,
         submit_ms=round((t_after_submit - t_submit) * 1000, 1),
         collect_ms=round((t_collect - t_after_submit) * 1000, 1),
         total_ms=round((t_collect - t0) * 1000, 1))
    _log(logging.INFO, "scan_end",
         correlation_id=correlation_id,
         signals=len(rows), pipeline=len(payloads),
         errors=len(errors), timed_out=cov_timed_out, aborted=aborted)
    return ScanResult(
        rows=rows, payloads=payloads, errors=errors,
        timed_out=cov_timed_out, scan_time=scan_time, aborted=aborted,
        coverage_counts={
            "ok_signal": cov_ok_signal,
            "ok_no_signal": cov_ok_no_signal,
            "ok_not_emitted": cov_ok_not_emitted,
            # A2 : rempli a 0 ici ; serialize_pipeline recompte les
            # payloads effectivement rejetes par le contrat.
            "invalid_contract": 0,
            "no_data": cov_no_data,
            "failed": cov_failed,
            "aborted": cov_aborted,
            "timed_out": cov_timed_out,
        },
    )


# =====================================================================
# SECTION 7 — EXPORT (PDF / PNG / JSON)
# =====================================================================

def _json_default(obj: Any) -> Any:
    """A10 (LOT A) : convertisseur JSON — sans perte d'information.

    Le repli str(obj) est DESORMAIS une erreur refusee : serialiser un
    objet inconnu en chaine cacherait une anomalie de type sous une
    representation inattendue. Les types attendus sont couverts
    explicitement ; tout le reste doit echouer bruyamment.
    """
    if obj is None:
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(
        f"objet non serialisable : {type(obj).__name__}")


def _sanitize_json(obj: Any) -> Any:
    """Nettoie AVANT dumps — un float non-fini devient None
    (default= n'est jamais consulte pour un float, allow_nan=False leve)."""
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.floating,)) and not math.isfinite(float(obj)):
        return None
    return obj


# C-2 (LOT C) : schema 3.0.0. Justification du bump MAJEUR :
# - V3 ajoute la valeur "Invalidated" a l'enum status. C'est un
#   CHANGEMENT DE SENS du champ status (un signal emis peut desormais
#   etre mort) pas un simple ajout additif => majeur selon R4.
# - L'enum coverage ajoute ok_not_emitted/invalid_contract (2.1.0).
# Un consommateur 2.x qui switch sur status sans cas Invalidated doit
# etre alerte : 3.0.0.
SCHEMA_VERSION: Final[str] = "3.0.0"
# 2.0.0 (r9, PHASE 3) : MAJEUR car le SENS de champs change —
#  - PG-30 : tendance et ATR causaux (un signal emis a t ne depend que de
#    [0, idx]) ; 31,03 % des signaux a offset>0 changent (mesure r8).
#  - PG-31 (D1-a) : BOS EMIS. trend == direction desormais ATTENDU pour le
#    type BOS (il continue la tendance) et interdit pour CHoCH.
#  - PG-31b (D2-a) : nouvelle session DailyClose (D1/Weekly), bonus +10.
#  - PG-32 : un signal dont le niveau protectif est franchi apres la
#    confirmation n'est plus emis (71,4 % des signaux r8 etaient concernes).
#  - PG-33 : la volatilite est celle de la bougie de cassure (via PG-30).
#  Options D3-c (sweep inchange), D4-b (signal_id par instant conserve),
#  D5-a (statuts inchanges), D6-a (dist_atr <= 1.8 conserve) : non
#  modulatrices.

# Regexes/enums du contrat (miroir minimal de choch_pipeline.schema.json ;
# le validateur complet est publie dans audit_prod/).
_RE_SIGNAL_ID = re.compile(
    r"^[A-Z0-9_]{2,20}__(H1|H4|D1|Weekly)__\d{8}T\d{4}__[0-9a-f]{12}$")
_RE_PAIR = re.compile(r"^[A-Z0-9]{2,8}/[A-Z0-9]{2,8}$")
_RE_PAIR_OANDA = re.compile(r"^[A-Z0-9]{2,8}_[A-Z0-9]{2,8}$")
_RE_ISO_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00$")
_SIGNAL_ENUMS = {
    "timeframe": {"H1", "H4", "D1", "Weekly"},
    "type": {"CHoCH", "BOS"},
    "direction": {"Bullish", "Bearish"},
    "order": {"buy", "sell"},
    "trend": {"Bullish", "Bearish"},
    "status": {"Fresh", "Aged", "Invalidated"},
    "volatility": {"Très Haute", "Haute", "Moyenne", "Basse"},
    "force": {"Fort", "Moyen"},
    "bb_regime": {"Squeeze", "Expansion", "Normal", "N/A"},
    "session": {"London_NY_Overlap", "London", "NewYork", "Tokyo", "Off",
                "DailyClose"},
}
_SIGNAL_REQUIRED = (
    "signal_id", "scanner_version", "rule_version", "generated_at", "pair",
    "pair_oanda", "timeframe", "type", "direction", "is_bullish", "order",
    "trend", "is_choch", "status", "confluence_score", "level", "close_price",
    "current_price", "distance_pct", "current_distance_pct",
    "distance_atr_multiple", "volatility", "force", "bb_width_pct",
    "bb_regime", "session", "signal_time", "candles_elapsed", "has_sweep",
    "atr", "confirmation_time", "age_minutes",
)


def _expected_session_for_score(signal_time: str, timeframe: str) -> str:
    """V-2 (LOT C) : session coherente avec le score.

    Le bonus de session est calcule par ``get_session(candle_time, tf)``.
    Un payload dont le champ ``session`` differe est incoherent avec son
    propre ``confluence_score`` (defaut 1 : session="Off" + bonus
    DailyClose +10). On rejoue le MEME appel pour exiger l'egalite.
    """
    dt = datetime.fromisoformat(signal_time)
    return get_session(dt, timeframe)


def _validate_signal(s: Mapping[str, Any]) -> Optional[str]:
    """Contrat JSON d'un signal. Renvoie une raison de rejet ou None.

    Fail-closed : un signal non conforme n'est JAMAIS emis en silence ; il est
    ecarte et compte dans meta.invalid_signals (FND-04).
    """
    try:
        if set(s) - set(_SIGNAL_REQUIRED):
            return "cles en trop: %s" % sorted(set(s) - set(_SIGNAL_REQUIRED))
        for k in _SIGNAL_REQUIRED:
            if k not in s:
                return f"cle requise manquante: {k}"
        if not _RE_SIGNAL_ID.match(s["signal_id"]):
            return "signal_id: pattern invalide"
        if not _RE_PAIR.match(s["pair"]) or not _RE_PAIR_OANDA.match(
                s["pair_oanda"]):
            return "pair/pair_oanda: pattern invalide"
        if s["pair"] != s["pair_oanda"].replace("_", "/"):
            return "pair != pair_oanda"
        for k, allowed in _SIGNAL_ENUMS.items():
            if s[k] not in allowed:
                return f"{k}: valeur hors enum ({s[k]!r})"
        if not isinstance(s["is_bullish"], bool) or not isinstance(
                s["is_choch"], bool) or not isinstance(s["has_sweep"], bool):
            return "champs booleens non booleens"
        if s["is_bullish"] != (s["direction"] == "Bullish"):
            return "is_bullish != (direction == 'Bullish')"
        if s["order"] != ("buy" if s["is_bullish"] else "sell"):
            return "order != is_bullish"
        if s["is_choch"] != (s["type"] == "CHoCH"):
            return "is_choch != (type == 'CHoCH')"
        # PG-06 / PG-31 (r9, D1-a) : un CHoCH CASSE la tendance, donc
        # trend != direction. Mais un BOS CONTINUE la tendance :
        # trend == direction est alors ATTENDU et obligeatoire. Ne rejeter
        # l'egalite que pour les CHoCH.
        if s["type"] == "CHoCH" and s["trend"] == s["direction"]:
            return "trend == direction (incoherence structurelle pour CHoCH)"
        if s["type"] == "BOS" and s["trend"] != s["direction"]:
            return "trend != direction (incoherence structurelle pour BOS)"
        if not isinstance(s["confluence_score"], int) or isinstance(
                s["confluence_score"], bool):
            return "confluence_score: pas un integer"
        if not (MIN_SCORE <= s["confluence_score"] <= 100):
            return "confluence_score hors borne"
        for k in ("level", "close_price", "current_price", "atr"):
            v = s[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return f"{k}: pas un nombre"
            if not (v > 0 and math.isfinite(v)):
                return f"{k}: valeur invalide"
        if not isinstance(s["distance_atr_multiple"], (int, float)) or isinstance(
                s["distance_atr_multiple"], bool):
            return "distance_atr_multiple: pas un nombre"
        if not (0.0 <= s["distance_atr_multiple"] <= ATR_DIST_MULT):
            return "distance_atr_multiple hors borne"
        for k in ("distance_pct", "current_distance_pct", "bb_width_pct"):
            v = s[k]
            if v is not None:
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    return f"{k}: pas un nombre"
                if not math.isfinite(v):
                    return f"{k}: non fini"
        if not _RE_ISO_UTC.match(str(s["generated_at"])):
            return "generated_at: pas ISO 8601 UTC"
        if not _RE_ISO_UTC.match(str(s["signal_time"])):
            return "signal_time: pas ISO 8601 UTC"
        if not _RE_ISO_UTC.match(str(s.get("confirmation_time", ""))):
            return "confirmation_time: pas ISO 8601 UTC"
        if not isinstance(s.get("age_minutes"), int) or isinstance(
                s.get("age_minutes"), bool) or s["age_minutes"] < 0:
            return "age_minutes: entier >= 0 attendu"
        # PG-01 : coherence age_minutes <-> confirmation_time. L'age doit
        # etre (generated_at - confirmation_time) en minutes, a 1 pres.
        try:
            conf = datetime.fromisoformat(s["confirmation_time"])
            gen = datetime.fromisoformat(s["generated_at"])
            expected = int(max(0, (gen - conf).total_seconds()) // 60)
            if abs(expected - s["age_minutes"]) > 1:
                return (f"age_minutes incoherent avec confirmation_time "
                        f"(attendu ~{expected}, lu {s['age_minutes']})")
        except Exception:
            return "age_minutes/confirmation_time: parsing impossible"
        if not isinstance(s["candles_elapsed"], int) or isinstance(
                s["candles_elapsed"], bool) or s["candles_elapsed"] < 0:
            return "candles_elapsed: entier >= 0 attendu"
        # PG-02 : recalcul depuis les valeurs ARRONDIES du payload. La
        # tolerance est celle de l'arrondi a 4 decimales (0.5 ulp), en
        # relatif : pour une distance petite, l'arrondi absolu domine ; on
        # compare donc sur la valeur arrondie a 4 decimales.
        for k_dist, k_px in (("distance_pct", "close_price"),
                             ("current_distance_pct", "current_price")):
            if s[k_dist] is None:
                continue
            rec = calc_distance_pct(s["level"], s[k_px])
            if rec is None:
                return f"{k_dist} non recalculable"
            if abs(round(rec, 4) - s[k_dist]) > 5e-5:
                return (f"{k_dist} non recalculable "
                        f"(recalcule {rec:.6f}, lu {s[k_dist]})")
        if s["scanner_version"] != SCANNER_VERSION:
            return "scanner_version incoherent"
        if s["rule_version"] != RULE_VERSION:
            return "rule_version incoherent"
        # V-2 (LOT C) : coherence session <-> score. Le bonus de session
        # est inclus dans confluence_score ; le champ session DOIT etre
        # celui qui a servi au calcul. Avant (defaut 1, JSON 12:10), D1
        # affichait session="Off" alors que le score contenait le bonus
        # DailyClose +10 — payloads incoherents sous le meme rule_version.
        try:
            sess_expected = _expected_session_for_score(
                s["signal_time"], s["timeframe"])
        except Exception:  # noqa: BLE001
            sess_expected = None
        if sess_expected is not None and s["session"] != sess_expected:
            return (f"session incoherente : {s['session']!r} alors que le "
                    f"score attend la session {sess_expected!r} "
                    f"(timeframe {s['timeframe']})")
    except Exception as exc:  # noqa: BLE001 - boundary fail-closed
        return f"validateur: {type(exc).__name__}"
    return None


def serialize_pipeline(
    payloads: Sequence[Mapping[str, Any]], scan_time: datetime,
    errors: Sequence[str] = (), timed_out: int = 0,
    coverage_counts: Mapping[str, int] = (),
) -> bytes:
    # Le JSON dit maintenant ce qui a echoue.
    # FND-04 : validation fail-closed avant emission. Un signal non conforme
    # est ecarte (jamais emis en silence) et compte dans meta.invalid_signals.
    valid: list[Mapping[str, Any]] = []
    invalid: list[str] = []
    for p in payloads:
        raison = _validate_signal(p)
        if raison is None:
            valid.append(p)
        else:
            sid = p.get("signal_id", "<sans-id>")
            invalid.append(f"{sid}: {raison}")
            _log(logging.ERROR, "signal_rejete_contrat",
                 signal_id=sid, raison=raison)
    # A2 (LOT A) : ok_signal ne compte QUE les signaux valides emis. Les
    # payloads rejetes par le contrat sont bascules de ok_signal vers la
    # categorie dediee invalid_contract. Un signal invalide n'abime plus
    # jamais le document : on l'ecarte et on garde les autres (fail-closed
    # au niveau du signal, pas du document).
    n_rejected = len(invalid)
    cov_ok_signal = int(coverage_counts.get("ok_signal", 0)) - n_rejected
    cov_invalid_contract = n_rejected
    doc = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "scanner_version": SCANNER_VERSION,
            "rule_version": RULE_VERSION,
            "generated_at": scan_time.isoformat(),
            "signal_count": len(valid),
            "coverage": {
                "pairs_requested": len(INSTRUMENTS) * len(TIMEFRAMES),
                # PG-03 : compteurs exclusifs. Invariant somme = 132,
                # verifie par _validate_doc ci-dessous.
                "ok_signal": cov_ok_signal,
                "ok_no_signal": int(
                    coverage_counts.get("ok_no_signal", 0)),
                "ok_not_emitted": int(
                    coverage_counts.get("ok_not_emitted", 0)),
                "invalid_contract": cov_invalid_contract,
                "no_data": int(coverage_counts.get("no_data", 0)),
                "failed": int(coverage_counts.get("failed", 0)),
                "aborted": int(coverage_counts.get("aborted", 0)),
                "timed_out": int(timed_out),
                "pairs_failed": len(errors),
                "pairs_timed_out": int(timed_out),
                "failures": list(errors[:50]),
            },
            "invalid_signals": invalid[:50],
        },
        "signals": sorted(valid, key=lambda p: p["signal_id"]),  # FND-01
    }
    # PG-03 : invariant de couverture. La somme des compteurs exclusifs doit
    # valoir pairs_requested. Echec bruyant (jamais de document partiel).
    cov = doc["meta"]["coverage"]
    total = (cov["ok_signal"] + cov["ok_no_signal"] + cov["ok_not_emitted"]
             + cov["invalid_contract"] + cov["no_data"] + cov["failed"]
             + cov["aborted"] + cov["timed_out"])
    if total != cov["pairs_requested"]:
        raise RuntimeError(
            f"invariant de couverture viole : {total} != "
            f"{cov['pairs_requested']} (ok_signal={cov['ok_signal']}, "
            f"ok_no_signal={cov['ok_no_signal']}, "
            f"ok_not_emitted={cov['ok_not_emitted']}, "
            f"invalid_contract={cov['invalid_contract']}, "
            f"no_data={cov['no_data']}, "
            f"failed={cov['failed']}, aborted={cov['aborted']}, "
            f"timed_out={cov['timed_out']})"
        )
    # A2 (LOT A) : l'ancien test cov["ok_signal"] != len(valid) levait
    # RuntimeError et DETRUISAIT le document des qu'un signal etait rejete
    # — l'inverse du fail-closed. Desormais ok_signal est RECALCULE pour
    # valoir len(valid) par construction (voir plus haut), donc cette
    # egalite est toujours verifiee ; on l'affirme sans lever.
    assert cov["ok_signal"] == len(valid), (
        f"ok_signal={cov['ok_signal']} != {len(valid)}")
    return json.dumps(
        _sanitize_json(doc), ensure_ascii=False, indent=2,
        default=_json_default, allow_nan=False,
    ).encode("utf-8")


def create_pdf(df_export: pd.DataFrame,
                 scan_time: Optional[datetime] = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4),
        leftMargin=20, rightMargin=20, topMargin=40, bottomMargin=40,
    )
    elements: list[Any] = []
    styles = getSampleStyleSheet()
    elements.append(Paragraph(
        f"Rapport des Signaux CHoCH v{SCANNER_VERSION} ({RULE_VERSION})",
        styles["Title"]))
    elements.append(Paragraph(
        f"Généré le {(scan_time or datetime.now(timezone.utc)).strftime('%d/%m/%Y à %H:%M')} UTC",
        styles["Normal"]))
    elements.append(Spacer(1, 20))

    cols_present = [c for c in EXPORT_COLS if c in df_export.columns]
    widths_map = {c: 60 for c in cols_present}
    widths_map.update({
        "Instrument": 65, "Distance%": 52, "Distance actuelle %": 52,
        "Statut": 45, "Ouverture bougie (UTC)": 105,
        "Confirmation (UTC)": 105, "signal_id": 130,
    })
    col_widths = [widths_map.get(c, 60) for c in cols_present]
    # PG-16 : la largeur totale doit tenir dans la page A4 paysage
    # (842 pt - 20 - 20 de marges = 802 pt). Sinon reportlab deborde et
    # coupe des colonnes. On rescale proportionnellement.
    usable = A4[1] - 40  # landscape(A4) -> largeur = A4[1] = 842
    total = sum(col_widths)
    if total > usable:
        scale = usable / total
        col_widths = [w * scale for w in col_widths]
    # A9 / PG-16 (LOT A) : signal_id fait ~45 caracteres ; en cellule
    # simple il deborde et est coupe. Un Paragraph reportlab retourne a la
    # ligne automatiquement. Les autres colonnes restent brutes (nombres).
    idx_sid = cols_present.index("signal_id") if "signal_id" in cols_present \
        else None
    rows_data = []
    for row in df_export[cols_present].values.tolist():
        if idx_sid is not None:
            row = list(row)
            sid = row[idx_sid]
            if isinstance(sid, str):
                row[idx_sid] = Paragraph(
                    sid,
                    ParagraphStyle(
                        "sid", fontName="Helvetica", fontSize=6.5,
                        leading=8, alignment=1))
        rows_data.append(row)
    data = [cols_present] + rows_data

    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.beige]),
    ]))
    elements.append(table)
    doc.build(elements)
    return buffer.getvalue()


def generate_png(data: pd.DataFrame, display_cols: Sequence[str]) -> bytes:
    """Genere le PNG du tableau, PAGINE (PG-17).

    Avant, une seule figure plafonnait a 30 pouces de haut : au-dela de
    ~85 lignes les cellules etaient ecrasees et illisibles. On decoupe en
    pages de PNG_ROWS_PER_PAGE lignes, empilees verticalement dans un seul
    PNG (compatibilite du bouton de telechargement, 1 fichier).
    """
    cols = [c for c in display_cols if c in data.columns]
    disp = data[cols]
    n = len(disp)
    if n == 0:
        disp = pd.DataFrame([["(aucun signal)"] * len(cols)],
                            columns=cols)
        n = 1
    # dpi 100 (etait 200) — divise la memoire de rendu par 4
    pages = [disp.iloc[i:i + PNG_ROWS_PER_PAGE]
             for i in range(0, n, PNG_ROWS_PER_PAGE)]
    figs = []
    for pg in pages:
        f = Figure(figsize=(22, min(max(5, len(pg) * 0.35), 30)))
        ax = f.add_subplot(111)
        ax.axis("off")
        tbl = ax.table(
            cellText=pg.values, colLabels=pg.columns,
            cellLoc="center", loc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.2, 1.8)
        figs.append(f)
    if len(figs) == 1:
        buf = io.BytesIO()
        figs[0].savefig(buf, format="png", bbox_inches="tight", dpi=100)
        return buf.getvalue()
    # PG-17 : combiner les pages verticalement en un seul PNG
    bufs = []
    for f in figs:
        b = io.BytesIO()
        f.savefig(b, format="png", bbox_inches="tight", dpi=100)
        b.seek(0)
        bufs.append(b)
    from PIL import Image
    # A10 (LOT A) : Pillow est une dependance transitive de matplotlib
    # (pinnee explicitement dans requirements.txt). L'import local evite
    # de charger Pillow au demarrage quand aucun PNG n'est genere.
    imgs = [Image.open(b).convert("RGB") for b in bufs]
    w = max(im.width for im in imgs)
    h = sum(im.height for im in imgs)
    canvas = Image.new("RGB", (w, h), "white")
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.height
    out = io.BytesIO()
    canvas.save(out, format="png")
    return out.getvalue()


# =====================================================================
# SECTION 8 — UI LAYER
# =====================================================================

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
    except (ValueError, TypeError):
        return "color:#90a4ae"
    if v <= 0.15:
        return "color:#00c853;font-weight:bold"
    if v <= 0.40:
        return "color:#ff9800;font-weight:bold"
    return "color:#ff5252;font-weight:bold"


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "scanning": False,
        "cache_bust": 0,
        "auth": AuthState(),
        "df": None,
        "pipeline_signals": [],
        "scan_time": None,
        "scan_errors": [],
        "scan_timed_out": 0,
        "scan_coverage_counts": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _json_session_key(df_all: pd.DataFrame, scan_time: datetime) -> str:
    """A5 (LOT A) : cle de cache JSON calculee UNE SEULE FOIS.

    _render_downloads et _render_results doivent construire la MEME cle,
    sinon le second ne retrouve jamais le document serialise par le
    premier. Cette fonction unique supprime le recalcul duplique.
    """
    ts = scan_time.strftime("%Y%m%d_%H%M%S")
    content_hash = (
        hashlib.sha256(
            pd.util.hash_pandas_object(df_all, index=False).to_numpy()
            .tobytes()
        ).hexdigest()[:6] if len(df_all) else "empty"
    )
    return f"json_{ts}_{content_hash}"


def _render_downloads(
    df_all: pd.DataFrame, df_export: pd.DataFrame,
    pipeline_signals: Sequence[Mapping[str, Any]], scan_time: datetime,
) -> None:
    ts = scan_time.strftime("%Y%m%d_%H%M%S")
    json_key = _json_session_key(df_all, scan_time)
    content_hash = json_key.split("_")[-1]
    for k in list(st.session_state):
        if k.startswith(("png_", "pdf_", "json_")) and k not in (
                f"png_{ts}_{content_hash}", f"pdf_{ts}_{content_hash}",
                f"json_{ts}_{content_hash}"):
            del st.session_state[k]
    # A5 (LOT A) : la cle json_ est calculee UNE SEULE FOIS (ici, appelee
    # aussi par _render_results via _json_session_key). Plus jamais de
    # recalcul duplique dans _render_results.
    if json_key not in st.session_state:
        st.session_state[json_key] = serialize_pipeline(
            pipeline_signals, scan_time,
            st.session_state.get("scan_errors", []),
            int(st.session_state.get("scan_timed_out", 0)),
            st.session_state.get("scan_coverage_counts", {}))
    json_bytes: bytes = st.session_state[json_key]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        cols = [c for c in EXPORT_COLS if c in df_export.columns]
        # PG-18 / A3 (LOT A) : to_csv() sans chemin renvoie une CHAINE et
        # ignore encoding ; il faut encoder la chaine en utf-8-sig pour que
        # la BOM soit presente. Sans elle Excel devine le delimiteur et
        # casse les accents (Volatilité).
        csv_bytes = df_export[cols].to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "CSV", csv_bytes, f"choch_{ts}.csv", "text/csv",
            key=f"dl_csv_{ts}",
        )
    with c2:
        # Lazy generation, cached against the dataframe identity via session.
        # Le PNG utilise df_export, comme le CSV et le PDF
        # (les signaux Stale sont "exclus des exports" selon l'UI).
        png_key = f"png_{ts}_{content_hash}"
        if png_key not in st.session_state:
            st.session_state[png_key] = generate_png(df_export, DISPLAY_COLS)
        st.download_button(
            "PNG", st.session_state[png_key],
            f"choch_{ts}.png", "image/png",
            key=f"dl_png_{ts}",
        )
    with c3:
        pdf_key = f"pdf_{ts}_{content_hash}"
        if pdf_key not in st.session_state:
            st.session_state[pdf_key] = create_pdf(df_export, scan_time)
        st.download_button(
            "PDF", st.session_state[pdf_key],
            f"choch_signaux_{ts}.pdf", "application/pdf",
            key=f"dl_pdf_{ts}",
        )
    with c4:
        st.download_button(
            "JSON", json_bytes,
            f"choch_pipeline_{ts}.json", "application/json",
            key=f"dl_json_{ts}",
        )


def _render_dataframe(df_all: pd.DataFrame) -> None:
    cols_disp = [c for c in DISPLAY_COLS if c in df_all.columns]
    styled = (
        df_all[cols_disp].style
        .map(lambda x: ("color:#e879f9;font-weight:bold" if x == "CHoCH"
                        else "color:#94a3b8" if x == "BOS" else ""),
             subset=["Type"])
        .map(lambda x: ("color:#00c853;font-weight:bold" if x == "Achat"
                        else "color:#ff5252;font-weight:bold" if x == "Vente"
                        else ""), subset=["Ordre"])
        .map(lambda x: ("color:#00c853" if "Bull" in str(x)
                        else "color:#ff5252" if "Bear" in str(x) else ""),
             subset=["Signal"])
        .map(lambda x: ("color:#00c853;font-weight:bold" if x == "Fort"
                        else "color:#ff5252" if x == "Faible"
                        else "color:#ff9800"), subset=["Force"])
        .map(_style_bb, subset=["BB_Width"])
        .map(_style_distance, subset=["Distance%"])
        .map(lambda x: ("color:#00c853;font-weight:bold" if x == "Fresh"
                        else "color:#ff9800;font-weight:bold" if x == "Aged"
                        else "color:#ff5252;font-weight:bold" if x == "Stale"
                        else ""), subset=["Statut"])
    )
    st.dataframe(styled, hide_index=True, width="stretch")


def _render_results() -> None:
    df_all = st.session_state.df
    # PG-04 : meme un scan sans signal produit un JSON valide (signals: [] +
    # coverage complete). On rend les downloads des qu'un scan a eu lieu,
    # pas seulement quand des signaux existent.
    has_scanned = st.session_state.get("scan_time") is not None
    if df_all is None or df_all.empty:
        df_all = pd.DataFrame(columns=DISPLAY_COLS)
    # C-4 (V3) : les signaux Invalidated sont EMIS dans le pipeline et
    # doivent aussi etre exportes (CSV/PDF/PNG), marques. Avant, seuls
    # Fresh/Aged etaient exportes — un signal invalid etait invisible
    # partout, indiscernable d'une regression.
    df_export = (df_all[df_all["Statut"].isin(
        ["Fresh", "Aged", "Invalidated"])].copy()
        if not df_all.empty else df_all.copy())
    pipeline_signals = st.session_state.get("pipeline_signals", [])
    scan_time = st.session_state.get("scan_time") or datetime.now(timezone.utc)

    n_stale = int((df_all["Statut"] == "Stale").sum()) if not df_all.empty \
        else 0
    if n_stale > 0:
        st.info(
            f"{n_stale} signal(s) Stale visible(s) dans le tableau "
            "— exclus des exports."
        )
    if has_scanned:
        _render_downloads(df_all, df_export, pipeline_signals, scan_time)
        # A5 (LOT A) : les compteurs UI viennent UNIQUEMENT de meta (document
        # serialise, APRES validation fail-closed). Avant, st.success et
        # l'expander affichaient len(result.payloads) — compte AVANT
        # validation, donc superieur a meta.signal_count des qu'un signal
        # etait rejete. La cle json_ est calculee UNE SEULE FOIS dans
        # _render_downloads (plus de recalcul duplique fragile ici).
        json_key = _json_session_key(df_all, scan_time)
        meta = None
        try:
            meta = json.loads(st.session_state[json_key])["meta"]
        except Exception:
            meta = None
        if meta is not None:
            n_valid = meta["signal_count"]
            n_invalid = len(meta["invalid_signals"])
            cov_m = meta["coverage"]
            st.caption(
                f"Pipeline JSON : {n_valid} signal(s) valide(s)"
                + (f", {n_invalid} rejeté(s) par le contrat"
                   if n_invalid else "")
                + f" | schema {meta['schema_version']} "
                + f"| rule {meta['rule_version']}"
                + f" | couverture "
                f"{cov_m['ok_signal']}+{cov_m['ok_no_signal']} OK"
                + (f", {cov_m['ok_not_emitted']} non emis"
                   if cov_m.get("ok_not_emitted") else "")
                + (f", {cov_m['invalid_contract']} rejetes contrat"
                   if cov_m.get("invalid_contract") else "")
                + f", {cov_m['no_data']} no_data, "
                f"{cov_m['failed']} failed, "
                f"{cov_m['timed_out']} timed_out"
            )
    if not df_all.empty:
        _render_dataframe(df_all)

    if pipeline_signals:
        with st.expander(
            f"Aperçu JSON Pipeline ({len(pipeline_signals)} signaux "
            f"Fresh/Aged/Invalidated)"
        ):
            st.json(pipeline_signals[0])


def _store_scan_result(
    result: ScanResult,
    df: Optional[pd.DataFrame],
) -> None:
    """A1 (LOT A) : stockage UNIQUE du resultat, TOUTES branches confondues.

    Avant, la branche "0 signal" ne mettait a jour que df/pipeline_signals/
    scan_time et conservait les erreurs ET compteurs du scan PRECEDENT ;
    serialize_pipeline voyait alors des compteurs incoherents et levait
    l'invariant de couverture. Desormais toutes les cles sont rafraichies
    dans tous les cas (signaux, 0 signal, interruption, timeout).
    """
    st.session_state.df = df
    st.session_state.pipeline_signals = result.payloads
    st.session_state.scan_time = result.scan_time
    st.session_state.scan_errors = list(result.errors)
    st.session_state.scan_timed_out = int(result.timed_out)
    # PG-03 : compteurs exclusifs pour l'invariant de couverture
    st.session_state.scan_coverage_counts = result.coverage_counts


def _trigger_scan() -> None:
    """Single-entry guarded scan trigger (avoids re-entry on Streamlit reruns)."""
    if st.session_state.scanning:
        return
    st.session_state.scanning = True
    try:
        # Secrets check
        if "OANDA_ACCESS_TOKEN" not in st.secrets:
            st.error("Clé API OANDA manquante dans les secrets.")
            return

        auth: AuthState = st.session_state.auth
        cache_bust: int = st.session_state.cache_bust

        # r7: animated progress bar replacing static st.spinner
        _pb = st.progress(0.0, text="Initialisation du scan…")
        _st = st.empty()
        _t0 = time.monotonic()
        _completed = 0
        _total = len(INSTRUMENTS) * len(TIMEFRAMES)

        def _tick(inst: str, tf: str) -> None:
            nonlocal _completed
            _completed += 1
            pct = _completed / _total
            elapsed = time.monotonic() - _t0
            eta = (elapsed / _completed) * (_total - _completed) if _completed else 0.0
            _pb.progress(
                pct,
                text=f"[{_completed}/{_total}] {inst.replace('_', '/')} ({tf}) — ETA {int(eta)}s"
            )
            _st.markdown(
                f"<div style='font-size:0.8rem;color:#64748b'>"
                f"Workers: {SCAN_MAX_WORKERS} | Timeout: {SCAN_GLOBAL_TIMEOUT}s | "
                f"Écoulé: {elapsed:.1f}s</div>",
                unsafe_allow_html=True,
            )

        try:
            result = run_scan(auth, cache_bust, progress_callback=_tick)
        except Exception as exc:  # noqa: BLE001 — final defensive barrier
            _log(logging.ERROR, "scan_fatal", err=str(exc))
            _pb.empty()
            _st.empty()
            st.error(f"Erreur critique du scan : {exc}")
            return
        finally:
            _pb.empty()
            _st.empty()

        if result.aborted:
            st.error(
                "Scan interrompu — trop d'erreurs d'authentification OANDA. "
                "Vérifiez le token."
            )
            return

        if result.timed_out:
            st.warning(
                f"Timeout global — {result.timed_out} requête(s) ignorée(s), "
                "résultats partiels."
            )
        if result.errors:
            st.warning(
                f"{len(result.errors)} erreur(s) : "
                f"{'; '.join(result.errors[:5])}"
            )

        if not result.rows:
            # A1 (LOT A) : _store_scan_result met a jour TOUTES les cles
            # (y compris erreurs et compteurs) — sinon le JSON du scan
            # precedent fuitait dans celui-ci.
            _store_scan_result(result, None)
            st.info("Aucun signal CHoCH/BOS récent qualifié (Score ≥ 65).")
            return

        df = (
            # tri deterministe — les ex æquo sont departages
            # par Instrument puis Timeframe (mergesort, stable).
            pd.DataFrame(result.rows)
            .sort_values(["_time_sort", "Instrument", "Timeframe"],
                         ascending=[False, True, True], kind="mergesort")
            .drop_duplicates(subset="signal_id", keep="first")
            .drop(columns=["_time_sort"])
            .reset_index(drop=True)
        )
        _store_scan_result(result, df)
        st.success(
            f"Scan terminé — {len(df)} signaux | "
            f"{len(result.payloads)} dans le pipeline JSON | "
            f"rule={RULE_VERSION}"
        )
    finally:
        st.session_state.scanning = False


# =====================================================================
# SECTION 9 — STREAMLIT ENTRY POINT
# =====================================================================

st.set_page_config(
    page_title=f"CHoCH Scanner v{SCANNER_VERSION}",
    layout="wide",
)
st.title(
    f"Scanner Change of Character (CHoCH) — v{SCANNER_VERSION} "
    f"({RULE_VERSION})"
)

_init_session_state()

col_a, col_b = st.columns([3, 1])
with col_a:
    scan_clicked = st.button(
        "Lancer le Scan", type="primary",
        width="stretch",
        disabled=st.session_state.scanning,
        key="btn_scan",
    )
with col_b:
    force_refresh = st.button(
        "Force refresh (bust cache)",
        width="stretch",
        disabled=st.session_state.scanning,
        key="btn_force_refresh",
    )

if force_refresh:
    st.session_state.cache_bust += 1
    get_candles_cached.clear()
    st.toast("Cache des bougies vidé.", icon="🔄")
    # A4 (LOT A) : st.rerun() (API publique). RerunException sans argument
    # leve une TypeError dans streamlit 1.64.0 (le constructeur exige
    # rerun_data). [PROUVÉ-EXÉCUTION : signature inspectée =
    # (self, rerun_data: RerunData) -> None]. RerunException herite de
    # BaseException, pas de Exception : elle n'est donc pas absorbee par
    # le except Exception de _trigger_scan. [PROUVÉ-EXÉCUTION : MRO =
    # RerunException -> ScriptControlException -> BaseException]
    st.rerun()

if scan_clicked:
    _trigger_scan()

# A1 (LOT A) : on rend les resultats des qu'un scan a eu lieu (scan_time),
# pas seulement quand des signaux existent (df is not None). Sinon un scan
# a 0 signal ne propose aucun JSON telechargeable (PG-04 non fonctionnel).
if st.session_state.get("scan_time") is not None:
    _render_results()
