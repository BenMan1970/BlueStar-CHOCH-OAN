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
# FIX #2 : as_completed remplacé par wait — plus de perte de résultats sur timeout
from concurrent.futures import ThreadPoolExecutor, wait, TimeoutError as FuturesTimeoutError
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from oandapyV20.exceptions import V20Error
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from typing import Optional, Dict, List, Any

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
SCANNER_VERSION = "5.8"

# ===================== CONFIG =====================
INSTRUMENTS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "USD_CAD", "AUD_USD", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "EUR_CHF", "EUR_AUD", "EUR_CAD", "EUR_NZD",
    "GBP_JPY", "GBP_CHF", "GBP_AUD", "GBP_CAD", "GBP_NZD",
    "AUD_JPY", "AUD_CAD", "AUD_CHF", "AUD_NZD", "CAD_JPY", "CAD_CHF", "CHF_JPY",
    "NZD_JPY", "NZD_CAD", "NZD_CHF",
    "DE30_EUR", "XAU_USD", "SPX500_USD", "NAS100_USD", "US30_USD",
]

VOLATILITY_STATIC = {
    "EUR_USD": "Basse", "GBP_USD": "Basse", "USD_JPY": "Basse", "USD_CHF": "Basse", "USD_CAD": "Basse",
    "AUD_USD": "Moyenne", "NZD_USD": "Moyenne", "EUR_GBP": "Moyenne", "EUR_JPY": "Moyenne",
    "EUR_CHF": "Moyenne", "EUR_AUD": "Moyenne", "EUR_CAD": "Moyenne", "EUR_NZD": "Moyenne",
    "GBP_JPY": "Haute", "GBP_CHF": "Haute", "GBP_AUD": "Haute", "GBP_CAD": "Haute", "GBP_NZD": "Haute",
    "AUD_JPY": "Haute", "AUD_CAD": "Moyenne", "AUD_CHF": "Haute", "AUD_NZD": "Moyenne",
    "CAD_JPY": "Haute", "CAD_CHF": "Haute", "CHF_JPY": "Haute", "NZD_JPY": "Haute",
    "NZD_CAD": "Moyenne", "NZD_CHF": "Haute", "DE30_EUR": "Très Haute", "XAU_USD": "Très Haute",
    "SPX500_USD": "Très Haute", "NAS100_USD": "Très Haute", "US30_USD": "Très Haute",
}

TIMEFRAMES     = {"H1": "H1", "H4": "H4", "D1": "D", "Weekly": "W"}
SWING_LOOKBACK = {"H1": 5, "H4": 5, "D1": 4, "Weekly": 3}
SWING_HISTORY  = {"H1": 120, "H4": 90, "D1": 60, "Weekly": 26}
ATR_DIST_MULT  = 1.8  # Filtre dynamique : distance max = ATR * 1.8
MIN_SCORE      = 65   # Seuil de confluence pour affichage

SCAN_GLOBAL_TIMEOUT   = 180
FUTURE_RESULT_TIMEOUT = 20

TF_STATUT = {
    "H1":     {"Fresh": 4, "Aged": 12},
    "H4":     {"Fresh": 3, "Aged": 8},
    "D1":     {"Fresh": 2, "Aged": 5},
    "Weekly": {"Fresh": 2, "Aged": 4},
}

DISPLAY_COLS = [
    "Instrument", "Timeframe", "Type", "Ordre", "Signal",
    "Niveau", "Distance%", "Volatilité", "Force", "BB_Width", "Statut", "Heure (UTC)"
]
EXPORT_COLS = DISPLAY_COLS

GRAN_COUNT     = {"H1": 400, "H4": 300, "D": 200, "W": 120}

# ===================== API THREAD-SAFE =====================
_thread_local = threading.local()
def _get_api() -> API:
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

try:
    _ = st.secrets["OANDA_ACCESS_TOKEN"]
except Exception as e:
    st.error(f"Token OANDA manquant dans les secrets Streamlit : {e}")
    st.stop()

# ===================== UTILITAIRES =====================
def _compute_true_range(df: pd.DataFrame) -> np.ndarray:
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    return np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))

def calc_atr_bundle(df: pd.DataFrame, inst: str, period: int = 14) -> tuple[float, str]:
    tr = _compute_true_range(df)
    if len(tr) < period * 3:
        return float("nan"), VOLATILITY_STATIC.get(inst, "Moyenne")
    atr_val = float(pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().iloc[-1])
    window = tr[-100:] if len(tr) >= 100 else tr
    median_tr = float(np.median(window))
    if np.isnan(atr_val) or np.isclose(median_tr, 0, atol=1e-10):
        return atr_val, VOLATILITY_STATIC.get(inst, "Moyenne")
    ratio = atr_val / median_tr
    if ratio >= 1.8: return atr_val, "Très Haute"
    if ratio >= 1.2: return atr_val, "Haute"
    if ratio >= 0.7: return atr_val, "Moyenne"
    return atr_val, "Basse"

def instrument_precision(inst: str) -> int:
    if any(k in inst for k in ["SPX500", "NAS100", "US30", "DE30", "XAU", "XAG"]): return 2
    if "JPY" in inst: return 3
    return 5

def format_niveau(niveau: Optional[float], inst: str) -> str:
    if niveau is None: return "N/A"
    return f"{niveau:.{instrument_precision(inst)}f}"

def calc_distance_pct(niveau: Optional[float], close_actuel: Optional[float]) -> Optional[float]:
    if niveau is None or close_actuel is None or np.isclose(niveau, 0, atol=1e-8): return None
    dist = abs(close_actuel - niveau) / abs(niveau) * 100
    return dist if dist <= 100 else None

def format_distance(dist_pct: Optional[float]) -> str:
    if dist_pct is None: return "N/A"
    return f"{dist_pct:.3f}%"

def get_session(dt: datetime) -> str:
    h = dt.hour
    london = 7 <= h < 16
    ny     = 13 <= h < 22
    tokyo  = 0 <= h < 9
    if london and ny: return "London_NY_Overlap"
    if london: return "London"
    if ny: return "NewYork"
    if tokyo: return "Tokyo"
    return "Off"

def is_premium_session(s: str) -> bool:
    return s in ("London", "NewYork", "London_NY_Overlap")

def get_candles(inst: str, gran: str) -> Optional[pd.DataFrame]:
    count = GRAN_COUNT.get(gran, 300)
    try:
        r = instruments.InstrumentsCandles(instrument=inst, params={"count": count, "granularity": gran})
        _get_api().request(r)
        candles = [c for c in r.response.get("candles", []) if c.get("complete")]
        if len(candles) < 50: return None
        df = pd.DataFrame([{
            "time": pd.to_datetime(c["time"], utc=True),
            "open": float(c["mid"]["o"]), "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]), "close": float(c["mid"]["c"])
        } for c in candles])
        df.set_index("time", inplace=True)
        return df
    except V20Error as e:
        logger.warning(f"V20Error [{inst}/{gran}] code={e.code}: {e}")
        return None
    except Exception as e:
        logger.error(f"Erreur get_candles [{inst}/{gran}]: {type(e).__name__}: {e}")
        return None

def compute_bb_width(df: pd.DataFrame, length: int = 20, std: int = 2) -> str:
    close = df["close"]
    if len(close) < length * 2: return "N/A"
    sma = close.rolling(length).mean()
    std_dev = close.rolling(length).std()
    upper, lower = sma + std * std_dev, sma - std * std_dev
    bb_w = (upper - lower) / sma
    bb_avg = bb_w.rolling(length).mean()
    avg_last = bb_avg.iloc[-1]
    if pd.isna(avg_last) or np.isclose(avg_last, 0, atol=1e-10): return "N/A"
    bb_avg_safe = bb_avg.replace(0, np.nan)
    pct = ((bb_w - bb_avg) / bb_avg_safe * 100).iloc[-1]
    if pd.isna(pct): return "N/A"
    sign = "+" if pct >= 0 else ""
    if pct <= -25: return f"{sign}{pct:.0f}%_Squeeze"
    if pct >= 25: return f"{sign}{pct:.0f}%_Expansion"
    return f"{sign}{pct:.0f}%_Normal"

def compute_statut(idx_sig: Optional[int], len_df: int, tf: str) -> str:
    if idx_sig is None: return "N/A"
    candles_elapsed = (len_df - 1) - idx_sig
    thresholds = TF_STATUT.get(tf, {"Fresh": 2, "Aged": 5})
    if candles_elapsed <= thresholds["Fresh"]: return "Fresh"
    if candles_elapsed <= thresholds["Aged"]: return "Aged"
    return "Stale"

# ===================== CORE V5.8 =====================
def detect_swing_points(df: pd.DataFrame, tf: str) -> list[dict]:
    lookback = SWING_LOOKBACK.get(tf, 5)
    history = SWING_HISTORY.get(tf, 60)
    h, l = df["high"].values, df["low"].values
    n = len(h)
    start = max(lookback, n - history - lookback)
    end = n - lookback - 1
    pivots = []
    for i in range(start, end):
        if h[i] == max(h[i-lookback:i+lookback+1]): pivots.append((i, h[i], 'H'))
        if l[i] == min(l[i-lookback:i+lookback+1]): pivots.append((i, l[i], 'L'))
    pivots.sort(key=lambda x: x[0])
    
    swings, prev_h, prev_l = [], None, None
    for idx, price, k in pivots:
        if k == 'H':
            kind = 'HH' if (prev_h is None or price > prev_h) else 'LH'
            swings.append({"idx": idx, "price": price, "kind": kind})
            prev_h = price
        else:
            kind = 'HL' if (prev_l is None or price > prev_l) else 'LL'
            swings.append({"idx": idx, "price": price, "kind": kind})
            prev_l = price
    return swings

def get_structural_trend(swings: list[dict]) -> str:
    if len(swings) < 4: return "Range"
    rec = swings[-6:]
    highs = [s for s in rec if s["kind"] in ("HH","LH")]
    lows  = [s for s in rec if s["kind"] in ("HL","LL")]
    if not highs or not lows: return "Range"
    if highs[-1]["kind"] == "HH" and lows[-1]["kind"] == "HL": return "Bullish"
    if highs[-1]["kind"] == "LH" and lows[-1]["kind"] == "LL": return "Bearish"
    return "Range"

def detect_choch_v58(df: pd.DataFrame, tf: str, inst: str) -> Optional[dict]:
    swings = detect_swing_points(df, tf)
    trend = get_structural_trend(swings)
    if trend == "Range": return None
    
    c, h, l, o = df["close"].values, df["high"].values, df["low"].values, df["open"].values
    n = len(c)
    atr_val, _ = calc_atr_bundle(df, inst)
    if np.isnan(atr_val) or atr_val <= 0: return None
    
    lookback_check = 5
    for offset in range(lookback_check):
        idx = n - 1 - offset
        if idx < 3: break
        
        sig_type, direction, level = None, None, None
        prev_swings = [s for s in swings if s["idx"] < idx - 2]
        if not prev_swings: continue

        if trend == "Bullish":
            hl = [s for s in prev_swings if s["kind"]=="HL"]
            if hl and c[idx] < hl[-1]["price"] and c[idx-1] >= hl[-1]["price"]:
                sig_type, direction, level = "CHoCH", "Bearish", hl[-1]["price"]
            # FIX F-002 : elif — empêche le BOS d'écraser silencieusement un CHoCH
            # si les deux conditions sont vraies sur la même bougie
            else:
                hh = [s for s in prev_swings if s["kind"]=="HH"]
                if hh and c[idx] > hh[-1]["price"] and c[idx-1] <= hh[-1]["price"]:
                    sig_type, direction, level = "BOS", "Bullish", hh[-1]["price"]
        elif trend == "Bearish":
            lh = [s for s in prev_swings if s["kind"]=="LH"]
            if lh and c[idx] > lh[-1]["price"] and c[idx-1] <= lh[-1]["price"]:
                sig_type, direction, level = "CHoCH", "Bullish", lh[-1]["price"]
            # FIX F-002 : elif — même logique côté bearish
            else:
                ll = [s for s in prev_swings if s["kind"]=="LL"]
                if ll and c[idx] < ll[-1]["price"] and c[idx-1] >= ll[-1]["price"]:
                    sig_type, direction, level = "BOS", "Bearish", ll[-1]["price"]
                
        if sig_type is None: continue

        # Validation: clôture confirmée + corps ≥ 40%
        rng = h[idx] - l[idx]
        body = abs(c[idx] - o[idx])
        if rng <= 0 or (body / rng) < 0.40: continue

        # Filtre Liquidity Sweep
        has_sweep = False
        if direction == "Bearish":
            sweep_candidates = [s for s in prev_swings if s["kind"] in ("HH", "LH")]
        else:
            sweep_candidates = [s for s in prev_swings if s["kind"] in ("HL", "LL")]
        
        for s in sweep_candidates:
            if direction == "Bearish" and h[idx] > s["price"] and (h[idx] - s["price"]) > (atr_val * 0.25):
                has_sweep = True
                break
            if direction == "Bullish" and l[idx] < s["price"] and (s["price"] - l[idx]) > (atr_val * 0.25):
                has_sweep = True
                break

        # FIX #1 : distance calculée sur la bougie de breakout (c[idx]), pas sur c[-1]
        # Avant : dist_atr = abs(c[-1] - level) / atr_val  ← non-stationnaire
        # Après : ancré sur l'instant réel du signal, résultat déterministe
        dist_atr = abs(c[idx] - level) / atr_val
        if dist_atr > ATR_DIST_MULT: continue

        # Scoring Confluence
        score = 25  # Base structure validée
        if dist_atr <= 1.0: score += 15
        if is_premium_session(get_session(df.index[idx])): score += 20
        if has_sweep: score += 15
        if sig_type == "CHoCH": score += 10
        score = min(score, 100)
        if score < MIN_SCORE: continue

        return {
            "sig_type": sig_type, "direction": direction, "level": level,
            "idx_break": idx,
            # FIX #1 : close_price = prix à la bougie du signal, pas le prix courant
            "close_price": c[idx],
            # Prix courant conservé séparément pour usage informatif dans le JSON
            "current_price": c[-1],
            "has_sweep": has_sweep,
            "atr_val": atr_val, "dist_atr": dist_atr, "score": score
        }
    return None

def build_pipeline_payload_v58(df, inst, inst_disp, tf_name, sig, trend, scan_time, len_df, bb_str, volatilite, force):
    time_sig = df.index[sig["idx_break"]]
    session = get_session(time_sig)
    dist_pct = calc_distance_pct(sig["level"], sig["close_price"])
    candles_since = (len_df - 1) - sig["idx_break"]
    statut = compute_statut(sig["idx_break"], len_df, tf_name)

    return {
        "signal_id": f"{inst}__{tf_name}__{time_sig.strftime('%Y%m%dT%H%M')}",
        "scanner_version": SCANNER_VERSION,
        "generated_at": scan_time.isoformat(),
        "pair": inst_disp,
        "pair_oanda": inst,
        "timeframe": tf_name,
        "type": sig["sig_type"],
        "direction": sig["direction"],
        "is_bullish": sig["direction"] == "Bullish",
        "order": "buy" if sig["direction"] == "Bullish" else "sell",
        "trend": trend,
        "is_choch": sig["sig_type"] == "CHoCH",
        "status": statut,
        "confluence_score": sig["score"],
        "level": round(float(sig["level"]), instrument_precision(inst)),
        # Prix ancré sur la bougie du breakout (stationnaire)
        "close_price": round(float(sig["close_price"]), instrument_precision(inst)),
        # Prix courant au moment du scan
        "current_price": round(float(sig["current_price"]), instrument_precision(inst)),
        "distance_pct": round(dist_pct, 4) if dist_pct is not None else None,
        "distance_atr_multiple": round(sig["dist_atr"], 2),
        "volatility": volatilite,
        "force": force,
        "bb_width_pct": bb_str.split("%")[0].replace("+","").strip() if "%" in bb_str else None,
        "bb_regime": bb_str.split("%")[-1].strip() if "%" in bb_str else "N/A",
        "session": session,
        "signal_time": time_sig.isoformat(),
        "candles_elapsed": candles_since,
    }

# ===================== EXPORT =====================
def create_pdf(df_export: pd.DataFrame) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), leftMargin=20, rightMargin=20, topMargin=40, bottomMargin=40)
    elements, styles = [], getSampleStyleSheet()
    elements.append(Paragraph(f"Rapport des Signaux CHoCH v{SCANNER_VERSION}", styles["Title"]))
    elements.append(Paragraph(f"Généré le {datetime.now(timezone.utc).strftime('%d/%m/%Y à %H:%M')} UTC", styles["Normal"]))
    elements.append(Spacer(1, 20))
    
    cols_present = [c for c in EXPORT_COLS if c in df_export.columns]
    col_widths_map = {c: 60 for c in cols_present}
    col_widths_map.update({"Instrument": 65, "Distance%": 52, "Statut": 45, "Heure (UTC)": 105})
    col_widths = [col_widths_map.get(c, 60) for c in cols_present]
    
    data = [cols_present] + df_export[cols_present].values.tolist()
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), colors.HexColor("#1e40af")), ('TEXTCOLOR', (0,0),(-1,0), colors.white),
        ('ALIGN', (0,0),(-1,-1), 'CENTER'), ('FONTNAME', (0,0),(-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0),(-1,0), 9), ('FONTSIZE', (0,1),(-1,-1), 8),
        ('GRID', (0,0),(-1,-1), 0.5, colors.grey), ('ROWBACKGROUNDS',(0,1),(-1,-1), [colors.white, colors.beige]),
    ]))
    elements.append(table); doc.build(elements); buffer.seek(0); return buffer

def generate_png(df: pd.DataFrame, display_cols: list) -> io.BytesIO:
    fig = Figure(figsize=(22, max(5, len(df) * 0.35)))
    ax = fig.add_subplot(111); ax.axis('off')
    disp = df[[c for c in display_cols if c in df.columns]]
    tbl = ax.table(cellText=disp.values, colLabels=disp.columns, cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.2, 1.8)
    buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', dpi=200); buf.seek(0); return buf

# ===================== UI =====================
st.set_page_config(page_title="CHoCH Scanner v5.8", layout="wide")
st.markdown(
    "<h1 style='text-align:center;color:#1e40af;margin-bottom:30px;'>"
    "Scanner Change of Character (CHoCH) — v5.8 Intraday</h1>",
    unsafe_allow_html=True
)

if "scanning" not in st.session_state: st.session_state.scanning = False

if st.button("Lancer le Scan", type="primary", use_container_width=True, disabled=st.session_state.scanning):
    st.session_state.scanning = True
    n_combos = len(INSTRUMENTS) * len(TIMEFRAMES)
    st.session_state.scan_time = datetime.now(timezone.utc)
    scan_time = st.session_state.scan_time

    try:
        with st.spinner(f"Scan en cours sur {n_combos} combinaisons…"):
            results, pipeline_signals, errors = [], [], []
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = {
                    executor.submit(get_candles, inst, tf_code): (inst, tf_name)
                    for inst in INSTRUMENTS
                    for tf_name, tf_code in TIMEFRAMES.items()
                }

                # FIX #2 : wait() sépare explicitement done / not_done
                # Plus de perte de résultats déjà complétés en cas de timeout global
                done, not_done = wait(futures.keys(), timeout=SCAN_GLOBAL_TIMEOUT)
                if not_done:
                    st.warning(f"Timeout global atteint — {len(not_done)} requête(s) ignorée(s), résultats partiels.")
                    for f in not_done:
                        f.cancel()

                for future in done:
                    inst, tf_name = futures[future]
                    try:
                        df = future.result(timeout=FUTURE_RESULT_TIMEOUT)
                    except FuturesTimeoutError:
                        errors.append(f"{inst}/{tf_name}: timeout")
                        continue
                    except Exception as e:
                        errors.append(f"{inst}/{tf_name}: {e}")
                        continue
                    if df is None: continue

                    sig = detect_choch_v58(df, tf_name, inst)
                    if not sig: continue

                    trend = get_structural_trend(detect_swing_points(df, tf_name))
                    
                    # Construction payload UI
                    df_sub = df.iloc[:sig["idx_break"]+1]
                    _, volatilite = calc_atr_bundle(df_sub, inst)
                    bb_str = compute_bb_width(df_sub)
                    # Distance affichée = depuis le close du breakout (cohérent avec le filtre)
                    dist_pct = calc_distance_pct(sig["level"], sig["close_price"])
                    inst_display = inst.replace("_", "/")
                    statut = compute_statut(sig["idx_break"], len(df), tf_name)
                    rng = df["high"].values[sig["idx_break"]] - df["low"].values[sig["idx_break"]]
                    body = abs(df["close"].values[sig["idx_break"]] - df["open"].values[sig["idx_break"]])
                    force = "Fort" if (rng > 0 and body/rng >= 0.6) else "Moyen" if (rng > 0 and body/rng >= 0.4) else "Faible"

                    results.append({
                        "Instrument": inst_display, "Paire": inst_display, "_time_sort": df.index[sig["idx_break"]],
                        "Timeframe": tf_name, "Type": sig["sig_type"],
                        "Ordre": "Achat" if sig["direction"] == "Bullish" else "Vente",
                        "Signal": f"{sig['direction']} {sig['sig_type']}",
                        "Niveau": format_niveau(sig["level"], inst),
                        "Distance%": format_distance(dist_pct),
                        "Volatilité": volatilite, "Force": force, "BB_Width": bb_str,
                        "Statut": statut, "Heure (UTC)": df.index[sig["idx_break"]].strftime("%Y-%m-%d %H:%M")
                    })

                    if statut in ("Fresh", "Aged"):
                        pipeline_signals.append(build_pipeline_payload_v58(
                            df, inst, inst_display, tf_name, sig, trend, scan_time, len(df),
                            bb_str, volatilite, force
                        ))

            if errors: st.warning(f"{len(errors)} erreur(s) : {'; '.join(errors[:5])}")
            if results:
                df_sorted = pd.DataFrame(results).sort_values("_time_sort", ascending=False)
                before_dedup = len(df_sorted)
                # FIX #3 : déduplication sur (Instrument, Timeframe, Type, Ordre)
                # Avant : subset=["Instrument","Timeframe"] — éliminait CHoCH si BOS présent sur même paire/TF
                # Après : deux signaux de nature différente sur la même paire/TF sont conservés
                df_result = (
                    df_sorted
                    .drop_duplicates(subset=["Instrument", "Timeframe", "Type", "Ordre"], keep="first")
                    .drop(columns=["_time_sort"])
                    .reset_index(drop=True)
                )
                if before_dedup > len(df_result):
                    logger.warning(f"{before_dedup - len(df_result)} doublon(s) éliminé(s)")
                st.session_state.df = df_result
                st.session_state.pipeline_signals = pipeline_signals
                st.session_state.png_buf = None  # reset — sera généré à la demande
                st.success(f"Scan terminé – {len(df_result)} signaux sur {n_combos} combinaisons | {len(pipeline_signals)} dans le pipeline JSON")
            else:
                st.info("Aucun signal CHoCH/BOS récent qualifié (Score ≥ 65)")
    except Exception as e:
        st.error(f"Erreur critique : {e}")
        logger.exception(e)
    finally:
        st.session_state.scanning = False

if "df" in st.session_state:
    df_all = st.session_state.df.copy()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    df_export = df_all[df_all["Statut"].isin(["Fresh", "Aged"])].copy()
    pipeline_signals = st.session_state.get("pipeline_signals", [])
    scan_time_meta = st.session_state.get("scan_time", datetime.now(timezone.utc))

    n_stale = len(df_all[df_all["Statut"] == "Stale"])
    if n_stale > 0: st.info(f"{n_stale} signal(s) Stale visible(s) dans le tableau — exclus des exports.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button("CSV", df_export[[c for c in EXPORT_COLS if c in df_export.columns]].to_csv(index=False).encode(), f"choch_{ts}.csv", "text/csv")
    with col2:
        # FIX #4 : PNG généré à la demande (lazy) — pas de blocage du thread principal après chaque scan
        if st.session_state.get("png_buf") is None:
            st.session_state.png_buf = generate_png(df_all, DISPLAY_COLS)
        st.session_state.png_buf.seek(0)
        st.download_button("PNG", st.session_state.png_buf, f"choch_{ts}.png", "image/png")
    with col3:
        st.download_button("PDF", create_pdf(df_export), f"choch_signaux_{ts}.pdf", "application/pdf")
    with col4:
        pipeline_json = json.dumps({
            "meta": {
                "scanner_version": SCANNER_VERSION,
                "generated_at": scan_time_meta.isoformat(),
                "signal_count": len(pipeline_signals)
            },
            "signals": pipeline_signals
        }, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        st.download_button("JSON", pipeline_json, f"choch_pipeline_{ts}.json", "application/json")

    def style_bb(val): return "color:#ff9800;font-weight:bold" if "Squeeze" in str(val) else "color:#ab47bc;font-weight:bold" if "Expansion" in str(val) else "color:#90a4ae"
    def style_distance(val):
        try:
            v = float(str(val).replace("%", ""))
            return "color:#00c853;font-weight:bold" if v <= 0.15 else "color:#ff9800;font-weight:bold" if v <= 0.40 else "color:#ff5252;font-weight:bold"
        except:
            return "color:#90a4ae"

    cols_display = [c for c in DISPLAY_COLS if c in df_all.columns]
    st.dataframe(
        df_all[cols_display].style
        .map(lambda x: "color:#e879f9;font-weight:bold" if x == "CHoCH" else "color:#94a3b8" if x == "BOS" else "", subset=["Type"])
        .map(lambda x: "color:#00c853;font-weight:bold" if x == "Achat" else "color:#ff5252;font-weight:bold" if x == "Vente" else "", subset=["Ordre"])
        .map(lambda x: "color:#00c853" if "Bull" in str(x) else "color:#ff5252" if "Bear" in str(x) else "", subset=["Signal"])
        .map(lambda x: "color:#00c853;font-weight:bold" if x == "Fort" else "color:#ff5252" if x == "Faible" else "color:#ff9800", subset=["Force"])
        .map(style_bb, subset=["BB_Width"]).map(style_distance, subset=["Distance%"])
        .map(lambda x: "color:#00c853;font-weight:bold" if x == "Fresh" else "color:#ff9800;font-weight:bold" if x == "Aged" else "color:#ff5252;font-weight:bold" if x == "Stale" else "", subset=["Statut"]),
        hide_index=True, use_container_width=True
    )

    if pipeline_signals:
        with st.expander(f"Aperçu JSON Pipeline ({len(pipeline_signals)} signaux Fresh/Aged)"):
            st.json(pipeline_signals[0])
