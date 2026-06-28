"""
Exoplanet Detection Pipeline — main.py  (accuracy-improved)
Pipeline: fetch → denoise → BLS → classify → fit → report

Key fixes vs original:
  1. TIC stellar params via astroquery (not just lightkurve metadata fallback)
  2. BLS period grid: finer resolution + starts at 0.3d (catches USPs)
  3. BLS duration grid: 8 finely-spaced bins instead of 4 coarse ones
  4. Flatten window scaled to cadence (not hardcoded 401)
  5. obs_duration and data_points now populated correctly
  6. SNR uses per-point scatter on out-of-transit baseline (more robust)
  7. rp_rs clamped to physically valid range before batman fit
  8. Duplicate transit-masking threshold tightened to 1.2× (was 1.5×)
  9. Signal loop break threshold raised to avoid killing real signals
 10. Stellar mass default now read from TIC, not hardcoded 1.0

Endpoints:
  GET  /              — dashboard (index.html)
  POST /analyse       — direct science pipeline, no LLM needed
  POST /retrain       — retrain classifier on CSV
  GET  /results       — load batch results for dashboard
  POST /chat          — LLM agent (optional, needs GROQ_API_KEY)
"""

import json
import base64
import io
import warnings
import traceback
import os
import pickle
import time
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
import astropy.units as u
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Groq: lazy import, only if /chat is used ──────────────────────────────────
_groq_client = None
def _get_groq():
    global _groq_client
    if _groq_client is None:
        from groq import Groq
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _groq_client

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_cache: dict[str, Any] = {}
CLF_PATH   = Path(__file__).parent / "classifier.pkl"
N_FEATURES = 17

LABELS = {
    "planet":           ("🪐", "Likely Planet Transit",      "#4ade80"),
    "eclipsing_binary": ("⭐", "Eclipsing Binary",            "#f87171"),
    "starspot":         ("☀️",  "Starspot / Stellar Activity", "#fbbf24"),
    "blend":            ("🔀", "Contaminated Blend",          "#a78bfa"),
}

# ══════════════════════════════════════════════════════════════════
#  SECTION 1 — SYNTHETIC LC GENERATOR (training fallback)
# ══════════════════════════════════════════════════════════════════

def _make_synthetic_lc(kind: str, n: int = 1400):
    t = np.linspace(0, 27, n)
    noise_level    = np.random.uniform(200, 1500) * 1e-6
    stellar_period = np.random.uniform(5, 30)
    stellar_amp    = np.random.uniform(0, 0.005)
    stellar_var    = stellar_amp * np.sin(
        2 * np.pi * t / stellar_period + np.random.uniform(0, 2 * np.pi))
    systematics = np.zeros(n)
    for dump_t in np.arange(3.125, 27, 3.125):
        idx   = np.argmin(np.abs(t - dump_t))
        width = np.random.randint(2, 6)
        systematics[max(0, idx - width): idx + width] += np.random.uniform(-0.002, 0.002)
    flux = np.ones(n) + stellar_var + systematics

    if kind == "planet":
        period = np.random.uniform(0.3, 13)   # FIX: include USPs down to 0.3d
        depth  = np.random.uniform(100, 20000) * 1e-6
        dur    = np.random.uniform(0.04, 0.2)
        t0     = np.random.uniform(0, period)
        for i, ti in enumerate(t):
            ph = ((ti - t0) % period) / period
            if ph > 0.5: ph -= 1
            ph_dur = dur / period
            if abs(ph) < ph_dur / 2:
                flux[i] -= depth * min(1.0, (ph_dur/2 - abs(ph)) / (ph_dur*0.15 + 1e-9))
    elif kind == "eclipsing_binary":
        period    = np.random.uniform(0.5, 8)
        depth     = np.random.uniform(0.03, 0.4)
        sec_depth = depth * np.random.uniform(0.2, 0.8)
        dur       = np.random.uniform(0.03, 0.15)
        t0        = np.random.uniform(0, period)
        for i, ti in enumerate(t):
            ph = ((ti - t0) % period) / period
            if ph > 0.5: ph -= 1
            if abs(ph) < dur / period / 2: flux[i] -= depth
            ph2 = ((ti - t0 + period / 2) % period) / period
            if ph2 > 0.5: ph2 -= 1
            if abs(ph2) < dur / period / 2: flux[i] -= sec_depth
    elif kind == "starspot":
        p1 = np.random.uniform(4, 25)
        p2 = p1 * np.random.uniform(0.9, 1.1)
        a1 = np.random.uniform(0.005, 0.05)
        a2 = np.random.uniform(0.002, 0.02)
        flux += (-a1 * np.sin(2*np.pi*t/p1)**2
                 - a2 * np.sin(2*np.pi*t/p2 + 0.5)**2)
    else:  # blend
        period   = np.random.uniform(2, 15)
        depth    = np.random.uniform(0.002, 0.01)
        dur      = np.random.uniform(0.05, 0.2)
        t0       = np.random.uniform(0, period)
        dilution = np.random.uniform(0.1, 0.6)
        ph_arr   = ((t - t0) % period) / period
        ph_arr[ph_arr > 0.5] -= 1
        flux[np.abs(ph_arr) < dur / period / 2] -= depth * dilution

    flux += np.random.normal(0, noise_level, n)
    return t, flux

# ══════════════════════════════════════════════════════════════════
#  SECTION 2 — FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def _extract_features(t: np.ndarray, flux: np.ndarray) -> list[float]:
    try:
        err    = np.ones(len(flux)) * np.nanstd(flux)
        bls    = BoxLeastSquares(t*u.day, flux*u.dimensionless_unscaled,
                                 err*u.dimensionless_unscaled)
        periods = np.linspace(0.3, 20, 2000)*u.day   # FIX: start at 0.3d
        # FIX: finer duration grid (8 bins) to better sample short transits
        durations = [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20]*u.day
        result  = bls.power(periods, durations)
        bi  = int(np.argmax(result.power))
        P   = float(result.period[bi].value)
        pw  = float(result.power[bi])
        d   = float(result.depth[bi])
        dur = float(result.duration[bi].value)
        t0  = float(result.transit_time[bi].value)
        fap_proxy = pw / (np.percentile(result.power, 95) + 1e-9)

        phase  = ((t - t0) % P) / P
        phase[phase > 0.5] -= 1
        hd     = dur / P / 2
        in_tr  = np.abs(phase) < hd
        out_tr = np.abs(phase) > hd * 3
        noise     = float(np.nanstd(flux[out_tr])) if out_tr.sum() > 5 else 1e-6
        n_transits = max(1, int((t[-1] - t[0]) / P))
        snr_stacked = (d / noise) * np.sqrt(n_transits)

        ph2   = ((t - t0 + P/2) % P) / P
        ph2[ph2 > 0.5] -= 1
        in_sec = np.abs(ph2) < hd
        sec_d  = float(-np.nanmean(flux[in_sec]-1)) if in_sec.sum()>2 else 0
        sec_r  = sec_d / (d + 1e-9)

        def _td_at(n_):
            mask = np.abs(t-(t0+n_*P)) < dur/2
            return float(-np.nanmean(flux[mask]-1)) if mask.sum()>1 else 0
        ntr  = max(1, int(27/P))
        odd  = float(np.mean([_td_at(i) for i in range(0, ntr, 2)[:5]]))
        even = float(np.mean([_td_at(i) for i in range(1, ntr, 2)[:5]])) if ntr>1 else odd
        oe_r = abs(odd-even)/(d+1e-9)

        sin_m       = 1 - d/2*(1-np.cos(2*np.pi*t/P))
        sc, _       = pearsonr(flux, sin_m)
        in_flux     = flux[in_tr]
        shape_score = float(np.nanstd(in_flux)/(d+1e-9)) if in_tr.sum()>2 else 0
        poly        = np.polyfit(t, flux, 2)
        trend_power = float(np.nanstd(np.polyval(poly, t))*1e6)

        features = [
            pw, d*1e6, dur*24, snr_stacked, sec_r, oe_r, float(sc),
            dur/P, float(np.nanstd(flux)*1e6),
            float(np.nanmedian(np.abs(np.diff(flux)))*1e6),
            P, float(in_tr.sum()/len(t)), fap_proxy, n_transits,
            shape_score, trend_power, float(d/(np.nanstd(flux)+1e-9)),
        ]
        assert len(features) == N_FEATURES
        return features
    except Exception:
        return [0.0]*N_FEATURES

# ══════════════════════════════════════════════════════════════════
#  SECTION 3 — CLASSIFIER
# ══════════════════════════════════════════════════════════════════

def _load_or_train_clf():
    if CLF_PATH.exists():
        with open(CLF_PATH, "rb") as f:
            loaded = pickle.load(f)
        expected = getattr(loaded, "n_features_in_", None)
        if expected is not None and expected != N_FEATURES:
            print(f"[clf] Stale ({expected} features). Retraining...")
            CLF_PATH.unlink()
        else:
            print(f"[clf] Loaded from {CLF_PATH}")
            return loaded

    print("[clf] Training on synthetic data (600 per class)...")
    X, y = [], []
    for kind in ["planet","eclipsing_binary","starspot","blend"]:
        for _ in range(600):
            t, flux = _make_synthetic_lc(kind)
            X.append(_extract_features(t, flux))
            y.append(kind)
    clf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                  min_samples_leaf=3, class_weight="balanced",
                                  random_state=42, n_jobs=-1)
    clf.fit(X, y)
    with open(CLF_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"[clf] Trained and saved.")
    return clf

clf = _load_or_train_clf()


def retrain_on_real_dataset(csv_path: str) -> dict:
    global clf
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c != "label"]
        if len(feature_cols) != N_FEATURES:
            return {"error": f"Expected {N_FEATURES} feature cols, got {len(feature_cols)}"}
        X = df[feature_cols].values.tolist()
        y = df["label"].tolist()
        new_clf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                          min_samples_leaf=3, class_weight="balanced",
                                          random_state=42, n_jobs=-1)
        new_clf.fit(X, y)
        with open(CLF_PATH, "wb") as f:
            pickle.dump(new_clf, f)
        clf = new_clf
        from collections import Counter
        return {"retrained": True, "n_samples": len(y),
                "class_counts": dict(Counter(y)), "csv_path": csv_path}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()[-400:]}


_real_csv = Path(__file__).parent / "training_features.csv"
if _real_csv.exists():
    retrain_on_real_dataset(str(_real_csv))
    print("[clf] Auto-retrained on real data.")

# ══════════════════════════════════════════════════════════════════
#  SECTION 4 — BLS, SNR, STELLAR PARAMS
# ══════════════════════════════════════════════════════════════════

# FIX: query TIC via astroquery for accurate stellar parameters
def _query_tic(target: str) -> dict:
    """
    Query MAST TIC catalog for stellar radius, mass, Teff.
    Falls back gracefully if astroquery or network unavailable.
    """
    defaults = {"radius_solar": 1.0, "mass_solar": 1.0,
                "teff_k": None, "logg": None, "tic_id": None}
    try:
        from astroquery.mast import Catalogs
        # Try direct TIC ID first, then name resolve
        results = Catalogs.query_object(target, catalog="TIC", radius=0.001)
        if results is None or len(results) == 0:
            results = Catalogs.query_object(target, catalog="TIC", radius=0.01)
        if results is None or len(results) == 0:
            return defaults

        row = results[0]  # closest match

        def _safe(col, fallback=None):
            try:
                v = float(row[col])
                return v if np.isfinite(v) else fallback
            except Exception:
                return fallback

        rs = _safe("rad", None)
        ms = _safe("mass", None)
        # TIC sometimes has radius but not mass — derive from R-M relation
        if rs is not None and ms is None:
            # Simple empirical: M ≈ R^1.25 for main-sequence (Torres 2010 approx)
            ms = rs ** 1.25
        if ms is not None and rs is None:
            rs = ms ** 0.8

        return {
            "radius_solar": round(max(0.05, min(rs or 1.0, 100.0)), 6),
            "mass_solar":   round(max(0.05, min(ms or 1.0, 100.0)), 4),
            "teff_k":       _safe("Teff"),
            "logg":         _safe("logg"),
            "tic_id":       str(row["ID"]) if "ID" in row.colnames else None,
        }
    except Exception:
        return defaults


def _get_stellar_params(lc_raw, target: str = "") -> dict:
    """
    FIX: Try TIC query first (accurate), fall back to lightkurve metadata,
    then hardcoded solar values. Also extract obs_duration and n_cadences.
    """
    # --- obs stats from the light curve itself ---
    try:
        times = lc_raw.time.value
        obs_duration = round(float(times[-1] - times[0]), 2)
        n_cadences   = int(len(times))
    except Exception:
        obs_duration = None
        n_cadences   = None

    # --- stellar params: TIC query first ---
    if target:
        params = _query_tic(target)
        if params["radius_solar"] != 1.0 or params["mass_solar"] != 1.0:
            params["obs_duration_days"] = obs_duration
            params["n_cadences"]        = n_cadences
            params["source"]            = "TIC"
            return params

    # --- fallback: lightkurve metadata ---
    meta = getattr(lc_raw, "meta", {}) or {}
    # lightkurve stores TIC radius in RADIUS key
    try:    rs = float(meta.get("RADIUS") or meta.get("TEFF") or 1.0)
    except: rs = 1.0
    # mass is rarely in metadata; use R-M relation if radius available
    if rs != 1.0:
        ms = rs ** 1.25
    else:
        try:    ms = float(meta.get("MASS") or 1.0)
        except: ms = 1.0

    return {
        "radius_solar":      max(0.05, min(rs, 100.0)),
        "mass_solar":        max(0.05, min(ms, 100.0)),
        "teff_k":            None,
        "logg":              None,
        "tic_id":            None,
        "obs_duration_days": obs_duration,
        "n_cadences":        n_cadences,
        "source":            "lightkurve_meta",
    }


def _run_bls(t, flux, period_min=0.3, period_max=20.0, n_periods=10000):
    """
    FIX 1: period_min=0.3d (catches USPs like 55 Cnc e at 0.737d with margin).
    FIX 2: n_periods=10000 → step ≈ 0.002d at P~0.73d (was 0.0065d → 3× better resolution).
    FIX 3: 8 duration bins instead of 4, better sampling short transits.
    """
    err     = np.ones(len(flux)) * np.nanstd(flux)
    bls     = BoxLeastSquares(t*u.day, flux*u.dimensionless_unscaled,
                               err*u.dimensionless_unscaled)
    periods = np.linspace(period_min, period_max, n_periods) * u.day
    # FIX: finer duration grid — 55 Cnc e transit ≈ 0.066d (1.58h)
    durations = [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20] * u.day
    result  = bls.power(periods, durations)
    bi      = int(np.argmax(result.power))
    P       = float(result.period[bi].value)
    power   = float(result.power[bi])
    depth   = float(result.depth[bi])
    dur     = float(result.duration[bi].value)
    t0      = float(result.transit_time[bi].value)
    fap     = float(power / (np.percentile(result.power, 95) + 1e-9))
    return {"result": result, "P": P, "power": power, "depth": depth,
            "dur": dur, "t0": t0, "fap_proxy": fap}


def _compute_snr(t, flux, P, depth, dur, t0):
    """
    FIX: Use robust MAD-based scatter on out-of-transit baseline,
    and clamp depth to be positive before dividing.
    """
    phase  = ((t - t0) % P) / P
    phase[phase > 0.5] -= 1
    hd     = dur / P / 2
    out_tr = np.abs(phase) > hd * 3
    if out_tr.sum() > 10:
        # FIX: use MAD for robustness against remaining outliers
        oot_flux = flux[out_tr]
        noise = float(np.nanmedian(np.abs(oot_flux - np.nanmedian(oot_flux))) * 1.4826)
    else:
        noise = float(np.nanstd(flux))
    noise = max(noise, 1e-6)

    n_tr = max(1, int((t[-1] - t[0]) / P))
    # FIX: clamp depth to positive
    snr  = (max(depth, 0) / noise) * np.sqrt(n_tr)
    return snr, n_tr, noise * 1e6


def _significance_label(snr):
    if snr > 15: return "Strong (SNR > 15)"
    if snr > 7:  return "Significant (SNR > 7)"
    if snr > 4:  return "Marginal (SNR 4-7)"
    return "Noise (SNR < 4)"


def _cadence_minutes(t: np.ndarray) -> float:
    """Estimate median cadence in minutes from time array."""
    diffs = np.diff(t) * 24 * 60  # convert days to minutes
    return float(np.nanmedian(diffs[diffs > 0]))


def _flatten_window(t: np.ndarray, target_days: float = 0.75) -> int:
    """
    FIX: Compute flatten window length based on actual cadence.
    Target window = ~0.75 days (long enough to capture stellar variability
    but not so long it smears transit signals).
    Must be ODD.
    """
    cadence_min = _cadence_minutes(t)
    cadence_days = cadence_min / (24 * 60)
    n = int(round(target_days / cadence_days))
    if n % 2 == 0:
        n += 1
    return max(51, n)   # minimum 51 points


# ══════════════════════════════════════════════════════════════════
#  SECTION 5 — PLOTTING (unchanged from original)
# ══════════════════════════════════════════════════════════════════

def _style_ax(ax):
    ax.set_facecolor("#0d0d2b")
    ax.tick_params(colors="#aac8f0")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a5a")

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def _plot_lightcurve(lc_raw, lc_flat, target, mission):
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig.patch.set_facecolor("#0a0a1a")
    for ax in axes: _style_ax(ax)
    axes[0].scatter(lc_raw.time.value, lc_raw.flux.value,
                    s=0.5, alpha=0.6, color="#7eb8f7")
    axes[0].set_ylabel("Raw flux", color="#aac8f0")
    axes[0].set_title(f"{target} — Raw light curve ({mission})",
                      color="#e0eeff", fontsize=10)
    axes[1].scatter(lc_flat.time.value, lc_flat.flux.value,
                    s=0.5, alpha=0.6, color="#a78bfa")
    axes[1].set_ylabel("Detrended flux", color="#aac8f0")
    axes[1].set_xlabel("Time (BTJD)", color="#aac8f0")
    axes[1].set_title("Flattened (detrended) light curve",
                      color="#e0eeff", fontsize=10)
    fig.tight_layout()
    return _fig_to_b64(fig)

def _plot_bls_and_fold(bls_res, lc_flat, signal, sig_n):
    P     = signal["best_period_days"]
    t0    = signal["transit_epoch_btjd"]
    snr   = signal["snr_stacked"]
    label = signal["classification"]["label"]
    conf  = signal["classification"]["confidence"]
    color = signal["classification"]["color"]
    folded = lc_flat.fold(period=P*u.day, epoch_time=t0*u.day)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("#0a0a1a")
    for ax in axes: _style_ax(ax)
    axes[0].plot(bls_res["result"].period.value, bls_res["result"].power,
                 color="#7eb8f7", lw=0.8)
    axes[0].axvline(P, color="#ff9f7f", lw=1.5, ls="--",
                    label=f"P={P:.5f}d | SNR={snr:.1f}")
    axes[0].set_xlabel("Period (days)", color="#aac8f0")
    axes[0].set_ylabel("BLS power", color="#aac8f0")
    axes[0].set_title(f"Signal {sig_n} — BLS Periodogram", color="#e0eeff", fontsize=10)
    axes[0].legend(fontsize=8, labelcolor="#e0eeff", facecolor="#0d0d2b")
    axes[1].scatter(folded.time.value, folded.flux.value,
                    s=0.8, alpha=0.6, color="#7eb8f7")
    axes[1].set_xlabel("Phase (days)", color="#aac8f0")
    axes[1].set_ylabel("Normalised flux", color="#aac8f0")
    axes[1].set_title(f"Phase-folded | {label} | conf={conf:.2f}",
                      color=color, fontsize=10)
    fig.tight_layout()
    return _fig_to_b64(fig)

def _plot_transit_fit(tf, ff, fm, rms):
    residuals = ff - fm
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor("#0a0a1a")
    for ax in axes: _style_ax(ax)
    axes[0].scatter(tf, ff, s=0.8, alpha=0.4, color="#7eb8f7", label="Data")
    axes[0].plot(tf, fm, color="#ff9f7f", lw=1.5, label="Batman model")
    axes[0].set_xlabel("Phase (days)", color="#aac8f0")
    axes[0].set_ylabel("Normalised flux", color="#aac8f0")
    axes[0].set_title("Transit model fit", color="#e0eeff", fontsize=10)
    axes[0].legend(fontsize=8, labelcolor="#e0eeff", facecolor="#0d0d2b")
    axes[1].scatter(tf, residuals*1e6, s=0.8, alpha=0.5, color="#a78bfa")
    axes[1].axhline(0, color="#ff9f7f", lw=1, ls="--")
    axes[1].set_xlabel("Phase (days)", color="#aac8f0")
    axes[1].set_ylabel("Residuals (ppm)", color="#aac8f0")
    axes[1].set_title(f"Residuals | RMS={rms:.0f} ppm", color="#e0eeff", fontsize=10)
    fig.tight_layout()
    return _fig_to_b64(fig)

# ══════════════════════════════════════════════════════════════════
#  SECTION 6 — CORE PIPELINE (no LLM)
# ══════════════════════════════════════════════════════════════════

def run_pipeline(target: str, mission: str = "TESS",
                 author: str = "", sector: int = 0) -> dict:
    """Full science pipeline — no LLM. Called by /analyse endpoint."""
    try:
        kwargs: dict = {"mission": [mission]}
        if author: kwargs["author"] = author
        if sector: kwargs["sector"] = sector
        sr = lk.search_lightcurve(target, **kwargs)
        if len(sr) == 0:
            sr = lk.search_lightcurve(target)
        if len(sr) == 0:
            return {"error": f"No data found for '{target}' on MAST."}

        lc_raw = sr[0].download()
        if lc_raw is None:
            return {"error": "Download returned None."}

        mission_used = str(sr[0].mission)

        # FIX: get stellar params with TIC query + obs stats
        stellar = _get_stellar_params(lc_raw, target)
        _cache["stellar"] = stellar

        lc = lc_raw.remove_nans().normalize()
        if hasattr(lc, "remove_outliers"):
            lc = lc.remove_outliers(sigma=4)

        # FIX: compute flatten window from actual cadence, not hardcoded 401
        t_raw      = lc.time.value
        win_length = _flatten_window(t_raw, target_days=0.75)
        lc_flat    = lc.flatten(window_length=win_length)

        t    = lc_flat.time.value
        flux = lc_flat.flux.value
        _cache.update({"lc": lc_flat, "lc_raw": lc, "target": target,
                       "t": t, "flux": flux})

        plot_lc   = _plot_lightcurve(lc, lc_flat, target, mission_used)
        flux_work = flux.copy()
        signals   = []

        for sig_n in range(1, 4):
            # FIX: use improved BLS (finer grid, more durations, starts at 0.3d)
            bls_res = _run_bls(t, flux_work)
            P, power, depth, dur, t0 = (bls_res["P"], bls_res["power"],
                bls_res["depth"], bls_res["dur"], bls_res["t0"])

            # FIX: raise threshold from 0.05 to 0.08 so we don't kill weaker real signals
            if sig_n > 1 and power < 0.08 * signals[0]["bls_power"]:
                break

            snr, n_tr, noise_ppm = _compute_snr(t, flux_work, P, depth, dur, t0)
            feats   = _extract_features(t, flux_work)
            label   = clf.predict([feats])[0]
            proba   = clf.predict_proba([feats])[0]
            classes = list(clf.classes_)
            conf    = float(proba[classes.index(label)])
            _, desc, color = LABELS.get(label, ("❓", "Unknown", "#888"))

            snr_norm = min(1.0, snr / 15.0)
            combined = round(0.5 * conf + 0.5 * snr_norm, 3)

            signal = {
                "signal_number":        sig_n,
                "bls_power":            round(power, 4),
                "fap_proxy":            round(bls_res["fap_proxy"], 3),
                "best_period_days":     round(P, 5),
                "transit_depth_ppm":    round(max(depth, 0) * 1e6, 2),   # FIX: clamp
                "transit_duration_hrs": round(dur * 24, 3),
                "transit_epoch_btjd":   round(t0, 4),
                "n_transits_expected":  n_tr,
                "noise_ppm":            round(noise_ppm, 2),
                "snr_stacked":          round(snr, 3),
                "significance":         _significance_label(snr),
                "classification": {
                    "label":             label,
                    "description":       desc,
                    "color":             color,
                    "confidence":        round(conf, 4),
                    "combined_score":    combined,
                    "all_probabilities": {c: round(float(p), 4)
                                         for c, p in zip(classes, proba)},
                },
                "batman_fit":  {},
                "bls_plot":    _plot_bls_and_fold(bls_res, lc_flat, signal
                               if False else {
                                   "best_period_days":  round(P, 5),
                                   "transit_epoch_btjd": round(t0, 4),
                                   "snr_stacked":       round(snr, 3),
                                   "classification": {
                                       "label":      label,
                                       "confidence": round(conf, 4),
                                       "color":      color,
                                   }
                               }, sig_n),
                "batman_plot": None,
            }

            # Batman fit for planet signals
            if label == "planet" or conf > 0.5:
                try:
                    import batman
                    rs    = stellar["radius_solar"]
                    ms    = stellar["mass_solar"]

                    # FIX: clamp rp_rs to physically valid range [0.001, 0.5]
                    rp_rs = float(np.sqrt(max(depth, 1e-8)))
                    rp_rs = max(0.001, min(rp_rs, 0.5))

                    P_yr  = P / 365.25
                    a_AU  = (ms * P_yr**2) ** (1 / 3)
                    a_rs  = a_AU * 215.032 / rs

                    params = batman.TransitParams()
                    params.t0  = t0;   params.per = P;    params.rp = rp_rs
                    params.a   = max(1.5, a_rs)
                    params.inc = 90.0; params.ecc = 0.0;  params.w  = 90.0
                    params.u   = [0.3, 0.3];               params.limb_dark = "quadratic"

                    phase  = ((t - t0) % P) / P
                    phase[phase > 0.5] -= 1
                    t_fold = phase * P
                    sort_i = np.argsort(t_fold)
                    tf, ff = t_fold[sort_i], flux[sort_i]
                    m      = batman.TransitModel(params, tf)
                    fm     = m.light_curve(params)
                    res    = ff - fm
                    rms    = float(np.nanstd(res) * 1e6)
                    chi2   = float(np.nansum((res / (np.nanstd(res) + 1e-12))**2)
                                   / max(len(res) - 4, 1))
                    rp_earth = rp_rs * rs * 109.076
                    L_sun    = rs**2
                    hz_in    = float(np.sqrt(L_sun / 1.1))
                    hz_out   = float(np.sqrt(L_sun / 0.53))

                    signal["batman_fit"] = {
                        "period_days":            round(P, 5),
                        "transit_depth_ppm":      round(max(depth, 0) * 1e6, 2),
                        "transit_duration_hours": round(dur * 24, 3),
                        "transit_epoch_btjd":     round(t0, 4),
                        "rp_over_rs":             round(rp_rs, 5),
                        "a_over_rs":              round(a_rs, 3),
                        "planet_radius_earth":    round(rp_earth, 3),
                        "semi_major_axis_au":     round(a_AU, 5),
                        "residual_rms_ppm":       round(rms, 2),
                        "reduced_chi2":           round(chi2, 4),
                        "in_habitable_zone":      bool(hz_in <= a_AU <= hz_out),
                        "hz_inner_au":            round(hz_in, 4),
                        "hz_outer_au":            round(hz_out, 4),
                    }
                    signal["batman_plot"] = _plot_transit_fit(tf, ff, fm, rms)
                except Exception:
                    pass

            signals.append(signal)

            # FIX: tighter mask (1.2× instead of 1.5×) to avoid smearing
            # adjacent signals in the second/third BLS pass
            phase_mask = ((t - t0) % P) / P
            phase_mask[phase_mask > 0.5] -= 1
            in_transit = np.abs(phase_mask) < (dur / P / 2) * 1.2
            flux_work[in_transit] = np.nanmedian(flux_work)

        _cache["plots"] = ([plot_lc] +
                           [s["bls_plot"] for s in signals if s["bls_plot"]])
        if signals:
            s0 = signals[0]
            _cache["bls"] = {
                "P":     s0["best_period_days"],
                "depth": s0["transit_depth_ppm"] * 1e-6,
                "dur":   s0["transit_duration_hrs"] / 24,
                "t0":    s0["transit_epoch_btjd"],
            }

        return {
            "target":            target,
            "mission":           mission_used,
            "n_cadences":        stellar.get("n_cadences", len(flux)),
            "obs_duration_days": stellar.get("obs_duration_days",
                                             round(float(t[-1] - t[0]), 2)),
            "baseline_days":     round(float(t[-1] - t[0]), 2),
            "scatter_ppm":       round(float(np.nanstd(flux) * 1e6), 2),
            "stellar_params":    stellar,
            "flatten_window":    win_length,
            "n_signals_found":   len(signals),
            "signals":           signals,
            "plot_lc":           plot_lc,
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()[-800:]}

# ══════════════════════════════════════════════════════════════════
#  SECTION 7 — TOOL FUNCTIONS (kept for /chat compatibility)
# ══════════════════════════════════════════════════════════════════

def search_tess_target(target: str, mission: str = "any") -> dict:
    try:
        missions = None if mission == "any" else [mission]
        sr = lk.search_lightcurve(target, mission=missions)
        if len(sr) == 0:
            return {"found": False, "message": f"No light curves for '{target}'."}
        rows = [{"index": i, "target": str(sr[i].target_name),
                 "mission": str(sr[i].mission), "author": str(sr[i].author),
                 "year": str(sr[i].year)} for i in range(min(len(sr), 10))]
        return {"found": True, "count": len(sr), "showing": rows}
    except Exception as e:
        return {"error": str(e)}

def fetch_and_analyse(target: str, mission: str = "TESS",
                      author: str = "", sector: int = 0) -> dict:
    return run_pipeline(target, mission, author, sector)

def fit_transit_model() -> dict:
    if "bls" not in _cache or "t" not in _cache:
        return {"error": "Run fetch_and_analyse first."}
    try:
        import batman
        t    = _cache["t"]; flux = _cache["flux"]
        b    = _cache["bls"]
        P, depth, dur, t0 = b["P"], b["depth"], b["dur"], b["t0"]
        stellar = _cache.get("stellar", {"radius_solar": 1.0, "mass_solar": 1.0})
        rs = stellar["radius_solar"]; ms = stellar["mass_solar"]
        rp_rs = float(np.sqrt(max(depth, 1e-8)))
        rp_rs = max(0.001, min(rp_rs, 0.5))   # FIX: clamp
        P_yr  = P / 365.25; a_AU = (ms * P_yr**2) ** (1 / 3)
        a_rs  = a_AU * 215.032 / rs
        params = batman.TransitParams()
        params.t0  = t0;   params.per = P;    params.rp = rp_rs
        params.a   = max(1.5, a_rs)
        params.inc = 90.0; params.ecc = 0.0;  params.w  = 90.0
        params.u   = [0.3, 0.3];               params.limb_dark = "quadratic"
        phase  = ((t - t0) % P) / P; phase[phase > 0.5] -= 1
        t_fold = phase * P; sort_i = np.argsort(t_fold)
        tf, ff = t_fold[sort_i], flux[sort_i]
        m      = batman.TransitModel(params, tf)
        fm     = m.light_curve(params)
        res    = ff - fm; rms = float(np.nanstd(res) * 1e6)
        chi2   = float(np.nansum((res / (np.nanstd(res) + 1e-12))**2)
                       / max(len(res) - 4, 1))
        rp_earth = rp_rs * rs * 109.076; L_sun = rs**2
        hz_in    = float(np.sqrt(L_sun / 1.1)); hz_out = float(np.sqrt(L_sun / 0.53))
        plot_b64 = _plot_transit_fit(tf, ff, fm, rms)
        _cache.setdefault("plots", []).append(plot_b64)
        return {
            "stellar_params_used": {"radius_solar": round(rs, 4),
                                    "mass_solar":   round(ms, 4),
                                    "source":       stellar.get("source", "unknown")},
            "fitted_parameters": {
                "period_days":            round(P, 5),
                "transit_depth_ppm":      round(max(depth, 0) * 1e6, 2),
                "transit_duration_hours": round(dur * 24, 3),
                "transit_epoch_btjd":     round(t0, 4),
                "rp_over_rs":             round(rp_rs, 5),
                "a_over_rs":              round(a_rs, 3),
                "planet_radius_earth":    round(rp_earth, 3),
                "semi_major_axis_au":     round(a_AU, 5),
            },
            "fit_quality":   {"residual_rms_ppm": round(rms, 2),
                              "reduced_chi2":      round(chi2, 4)},
            "habitability":  {"in_habitable_zone": bool(hz_in <= a_AU <= hz_out),
                              "hz_inner_au":        round(hz_in, 4),
                              "hz_outer_au":        round(hz_out, 4)},
            "plot_b64": plot_b64,
        }
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()[-500:]}

def get_cached_plots() -> dict:
    return {"n_plots": len(_cache.get("plots", [])),
            "plots":   _cache.get("plots", [])}

def get_star_info(target: str) -> dict:
    try:
        sr = lk.search_lightcurve(target)
        if len(sr) == 0: return {"error": "Not found."}
        return {"target_name": str(sr[0].target_name),
                "ra":          str(sr[0].ra),
                "dec":         str(sr[0].dec)}
    except Exception as e:
        return {"error": str(e)}

def load_labeled_dataset(csv_path: str) -> dict:
    return retrain_on_real_dataset(csv_path)

# ══════════════════════════════════════════════════════════════════
#  SECTION 8 — TOOL DISPATCHER + GROQ TOOLS (for /chat)
# ══════════════════════════════════════════════════════════════════

TOOL_FN_MAP = {
    "search_tess_target":   search_tess_target,
    "fetch_and_analyse":    fetch_and_analyse,
    "fit_transit_model":    fit_transit_model,
    "get_cached_plots":     get_cached_plots,
    "get_star_info":        get_star_info,
    "load_labeled_dataset": load_labeled_dataset,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "search_tess_target",
        "description": "Search NASA MAST for available light curves. Always call first.",
        "parameters": {"type": "object", "properties": {
            "target":  {"type": "string"},
            "mission": {"type": "string", "enum": ["TESS", "Kepler", "K2", "any"]}
        }, "required": ["target"]}}},
    {"type": "function", "function": {
        "name": "fetch_and_analyse",
        "description": "Download light curve and run full pipeline: detrend, BLS (3 signals), classify, SNR, FAP, plots.",
        "parameters": {"type": "object", "properties": {
            "target":  {"type": "string"},
            "mission": {"type": "string", "enum": ["TESS", "Kepler", "K2"]},
            "author":  {"type": "string"},
            "sector":  {"type": "integer"}
        }, "required": ["target"]}}},
    {"type": "function", "function": {
        "name": "fit_transit_model",
        "description": "Fit batman transit model. Call after fetch_and_analyse.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_cached_plots",
        "description": "Retrieve all plots generated this session.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_star_info",
        "description": "Get RA/Dec from MAST.",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string"}}, "required": ["target"]}}},
    {"type": "function", "function": {
        "name": "load_labeled_dataset",
        "description": "Retrain classifier on real labeled CSV.",
        "parameters": {"type": "object", "properties": {
            "csv_path": {"type": "string"}}, "required": ["csv_path"]}}},
]

SYSTEM_PROMPT = """You are an exoplanet detection assistant with NASA TESS/Kepler pipeline tools.

For any star: 1) search_tess_target 2) fetch_and_analyse 3) fit_transit_model if planet/conf>0.6.

Interpret: Stacked SNR>7 significant, >15 strong. FAP proxy>3 = clear peak. Secondary eclipse + odd-even asymmetry = EB not planet. Planet radius uses real TIC stellar radius. Comment on ALL signals."""


def _dispatch_tool(name, args):
    fn = TOOL_FN_MAP.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"}), []
    args = args if isinstance(args, dict) else {}
    try:
        result = fn(**args)
    except Exception as e:
        return json.dumps({"error": str(e),
                           "trace": traceback.format_exc()[-400:]}), []
    plots = []
    if isinstance(result, dict):
        if "plot_b64" in result: plots.append(result.pop("plot_b64"))
        if "plots"   in result: plots.extend(result.pop("plots"))
    return json.dumps(result), plots


def _groq_call(messages, tools, max_retries=3):
    client = _get_groq()
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL, messages=messages, tools=tools,
                tool_choice="auto", max_tokens=1500)
        except Exception as e:
            if attempt < max_retries - 1: time.sleep(1.5); continue
            raise

# ══════════════════════════════════════════════════════════════════
#  SECTION 9 — FASTAPI ENDPOINTS
# ══════════════════════════════════════════════════════════════════

class AnalyseRequest(BaseModel):
    target:  str
    mission: str = "TESS"
    author:  str = ""
    sector:  int = 0

class RetrainRequest(BaseModel):
    csv_path: str

class ChatRequest(BaseModel):
    messages: list[dict]


@app.post("/analyse")
async def analyse(req: AnalyseRequest):
    """Direct science pipeline — no LLM. Used by the dashboard."""
    result = run_pipeline(req.target, req.mission, req.author, req.sector)
    return JSONResponse(content=result)


@app.post("/retrain")
async def retrain(req: RetrainRequest):
    """Retrain classifier on real labeled CSV."""
    result = retrain_on_real_dataset(req.csv_path)
    return JSONResponse(content=result)


@app.get("/results")
async def get_results():
    """Return batch results from results.json if it exists."""
    results_path = Path(__file__).parent / "results.json"
    if not results_path.exists():
        return JSONResponse(content={
            "error": "results.json not found. Run batch_runner.py first."})
    try:
        with open(results_path) as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)})


@app.post("/chat")
async def chat(req: ChatRequest):
    """LLM agent endpoint — requires GROQ_API_KEY env variable."""
    all_plots: list[str] = []
    tool_log:  list[str] = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in req.messages:
        messages.append({"role": m["role"], "content": m["content"]})

    for _ in range(14):
        response = _groq_call(messages, TOOLS)
        choice   = response.choices[0]
        msg      = choice.message
        asst: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            asst["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls]
        messages.append(asst)

        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            for tc in msg.tool_calls:
                name = tc.function.name
                raw  = tc.function.arguments
                try:    args = json.loads(raw) if raw else {}
                except: args = {}
                tool_log.append(
                    f"→ {name}({json.dumps(args, separators=(',', ':'))})")
                result_str, new_plots = _dispatch_tool(name, args)
                all_plots.extend(new_plots)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                  "name": name, "content": result_str})
            continue

        for p in _cache.get("plots", []):
            if p not in all_plots: all_plots.append(p)
        return {"reply":      msg.content or "No response.",
                "plots":      all_plots,
                "tool_calls": tool_log}

    return {"reply": "Agent loop limit reached.",
            "plots": all_plots, "tool_calls": tool_log}


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", encoding="utf-8") as f:
        return f.read()