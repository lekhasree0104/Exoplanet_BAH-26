"""
batch_runner.py
===============
Runs the full exoplanet detection pipeline on a list of TESS targets in batch,
saves results to results.json for the dashboard.

Usage:
    python batch_runner.py                    # runs default target list
    python batch_runner.py --targets targets.txt   # one target per line
    python batch_runner.py --sector 1 --max 50     # random sector 1 stars, max 50

Output:
    results.json  — all per-star results (no plots embedded, keeps file small)
    results_with_plots.json — same but with base64 plots (large file)
"""

import json
import time
import traceback
import warnings
import argparse
import sys
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import astropy.units as u
from astropy.timeseries import BoxLeastSquares
from scipy.stats import pearsonr
import lightkurve as lk

# ── Try importing batman (optional) ──────────────────────────────────────────
try:
    import batman
    HAS_BATMAN = True
except ImportError:
    HAS_BATMAN = False
    print("[warn] batman not installed — transit model fitting will be skipped.")

# ── Try importing classifier from main.py ────────────────────────────────────
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from main import clf, _extract_features, LABELS, _run_bls, _compute_snr, _get_stellar_params
    print("[ok] Loaded classifier from main.py")
    USE_MAIN_CLF = True
except Exception as e:
    print(f"[warn] Could not import from main.py ({e}). Training a fresh classifier.")
    USE_MAIN_CLF = False

# ── If we couldn't import, define everything locally ─────────────────────────
if not USE_MAIN_CLF:
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from pathlib import Path as _P

    N_FEATURES = 17
    CLF_PATH = _P("classifier.pkl")

    LABELS = {
        "planet":           ("🪐", "Likely Planet Transit",      "#4ade80"),
        "eclipsing_binary": ("⭐", "Eclipsing Binary",            "#f87171"),
        "starspot":         ("☀️",  "Starspot / Stellar Activity", "#fbbf24"),
        "blend":            ("🔀", "Contaminated Blend",          "#a78bfa"),
    }

    def _make_lc(kind, n=1400):
        t = np.linspace(0, 27, n)
        noise_level = np.random.uniform(200, 1500) * 1e-6
        stellar_period = np.random.uniform(5, 30)
        stellar_amp = np.random.uniform(0, 0.005)
        stellar_var = stellar_amp * np.sin(2 * np.pi * t / stellar_period)
        flux = np.ones(n) + stellar_var
        if kind == "planet":
            period = np.random.uniform(1, 13)
            depth = np.random.uniform(100, 20000) * 1e-6
            dur = np.random.uniform(0.04, 0.2)
            t0 = np.random.uniform(0, period)
            for i, ti in enumerate(t):
                ph = ((ti - t0) % period) / period
                if ph > 0.5: ph -= 1
                if abs(ph) < dur / period / 2:
                    flux[i] -= depth
        elif kind == "eclipsing_binary":
            period = np.random.uniform(0.5, 8)
            depth = np.random.uniform(0.03, 0.4)
            dur = np.random.uniform(0.03, 0.15)
            t0 = np.random.uniform(0, period)
            for i, ti in enumerate(t):
                ph = ((ti - t0) % period) / period
                if ph > 0.5: ph -= 1
                if abs(ph) < dur / period / 2:
                    flux[i] -= depth
        elif kind == "starspot":
            p1 = np.random.uniform(4, 25)
            a1 = np.random.uniform(0.005, 0.05)
            flux += -a1 * np.sin(2 * np.pi * t / p1) ** 2
        else:
            period = np.random.uniform(2, 15)
            depth = np.random.uniform(0.002, 0.01) * np.random.uniform(0.1, 0.6)
            dur = np.random.uniform(0.05, 0.2)
            t0 = np.random.uniform(0, period)
            ph_arr = ((t - t0) % period) / period
            ph_arr[ph_arr > 0.5] -= 1
            flux[np.abs(ph_arr) < dur / period / 2] -= depth
        flux += np.random.normal(0, noise_level, n)
        return t, flux

    def _extract_features(t, flux):
        try:
            err = np.ones(len(flux)) * np.nanstd(flux)
            bls = BoxLeastSquares(t * u.day, flux * u.dimensionless_unscaled,
                                  err * u.dimensionless_unscaled)
            periods = np.linspace(0.5, 20, 2000) * u.day
            result = bls.power(periods, [0.04, 0.08, 0.12, 0.16] * u.day)
            bi = int(np.argmax(result.power))
            P = float(result.period[bi].value)
            pw = float(result.power[bi])
            d = float(result.depth[bi])
            dur = float(result.duration[bi].value)
            t0 = float(result.transit_time[bi].value)
            fap_proxy = pw / (np.percentile(result.power, 95) + 1e-9)
            phase = ((t - t0) % P) / P
            phase[phase > 0.5] -= 1
            hd = dur / P / 2
            in_tr = np.abs(phase) < hd
            out_tr = np.abs(phase) > hd * 3
            noise = float(np.nanstd(flux[out_tr])) if out_tr.sum() > 5 else 1e-6
            n_tr = max(1, int((t[-1] - t[0]) / P))
            snr = (d / noise) * np.sqrt(n_tr)
            ph2 = ((t - t0 + P / 2) % P) / P
            ph2[ph2 > 0.5] -= 1
            sec_d = float(-np.nanmean(flux[np.abs(ph2) < hd] - 1)) if (np.abs(ph2) < hd).sum() > 2 else 0
            sec_r = sec_d / (d + 1e-9)

            def td_at(n_):
                m = np.abs(t - (t0 + n_ * P)) < dur / 2
                return float(-np.nanmean(flux[m] - 1)) if m.sum() > 1 else 0

            ntr = max(1, int(27 / P))
            odd = np.mean([td_at(i) for i in range(0, ntr, 2)[:5]])
            even = np.mean([td_at(i) for i in range(1, ntr, 2)[:5]]) if ntr > 1 else odd
            oe_r = abs(odd - even) / (d + 1e-9)
            sin_m = 1 - d / 2 * (1 - np.cos(2 * np.pi * t / P))
            sc, _ = pearsonr(flux, sin_m)
            in_flux = flux[in_tr]
            shape_score = float(np.nanstd(in_flux) / (d + 1e-9)) if in_tr.sum() > 2 else 0
            poly = np.polyfit(t, flux, 2)
            trend_power = float(np.nanstd(np.polyval(poly, t)) * 1e6)
            features = [
                pw, d * 1e6, dur * 24, snr, sec_r, oe_r, float(sc),
                dur / P, float(np.nanstd(flux) * 1e6),
                float(np.nanmedian(np.abs(np.diff(flux))) * 1e6),
                P, float(in_tr.sum() / len(t)), fap_proxy, n_tr,
                shape_score, trend_power, float(d / (np.nanstd(flux) + 1e-9)),
            ]
            assert len(features) == N_FEATURES
            return features
        except Exception:
            return [0.0] * N_FEATURES

    def _run_bls(t, flux, period_min=0.5, period_max=20.0, n_periods=3000):
        err = np.ones(len(flux)) * np.nanstd(flux)
        bls_obj = BoxLeastSquares(t * u.day, flux * u.dimensionless_unscaled,
                                  err * u.dimensionless_unscaled)
        periods = np.linspace(period_min, period_max, n_periods) * u.day
        result = bls_obj.power(periods, [0.04, 0.08, 0.12, 0.16] * u.day)
        bi = int(np.argmax(result.power))
        P = float(result.period[bi].value)
        power = float(result.power[bi])
        depth = float(result.depth[bi])
        dur = float(result.duration[bi].value)
        t0 = float(result.transit_time[bi].value)
        fap_proxy = float(power / (np.percentile(result.power, 95) + 1e-9))
        return {"result": result, "P": P, "power": power,
                "depth": depth, "dur": dur, "t0": t0, "fap_proxy": fap_proxy}

    def _compute_snr(t, flux, P, depth, dur, t0):
        phase = ((t - t0) % P) / P
        phase[phase > 0.5] -= 1
        hd = dur / P / 2
        out_tr = np.abs(phase) > hd * 3
        noise = float(np.nanstd(flux[out_tr])) if out_tr.sum() > 5 else 1e-6
        n_tr = max(1, int((t[-1] - t[0]) / P))
        snr = (depth / noise) * np.sqrt(n_tr)
        return snr, n_tr, noise * 1e6

    def _get_stellar_params(lc_raw):
        meta = getattr(lc_raw, "meta", {}) or {}
        try:
            rs = float(meta.get("RADIUS") or 1.0)
        except Exception:
            rs = 1.0
        try:
            ms = float(meta.get("TMASS_J") or 1.0)
        except Exception:
            ms = 1.0
        rs = max(0.05, min(rs, 100.0))
        ms = max(0.05, min(ms, 100.0))
        return {"radius_solar": rs, "mass_solar": ms}

    def _load_or_train_clf():
        if CLF_PATH.exists():
            with open(CLF_PATH, "rb") as f:
                return pickle.load(f)
        print("Training classifier...")
        X, y = [], []
        for kind in ["planet", "eclipsing_binary", "starspot", "blend"]:
            for _ in range(300):
                t, flux = _make_lc(kind)
                X.append(_extract_features(t, flux))
                y.append(kind)
        clf_new = RandomForestClassifier(n_estimators=200, max_depth=10,
                                         class_weight="balanced", random_state=42, n_jobs=-1)
        clf_new.fit(X, y)
        with open(CLF_PATH, "wb") as f:
            pickle.dump(clf_new, f)
        return clf_new

    clf = _load_or_train_clf()


# ── Plotting (optional — only if save_plots=True) ────────────────────────────
import io, base64, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _make_plots(lc_raw, lc_flat, t, flux, bls_res, signal) -> list[str]:
    """Generate light curve + phase fold plots, return as base64 list."""
    plots = []
    try:
        P = signal["best_period_days"]
        t0 = signal["transit_epoch_btjd"]

        # Raw + flattened
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        fig.patch.set_facecolor("#0a0a1a")
        for ax in axes:
            ax.set_facecolor("#0d0d2b")
            ax.tick_params(colors="#aac8f0")
        axes[0].scatter(lc_raw.time.value, lc_raw.flux.value, s=0.5, alpha=0.5, color="#7eb8f7")
        axes[0].set_ylabel("Raw flux", color="#aac8f0")
        axes[0].set_title("Raw light curve", color="#e0eeff", fontsize=9)
        axes[1].scatter(t, flux, s=0.5, alpha=0.5, color="#a78bfa")
        axes[1].set_ylabel("Detrended flux", color="#aac8f0")
        axes[1].set_xlabel("Time (BTJD)", color="#aac8f0")
        axes[1].set_title("Flattened light curve", color="#e0eeff", fontsize=9)
        fig.tight_layout()
        plots.append(_fig_to_b64(fig))

        # Phase fold
        folded = lc_flat.fold(period=P * u.day, epoch_time=t0 * u.day)
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        fig2.patch.set_facecolor("#0a0a1a")
        ax2.set_facecolor("#0d0d2b")
        ax2.tick_params(colors="#aac8f0")
        ax2.scatter(folded.time.value, folded.flux.value, s=0.8, alpha=0.5, color="#7eb8f7")
        ax2.set_xlabel("Phase (days)", color="#aac8f0")
        ax2.set_ylabel("Normalised flux", color="#aac8f0")
        label = signal["classification"]["label"]
        conf = signal["classification"]["confidence"]
        ax2.set_title(f"Phase-folded | {label} | conf={conf:.2f}", color="#e0eeff", fontsize=9)
        fig2.tight_layout()
        plots.append(_fig_to_b64(fig2))
    except Exception as e:
        print(f"      [plot error] {e}")
    return plots


# ── Core: analyse one star ────────────────────────────────────────────────────

def analyse_star(target: str, mission: str = "TESS", save_plots: bool = False) -> dict:
    """
    Run the full pipeline on one star.
    Returns a result dict ready for JSON serialisation.
    """
    result = {
        "target": target,
        "mission": mission,
        "status": "error",
        "timestamp": datetime.utcnow().isoformat(),
        "error": None,
        "stellar_params": {},
        "n_points": 0,
        "duration_days": 0,
        "scatter_ppm": 0,
        "n_signals_found": 0,
        "signals": [],
        "transit_fit": None,
        "plots": [],   # base64 strings, only populated if save_plots=True
    }

    try:
        # ── Download ──────────────────────────────────────────────────────
        sr = lk.search_lightcurve(target, mission=[mission])
        if len(sr) == 0:
            sr = lk.search_lightcurve(target)
        if len(sr) == 0:
            result["error"] = "No data found on MAST"
            return result

        lc_raw = sr[0].download()
        if lc_raw is None:
            result["error"] = "Download returned None"
            return result

        result["mission"] = str(sr[0].mission)

        # ── Stellar params ────────────────────────────────────────────────
        stellar = _get_stellar_params(lc_raw)
        result["stellar_params"] = stellar

        # ── Preprocessing ─────────────────────────────────────────────────
        lc = lc_raw.remove_nans().normalize()
        if hasattr(lc, "remove_outliers"):
            lc = lc.remove_outliers(sigma=4)
        lc_flat = lc.flatten(window_length=401)

        t = lc_flat.time.value
        flux = lc_flat.flux.value

        result["n_points"] = len(flux)
        result["duration_days"] = round(float(t[-1] - t[0]), 1)
        result["scatter_ppm"] = round(float(np.nanstd(flux) * 1e6), 1)

        # ── Multi-planet BLS: up to 3 signals ─────────────────────────────
        flux_work = flux.copy()
        signals = []

        for sig_n in range(3):
            bls_res = _run_bls(t, flux_work)
            P, power, depth, dur, t0 = (
                bls_res["P"], bls_res["power"], bls_res["depth"],
                bls_res["dur"], bls_res["t0"],
            )

            if sig_n > 0 and power < 0.05 * signals[0]["bls_power"]:
                break

            snr, n_tr, noise_ppm = _compute_snr(t, flux_work, P, depth, dur, t0)
            fap_proxy = bls_res["fap_proxy"]

            if snr > 15:     sig_str = "Strong (SNR>15)"
            elif snr > 7:    sig_str = "Significant (SNR>7)"
            elif snr > 4:    sig_str = "Marginal (SNR 4-7)"
            else:            sig_str = "Noise (SNR<4)"

            feats = _extract_features(t, flux_work)
            label = clf.predict([feats])[0]
            proba = clf.predict_proba([feats])[0]
            classes = clf.classes_
            conf = float(proba[list(classes).index(label)])

            if conf > 0.85:      conf_str = "High"
            elif conf > 0.65:    conf_str = "Moderate"
            else:                conf_str = "Low"

            _, desc, color = LABELS.get(label, ("❓", "Unknown", "#888"))

            sig_dict = {
                "signal_number":        sig_n + 1,
                "bls_power":            round(power, 3),
                "fap_proxy":            round(fap_proxy, 2),
                "best_period_days":     round(P, 5),
                "transit_depth_ppm":    round(depth * 1e6, 1),
                "transit_duration_hrs": round(dur * 24, 2),
                "transit_epoch_btjd":   round(t0, 4),
                "n_transits_expected":  n_tr,
                "noise_ppm":            round(noise_ppm, 1),
                "snr_stacked":          round(snr, 2),
                "significance":         sig_str,
                "classification": {
                    "label":             label,
                    "description":       desc,
                    "color":             color,
                    "confidence":        round(conf, 3),
                    "confidence_level":  conf_str,
                    "all_probabilities": {
                        c: round(float(p), 3) for c, p in zip(classes, proba)
                    },
                },
            }
            signals.append(sig_dict)

            # Plots for this signal
            if save_plots and sig_n == 0:
                result["plots"] = _make_plots(lc_raw, lc_flat, t, flux, bls_res, sig_dict)

            # Mask transits before next pass
            phase_mask = ((t - t0) % P) / P
            phase_mask[phase_mask > 0.5] -= 1
            in_transit = np.abs(phase_mask) < (dur / P / 2) * 1.5
            flux_work[in_transit] = np.nanmedian(flux_work)

        result["signals"] = signals
        result["n_signals_found"] = len(signals)

        # ── Transit fit (batman) for planet signals ────────────────────────
        if HAS_BATMAN and signals:
            primary = signals[0]
            if (primary["classification"]["label"] == "planet" or
                    primary["classification"]["confidence"] > 0.55):
                try:
                    P = primary["best_period_days"]
                    depth = primary["transit_depth_ppm"] * 1e-6
                    dur = primary["transit_duration_hrs"] / 24
                    t0 = primary["transit_epoch_btjd"]
                    rs = stellar["radius_solar"]
                    ms = stellar["mass_solar"]
                    rp_rs = float(np.sqrt(max(depth, 0)))
                    P_yr = P / 365.25
                    a_AU = (ms * P_yr ** 2) ** (1 / 3)
                    a_rs = a_AU * 215.032 / rs

                    params = batman.TransitParams()
                    params.t0 = t0
                    params.per = P
                    params.rp = rp_rs
                    params.a = max(1.5, a_rs)
                    params.inc = 90.0
                    params.ecc = 0.0
                    params.w = 90.0
                    params.u = [0.3, 0.3]
                    params.limb_dark = "quadratic"

                    phase = ((t - t0) % P) / P
                    phase[phase > 0.5] -= 1
                    t_fold = phase * P
                    sort_i = np.argsort(t_fold)
                    tf = t_fold[sort_i]
                    ff = flux[sort_i]
                    m = batman.TransitModel(params, tf)
                    fm = m.light_curve(params)
                    res = ff - fm
                    rms = float(np.nanstd(res) * 1e6)
                    chi2 = float(np.nansum((res / (np.nanstd(res) + 1e-12)) ** 2) /
                                 max(len(res) - 4, 1))
                    RSUN_TO_REARTH = 109.076
                    rp_earth = rp_rs * rs * RSUN_TO_REARTH
                    L_sun = rs ** 2
                    hz_in = np.sqrt(L_sun / 1.1)
                    hz_out = np.sqrt(L_sun / 0.53)
                    in_hz = bool(hz_in <= a_AU <= hz_out)

                    result["transit_fit"] = {
                        "period_days":            round(P, 5),
                        "transit_depth_ppm":      round(depth * 1e6, 1),
                        "transit_duration_hours": round(dur * 24, 2),
                        "transit_epoch_btjd":     round(t0, 4),
                        "rp_over_rs":             round(rp_rs, 5),
                        "a_over_rs":              round(a_rs, 2),
                        "planet_radius_earth":    round(rp_earth, 2),
                        "semi_major_axis_au":     round(a_AU, 4),
                        "residual_rms_ppm":       round(rms, 1),
                        "reduced_chi2":           round(chi2, 3),
                        "in_habitable_zone":      in_hz,
                        "hz_inner_au":            round(hz_in, 3),
                        "hz_outer_au":            round(hz_out, 3),
                    }
                except Exception as e:
                    result["transit_fit"] = {"error": str(e)}

        result["status"] = "ok"

    except Exception as e:
        result["error"] = str(e)
        result["trace"] = traceback.format_exc()[-600:]

    return result


# ── Default target list ───────────────────────────────────────────────────────
# Mix of confirmed planets, EBs, spotted stars — good for hackathon demo

DEFAULT_TARGETS = [
    # Confirmed multi-planet systems
    "TRAPPIST-1",
    "TOI-700",
    "TOI-1338",
    "TOI-2180",
    "TOI-421",
    "LTT 9779",
    "GJ 1132",
    "HD 21749",
    "TOI-125",
    "TOI-270",
    # Known eclipsing binaries
    "TIC 231663901",
    "TIC 159873822",
    "TIC 441420236",
    # Active / starspot stars
    "TIC 149603524",
    "TIC 261136679",
    "TIC 257527578",
    # Additional TOIs with varying signals
    "TOI-178",
    "TOI-500",
    "TOI-776",
    "TOI-1431",
    "TOI-1518",
    "TOI-811",
    "TOI-1233",
    "TOI-1410",
    "TOI-257",
]


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_batch(targets: list[str], out_path: str = "results.json",
              save_plots: bool = False, delay: float = 1.0,
              resume: bool = True):
    """
    Process all targets, save incrementally to out_path.
    Set save_plots=True to embed base64 plots (makes file large).
    """
    out = Path(out_path)

    # Resume: load existing results
    existing: dict[str, dict] = {}
    if resume and out.exists():
        try:
            with open(out) as f:
                data = json.load(f)
            for r in data.get("results", []):
                existing[r["target"]] = r
            print(f"[resume] Found {len(existing)} existing results in {out_path}")
        except Exception:
            pass

    results = list(existing.values())
    done_targets = set(existing.keys())

    total = len(targets)
    print(f"\n{'='*60}")
    print(f"  Exoplanet Batch Pipeline")
    print(f"  Targets: {total}  |  Plots: {save_plots}  |  Output: {out_path}")
    print(f"{'='*60}\n")

    for i, target in enumerate(targets):
        if target in done_targets:
            print(f"[{i+1:3d}/{total}] {target:30s} — already done, skipping")
            continue

        print(f"[{i+1:3d}/{total}] {target:30s}", end=" ... ", flush=True)
        t_start = time.time()

        r = analyse_star(target, save_plots=save_plots)
        elapsed = time.time() - t_start

        if r["status"] == "ok":
            ns = r["n_signals_found"]
            label = r["signals"][0]["classification"]["label"] if ns > 0 else "—"
            snr = r["signals"][0]["snr_stacked"] if ns > 0 else 0
            print(f"OK  {elapsed:.1f}s  |  signals={ns}  label={label}  SNR={snr:.1f}")
        else:
            print(f"ERROR: {r['error']}")

        results.append(r)

        # Save after every star
        summary = _make_summary(results)
        payload = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_targets": total,
            "completed": len(results),
            "summary": summary,
            "results": results,
        }
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)

        time.sleep(delay)

    print(f"\n{'='*60}")
    print(f"Done. {len(results)} stars processed.")
    print(f"Results saved to {out_path}")
    _print_summary(results)
    print(f"{'='*60}\n")
    return results


def _make_summary(results: list[dict]) -> dict:
    ok = [r for r in results if r["status"] == "ok"]
    label_counts = {}
    snr_values = []
    planet_candidates = []

    for r in ok:
        if r["signals"]:
            s = r["signals"][0]
            lbl = s["classification"]["label"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            snr_values.append(s["snr_stacked"])
            if lbl == "planet" and s["snr_stacked"] > 7:
                planet_candidates.append({
                    "target": r["target"],
                    "period_days": s["best_period_days"],
                    "depth_ppm": s["transit_depth_ppm"],
                    "snr": s["snr_stacked"],
                    "confidence": s["classification"]["confidence"],
                })

    return {
        "total_processed": len(results),
        "successful": len(ok),
        "errors": len(results) - len(ok),
        "label_distribution": label_counts,
        "mean_snr": round(float(np.mean(snr_values)), 2) if snr_values else 0,
        "planet_candidates": sorted(planet_candidates, key=lambda x: -x["snr"]),
    }


def _print_summary(results):
    s = _make_summary(results)
    print(f"\n  Successful: {s['successful']} / {s['total_processed']}")
    print(f"  Label distribution: {s['label_distribution']}")
    print(f"  Mean SNR: {s['mean_snr']}")
    if s["planet_candidates"]:
        print(f"  Planet candidates ({len(s['planet_candidates'])}):")
        for c in s["planet_candidates"][:5]:
            print(f"    {c['target']:25s}  P={c['period_days']:.3f}d  "
                  f"depth={c['depth_ppm']:.0f}ppm  SNR={c['snr']:.1f}  conf={c['confidence']:.2f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exoplanet batch pipeline")
    parser.add_argument("--targets", type=str, default=None,
                        help="Path to text file with one target per line")
    parser.add_argument("--out", type=str, default="results.json",
                        help="Output JSON path (default: results.json)")
    parser.add_argument("--plots", action="store_true",
                        help="Embed base64 plots in JSON (larger file)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to wait between downloads (default: 1.0)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh even if output file exists")
    parser.add_argument("--max", type=int, default=None,
                        help="Max number of targets to process")
    args = parser.parse_args()

    if args.targets:
        with open(args.targets) as f:
            targets = [line.strip() for line in f if line.strip()]
    else:
        targets = DEFAULT_TARGETS
        print(f"No --targets file given. Using default list of {len(targets)} targets.")

    if args.max:
        targets = targets[: args.max]

    run_batch(
        targets,
        out_path=args.out,
        save_plots=args.plots,
        delay=args.delay,
        resume=not args.no_resume,
    )