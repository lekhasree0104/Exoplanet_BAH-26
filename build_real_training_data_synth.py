"""
build_real_training_data_synth.py
────────────────────────────────────────────────────────────────────────────────
Fallback: fills missing class counts with realistic synthetic data
when real MAST downloads fail or are insufficient.
Called automatically by build_real_training_data.py — do not run directly.
────────────────────────────────────────────────────────────────────────────────
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import pearsonr
import astropy.units as u
from astropy.timeseries import BoxLeastSquares

N_FEATURES = 17


def _make_lc(kind, n=1400):
    t  = np.linspace(0, 27, n)
    noise_level = np.random.uniform(200, 1500) * 1e-6
    stellar_period = np.random.uniform(5, 30)
    stellar_amp    = np.random.uniform(0, 0.005)
    stellar_var    = stellar_amp * np.sin(2*np.pi*t/stellar_period + np.random.uniform(0, 2*np.pi))
    systematics = np.zeros(n)
    for dump_t in np.arange(3.125, 27, 3.125):
        idx = np.argmin(np.abs(t - dump_t))
        width = np.random.randint(2, 6)
        systematics[max(0, idx-width):idx+width] += np.random.uniform(-0.002, 0.002)
    flux = np.ones(n) + stellar_var + systematics

    if kind == "planet":
        period = np.random.uniform(1, 13)
        depth  = np.random.uniform(100, 20000) * 1e-6
        dur    = np.random.uniform(0.04, 0.2)
        t0     = np.random.uniform(0, period)
        for i, ti in enumerate(t):
            ph = ((ti - t0) % period) / period
            if ph > 0.5: ph -= 1
            ph_dur = dur / period
            if abs(ph) < ph_dur / 2:
                ingress = min(1.0, (ph_dur/2 - abs(ph)) / (ph_dur * 0.15 + 1e-9))
                flux[i] -= depth * min(1.0, ingress)

    elif kind == "eclipsing_binary":
        period    = np.random.uniform(0.5, 8)
        depth     = np.random.uniform(0.03, 0.4)
        sec_depth = depth * np.random.uniform(0.2, 0.8)
        dur       = np.random.uniform(0.03, 0.15)
        t0        = np.random.uniform(0, period)
        for i, ti in enumerate(t):
            ph = ((ti - t0) % period) / period
            if ph > 0.5: ph -= 1
            if abs(ph) < dur/period/2:
                flux[i] -= depth
            ph2 = ((ti - t0 + period/2) % period) / period
            if ph2 > 0.5: ph2 -= 1
            if abs(ph2) < dur/period/2:
                flux[i] -= sec_depth

    elif kind == "starspot":
        p1 = np.random.uniform(4, 25)
        p2 = p1 * np.random.uniform(0.9, 1.1)
        a1 = np.random.uniform(0.005, 0.05)
        a2 = np.random.uniform(0.002, 0.02)
        flux += -a1 * np.sin(2*np.pi*t/p1)**2 - a2 * np.sin(2*np.pi*t/p2 + 0.5)**2

    else:  # blend
        period   = np.random.uniform(2, 15)
        depth    = np.random.uniform(0.002, 0.01)
        dur      = np.random.uniform(0.05, 0.2)
        t0       = np.random.uniform(0, period)
        dilution = np.random.uniform(0.1, 0.6)
        ph_arr   = ((t - t0) % period) / period
        ph_arr[ph_arr > 0.5] -= 1
        flux[np.abs(ph_arr) < dur/period/2] -= depth * dilution

    flux += np.random.normal(0, noise_level, n)
    return t, flux


def extract_features_synth(t, flux):
    try:
        err    = np.ones(len(flux)) * np.nanstd(flux)
        bls    = BoxLeastSquares(t*u.day, flux*u.dimensionless_unscaled, err*u.dimensionless_unscaled)
        periods= np.linspace(0.5, 20, 2000)*u.day
        result = bls.power(periods, [0.04, 0.08, 0.12, 0.16]*u.day)
        bi     = int(np.argmax(result.power))
        P      = float(result.period[bi].value)
        pw     = float(result.power[bi])
        d      = float(result.depth[bi])
        dur    = float(result.duration[bi].value)
        t0     = float(result.transit_time[bi].value)
        fap_proxy = pw / (np.percentile(result.power, 95) + 1e-9)

        phase  = ((t - t0) % P) / P
        phase[phase > 0.5] -= 1
        hd     = dur/P/2
        in_tr  = np.abs(phase) < hd
        out_tr = np.abs(phase) > hd*3
        noise  = float(np.nanstd(flux[out_tr])) if out_tr.sum() > 5 else 1e-6
        n_tr   = max(1, int((t[-1]-t[0])/P))
        snr    = (d/noise)*np.sqrt(n_tr)

        ph2    = ((t - t0 + P/2) % P) / P
        ph2[ph2 > 0.5] -= 1
        sec_d  = float(-np.nanmean(flux[np.abs(ph2)<hd]-1)) if (np.abs(ph2)<hd).sum()>2 else 0
        sec_r  = sec_d/(d+1e-9)

        def td_at(n_):
            m = np.abs(t-(t0+n_*P)) < dur/2
            return float(-np.nanmean(flux[m]-1)) if m.sum()>1 else 0
        ntr  = max(1, int(27/P))
        odd  = np.mean([td_at(i) for i in range(0, ntr, 2)[:5]])
        even = np.mean([td_at(i) for i in range(1, ntr, 2)[:5]]) if ntr>1 else odd
        oe_r = abs(odd-even)/(d+1e-9)

        sin_m       = 1 - d/2*(1 - np.cos(2*np.pi*t/P))
        sc, _       = pearsonr(flux, sin_m)
        shape_score = float(np.nanstd(flux[in_tr])/(d+1e-9)) if in_tr.sum()>2 else 0
        poly        = np.polyfit(t, flux, 2)
        trend_power = float(np.nanstd(np.polyval(poly, t))*1e6)

        feats = [
            pw, d*1e6, dur*24, snr, sec_r, oe_r, float(sc), dur/P,
            float(np.nanstd(flux)*1e6),
            float(np.nanmedian(np.abs(np.diff(flux)))*1e6),
            P, float(in_tr.sum()/len(t)), fap_proxy, n_tr,
            shape_score, trend_power, float(d/(np.nanstd(flux)+1e-9)),
        ]
        assert len(feats) == N_FEATURES
        return feats
    except Exception:
        return None


def fill_synthetic(missing: dict) -> list:
    """Generate synthetic rows for classes that couldn't be filled from real data."""
    rows = []
    for kind, count in missing.items():
        generated = 0
        attempts  = 0
        while generated < count and attempts < count * 10:
            attempts += 1
            t, flux = _make_lc(kind)
            feats   = extract_features_synth(t, flux)
            if feats is not None:
                rows.append(feats + [kind])
                generated += 1
        print(f"  Synthetic fill: {generated}/{count} for '{kind}'")
    return rows