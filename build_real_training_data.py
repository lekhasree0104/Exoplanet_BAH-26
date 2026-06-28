"""
build_real_training_data.py
Downloads real labeled TESS light curves and extracts 17 features for classifier.
Output: training_features.csv

Run once before starting the server:
    python build_real_training_data.py

Then in agent chat: load_labeled_dataset("training_features.csv")
"""

import warnings
warnings.filterwarnings("ignore")

import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
import astropy.units as u
from astropy.timeseries import BoxLeastSquares
import lightkurve as lk

# ── Config ────────────────────────────────────────────────────────────────────
OUT_CSV       = Path("training_features.csv")
MAX_PER_CLASS = 60      # 60 per class = 240 total, finishes in ~15 min
TIMEOUT_SEC   = 60      # max seconds per star download
SLEEP_SEC     = 0.5
N_FEATURES    = 17

DISP_TO_LABEL = {
    "CP":  "planet",
    "KP":  "planet",
    "PC":  "planet",
    "EB":  "eclipsing_binary",
    "FP":  "blend",
}

# ── Feature extractor (must stay identical to main.py) ────────────────────────

def extract_features(t, flux):
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
        hd     = dur / P / 2
        in_tr  = np.abs(phase) < hd
        out_tr = np.abs(phase) > hd * 3
        noise  = float(np.nanstd(flux[out_tr])) if out_tr.sum() > 5 else 1e-6
        n_tr   = max(1, int((t[-1] - t[0]) / P))
        snr    = (d / noise) * np.sqrt(n_tr)

        ph2   = ((t - t0 + P/2) % P) / P
        ph2[ph2 > 0.5] -= 1
        sec_d = float(-np.nanmean(flux[np.abs(ph2)<hd]-1)) if (np.abs(ph2)<hd).sum()>2 else 0
        sec_r = sec_d / (d + 1e-9)

        def td_at(n_):
            m = np.abs(t-(t0+n_*P)) < dur/2
            return float(-np.nanmean(flux[m]-1)) if m.sum()>1 else 0
        ntr  = max(1, int(27/P))
        odd  = np.mean([td_at(i) for i in range(0, ntr, 2)[:5]])
        even = np.mean([td_at(i) for i in range(1, ntr, 2)[:5]]) if ntr>1 else odd
        oe_r = abs(odd-even)/(d+1e-9)

        sin_m       = 1 - d/2*(1-np.cos(2*np.pi*t/P))
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


# ── Download one TIC — no threading, just fast-fail ─────────────────────────

def fetch_one(tic_id, label):
    """Returns feature_row or None. No threading — avoids Windows stdout corruption."""
    try:
        target = f"TIC {int(tic_id)}"

        # Search only — fast, no download yet
        sr = lk.search_lightcurve(target, mission="TESS", author="SPOC",
                                   exptime="long")
        if len(sr) == 0:
            sr = lk.search_lightcurve(target, mission="TESS", exptime="long")
        if len(sr) == 0:
            return None

        # Download just the first result
        lc = sr[0].download()
        if lc is None:
            return None

        lc = lc.remove_nans().normalize()
        if hasattr(lc, "remove_outliers"):
            lc = lc.remove_outliers(sigma=4)
        lc_flat = lc.flatten(window_length=401)

        t    = lc_flat.time.value
        flux = lc_flat.flux.value
        if len(t) < 200:
            return None

        feats = extract_features(t, flux)
        if feats is None:
            return None

        return feats + [label]

    except KeyboardInterrupt:
        raise   # let Ctrl+C still work
    except Exception:
        return None


# ── Fetch TOI table ───────────────────────────────────────────────────────────

def fetch_toi_table():
    print("Fetching TOI table...")

    headers = {"User-Agent": "Mozilla/5.0 exoplanet-hackathon"}

    # ── Attempt 1: ExoFOP CSV via requests (with timeout) ────────────────────
    url1 = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    try:
        r = requests.get(url1, headers=headers, timeout=30)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), comment="#", low_memory=False)
        col_map = {
            "TOI": "toi", "TIC ID": "tic_id",
            "TFOPWG Disposition": "tfopwg_disp",
            "Period (days)": "period_days",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df = df.dropna(subset=["tic_id"])
        df["tic_id"] = df["tic_id"].astype(int)
        print(f"  Got {len(df)} TOIs from ExoFOP CSV.")
        return df
    except Exception as e:
        print(f"  ExoFOP CSV failed: {e}")

    # ── Attempt 2: ExoFOP pipe-delimited ─────────────────────────────────────
    url2 = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=pipe"
    try:
        r = requests.get(url2, headers=headers, timeout=30)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), sep="|", comment="#", low_memory=False)
        col_map = {
            "TOI": "toi", "TIC ID": "tic_id",
            "TFOPWG Disposition": "tfopwg_disp",
            "Period (days)": "period_days",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df = df.dropna(subset=["tic_id"])
        df["tic_id"] = df["tic_id"].astype(int)
        print(f"  Got {len(df)} TOIs from ExoFOP pipe.")
        return df
    except Exception as e:
        print(f"  ExoFOP pipe failed: {e}")

    # ── Attempt 3: NASA Archive confirmed planets only ────────────────────────
    url3 = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        "?query=select+tic_id+from+ps"
        "+where+tran_flag+%3D+1"
        "&format=csv"
    )
    try:
        r = requests.get(url3, headers=headers, timeout=30)
        r.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(r.text), comment="#")
        df["tfopwg_disp"] = "CP"
        df = df.dropna(subset=["tic_id"])
        df["tic_id"] = df["tic_id"].astype(int)
        print(f"  Got {len(df)} confirmed planets from NASA Archive.")
        return df
    except Exception as e:
        print(f"  NASA Archive failed: {e}")

    print("  All TOI fetches failed — will use synthetic data only.")
    return pd.DataFrame()


# ── Starspot TIC list — 100 known variable/active TESS stars ─────────────────

STARSPOT_TICS = [
    # Young active stars
    149603524, 261136679, 234523599, 277539431, 355703913,
    167602025, 410214986, 199376584, 350618622, 158025034,
    # Rapid rotators / flare stars
    257527578, 144065872, 271548206, 425933644, 382188124,
    146520535, 229945932, 237192154, 441765914, 231663901,
    # Active K/M dwarfs from TESS sectors 1-5
    38846515,  206544316, 264766922, 336732616, 219854185,
    343557665, 142394656, 427344863, 159873822, 176956893,
    206609630, 281705919, 29169215,  394137592, 261257684,
    441420236, 272086159, 261775454, 142748283, 261867566,
    # More spotted stars
    348538431, 150428135, 179317684, 243187151, 219114641,
    375506058, 260004324, 293820949, 206135267, 366532638,
    354682651, 152842718, 441462736, 176606549, 207110080,
    254113311, 220459352, 348115553, 237943064, 298663873,
    # Additional variable stars observed by TESS
    470171739, 200723869, 12421862,  266744225, 332558858,
    348318948, 435907908, 268766842, 373833617, 384628774,
    243822938, 177309964, 404142847, 427344863, 341553254,
    220459352, 229580010, 459837008, 200322593, 441765914,
]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    CHECKPOINT = Path("training_checkpoint.csv")
    cols = [
        "bls_power","depth_ppm","duration_hrs","snr_stacked",
        "sec_ratio","odd_even","sin_corr","duty_cycle",
        "scatter_ppm","pp_scatter","best_period","transit_frac",
        "fap_proxy","n_transits","shape_score","trend_power",
        "depth_sig","label",
    ]

    # ── Resume from checkpoint if it exists ──────────────────────────────────
    rows   = []
    counts = {"planet": 0, "eclipsing_binary": 0, "blend": 0, "starspot": 0}
    if CHECKPOINT.exists():
        df_prev = pd.read_csv(CHECKPOINT)
        rows    = df_prev.values.tolist()
        for row in rows:
            lbl = row[-1]
            if lbl in counts:
                counts[lbl] += 1
        print(f"Resuming from checkpoint: {len(rows)} rows already done.")
        print(f"  Counts so far: {counts}")
    else:
        print("Starting fresh.")

    # ── Step 1: planets / EBs / blends from TOI table ────────────────────────
    if any(counts[k] < MAX_PER_CLASS for k in ["planet", "eclipsing_binary", "blend"]):
        toi_df = fetch_toi_table()
        if not toi_df.empty and "tfopwg_disp" in toi_df.columns:
            toi_df = toi_df.sample(frac=1, random_state=42).reset_index(drop=True)
            for _, row in toi_df.iterrows():
                disp  = str(row.get("tfopwg_disp", "")).strip().upper()
                label = DISP_TO_LABEL.get(disp)
                if label is None or label == "starspot":
                    continue
                if counts[label] >= MAX_PER_CLASS:
                    continue

                tic_id = row.get("tic_id")
                if pd.isna(tic_id):
                    continue

                print(f"  [{label:18s}] TIC {int(tic_id):12d}  "
                      f"({counts[label]+1}/{MAX_PER_CLASS})", end=" ... ", flush=True)

                result = fetch_one(tic_id, label)
                if result is not None:
                    rows.append(result)
                    counts[label] += 1
                    print("OK")
                    # Save checkpoint after every successful fetch
                    pd.DataFrame(rows, columns=cols).to_csv(CHECKPOINT, index=False)
                else:
                    print("skip")

                time.sleep(SLEEP_SEC)

                if all(counts[k] >= MAX_PER_CLASS
                       for k in ["planet", "eclipsing_binary", "blend"]):
                    break
        else:
            print("TOI table unavailable — skipping planet/EB/blend fetch.")

    # ── Step 2: starspots from dedicated list ─────────────────────────────────
    if counts["starspot"] < MAX_PER_CLASS:
        print(f"\nFetching starspot light curves...")
        for tic_id in STARSPOT_TICS:
            if counts["starspot"] >= MAX_PER_CLASS:
                break
            print(f"  [starspot          ] TIC {tic_id:12d}  "
                  f"({counts['starspot']+1}/{MAX_PER_CLASS})", end=" ... ", flush=True)
            result = fetch_one(tic_id, "starspot")
            if result is not None:
                rows.append(result)
                counts["starspot"] += 1
                print("OK")
                pd.DataFrame(rows, columns=cols).to_csv(CHECKPOINT, index=False)
            else:
                print("skip")
            time.sleep(SLEEP_SEC)

    # ── Step 3: fill any gaps with realistic synthetic data ───────────────────
    missing = {k: MAX_PER_CLASS - counts[k]
               for k in counts if counts[k] < MAX_PER_CLASS}
    if any(v > 0 for v in missing.values()):
        print(f"\nFilling gaps with synthetic data: {missing}")
        from build_real_training_data_synth import fill_synthetic
        rows.extend(fill_synthetic(missing))

    # ── Save final CSV and clean up checkpoint ────────────────────────────────
    df_out = pd.DataFrame(rows, columns=cols)
    df_out.to_csv(OUT_CSV, index=False)
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()   # delete checkpoint now that we have final CSV

    print(f"\n{'='*60}")
    print(f"Saved {len(df_out)} rows to {OUT_CSV}")
    print("Class breakdown:")
    for lbl, cnt in df_out["label"].value_counts().items():
        print(f"  {lbl:20s}: {cnt}")
    print(f"{'='*60}")
    print(f"\nNext: start server, then tell agent: "
          f"load_labeled_dataset('training_features.csv')")


if __name__ == "__main__":
    main()