#!/usr/bin/env python
"""raw_voxel_baseline.py -- is the site fingerprint already in the RAW image?

The intrinsic result (random encoders separate sites) suggests the fingerprint is
low-level image statistics, not anything the network computes. We test that
directly: decode site from heavily downsampled RAW preprocessed voxels, with no
encoder at all. If raw voxels already give ~0.9 site decodability, the fingerprint
lives in the input and any encoder (trained or random) merely preserves it.

Reuses the same preprocessing as the encoders (RAS, resample, scale, 96^3), then
block-averages to a coarse grid and flattens. CPU-only.
Self-test: python raw_voxel_baseline.py --sandbox
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np

from confound_audit import _standardize, probe_decodability, probe_r2

GRID = int(os.environ.get("RAW_GRID", "12"))    # coarse grid side (96 -> 12 = 8^3 blocks)


def downsample(vol96, k):
    """[96,96,96] -> [k,k,k] block-mean -> flat [k^3]. 96 must be divisible by k."""
    b = vol96.shape[0] // k
    v = vol96[:k * b, :k * b, :k * b].reshape(k, b, k, b, k, b)
    return v.mean(axis=(1, 3, 5)).ravel()


def main_real():
    import build_cohort as B
    root = os.environ["ABIDE_ROOT"]
    try:
        pheno = B.load_phenotype()
    except Exception as ex:
        print(f"[warn] no phenotype ({str(ex)[:50]})"); pheno = {}
    tfm = B.make_transform()
    subs = sorted(B.iter_t1(root), key=lambda x: x[1])
    cap = int(os.environ.get("SUBJ_PER_SITE", "60"))
    import re
    feats, sites, ages, sexes = [], [], [], []
    persite = {}
    for k, (sid, t1) in enumerate(subs):
        rel = os.path.relpath(t1, root); site = rel.split(os.sep)[0]
        if cap and persite.get(site, 0) >= cap:
            continue
        try:
            vol = np.asarray(tfm(t1)[0], dtype=np.float64)      # [96,96,96]
            feats.append(downsample(vol, GRID))
        except Exception as ex:
            print(f"  [skip] {sid}: {str(ex)[:60]}"); continue
        age, sex = (np.nan, np.nan)
        if sid in pheno:
            _, a, s = pheno[sid]
            age = a if np.isfinite(a) else np.nan
            sex = (s - 1) if s in (1, 2) else np.nan
        sites.append(site); ages.append(age); sexes.append(sex)
        persite[site] = persite.get(site, 0) + 1
        if (k + 1) % 50 == 0:
            print(f"  ...{k+1}/{len(subs)} ({len(sites)} kept)")
    if len(sites) < 40:
        sys.exit("too few subjects")
    X = _standardize(np.stack(feats))[0]
    site = np.array(sites); age = np.array(ages, float); sex = np.array(sexes, float)
    res = {"grid": GRID, "n": len(site), "dim": int(X.shape[1]),
           "raw_voxel_site_decodability": probe_decodability(X, site)}
    ms = np.isfinite(sex)
    res["raw_voxel_sex_decodability"] = (probe_decodability(X[ms], sex[ms].astype(int))
                                         if ms.sum() > 40 else None)
    ma = np.isfinite(age)
    res["raw_voxel_age_r2"] = probe_r2(X[ma], age[ma]) if ma.sum() > 40 else None
    print(json.dumps(res, indent=2))
    with open("raw_voxel_baseline_results.json", "w") as f:
        json.dump(res, f, indent=2)


def main_sandbox():
    """Coarse raw voxels with a per-site intensity/geometry offset must be
    site-decodable, mirroring how scanners differ in low-level statistics."""
    print("[sandbox] raw-voxel site decodability")
    rng = np.random.RandomState(0)
    ns, nps, k = 6, 100, 8
    site = np.array([s for s in range(ns) for _ in range(nps)])
    base = rng.randn(ns, k * k * k) * 1.5              # per-site low-level pattern
    X = np.stack([base[s] for s in site]) + rng.randn(len(site), k * k * k) * 1.0
    # also sanity-check the downsample op
    v = rng.randn(96, 96, 96); assert downsample(v, 12).shape == (1728,)
    d = probe_decodability(_standardize(X)[0], site)
    print(f"  raw-voxel site decodability = {d:.3f}")
    ok = d > 0.5
    print("SANDBOX", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    a = ap.parse_args()
    sys.exit(main_sandbox() if a.sandbox else (main_real() or 0))
