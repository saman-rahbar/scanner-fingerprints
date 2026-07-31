#!/usr/bin/env python
"""population_adjust.py -- how much of site decodability is ACQUISITION vs
POPULATION? Site labels bundle scanner/protocol with the site's subject
population (age, sex, diagnosis differ across sites). We residualize the
embeddings against the measured population covariates (age, sex, diagnosis) --
removing the linear component predictable from them -- and re-measure site
decodability. If site stays high after adjustment, it is not merely population;
if it drops substantially, population explains part of it. We report both.

CPU-only; reuses the multilayer_*.npz caches. Self-test: python population_adjust.py --sandbox
"""
from __future__ import annotations

import os
import sys
import glob
import json
import argparse
import numpy as np

from confound_audit import _standardize, probe_decodability
from scanner_dominance import _join_dx, _join_pheno


def _design(age, sex, dx, n):
    """Population covariate design [n, p] with an intercept; NaNs mean-imputed so
    every subject is kept. Columns present only when the variable has signal."""
    cols, names = [np.ones(n)], ["intercept"]
    for v, nm in [(age, "age"), (sex, "sex"), (dx, "dx")]:
        if v is None:
            continue
        v = np.asarray(v, float)
        finite = np.isfinite(v)
        if finite.sum() < 0.5 * n or len(np.unique(v[finite])) < 2:
            continue
        vv = v.copy()
        vv[~finite] = v[finite].mean()
        cols.append((vv - vv.mean()) / (vv.std() + 1e-8)); names.append(nm)
    return np.stack(cols, axis=1), names


def residualize(E, C):
    """Remove the linear component of E predictable from covariate design C."""
    beta, *_ = np.linalg.lstsq(C, E, rcond=None)
    return E - C @ beta


def run(E, site, age, sex, dx):
    Es, _, _ = _standardize(E)
    C, names = _design(age, sex, dx, len(site))
    Eadj, _, _ = _standardize(residualize(Es, C))
    before = probe_decodability(Es, site)
    after = probe_decodability(Eadj, site)
    return {"covariates_adjusted": names[1:], "n": int(len(site)),
            "site_decodability_raw": before,
            "site_decodability_pop_adjusted": after,
            "drop": float(before - after),
            "retained_fraction": float(after / (before + 1e-9))}


def main_real():
    files = {os.path.basename(p)[len("multilayer_"):-len(".npz")]: p
             for p in sorted(glob.glob(os.environ.get("GLOB", "multilayer_*.npz")))}
    if not files:
        sys.exit("no multilayer_*.npz found")
    deep = [int(x) for x in os.environ.get("DEEP_LAYERS", "3,4").split(",")]
    out = {}
    for tag, p in files.items():
        d = np.load(p, allow_pickle=True)
        E = np.concatenate([d[f"emb_L{l}"].astype(np.float64) for l in deep], axis=1)
        site = d["site"]; age = d["age"].astype(float); sex = d["sex"].astype(float)
        dx = _join_dx(d["subject_id"]) if "subject_id" in d.files else None
        if "subject_id" in d.files:
            aj, sj = _join_pheno(d["subject_id"])
            if aj is not None:
                age = np.where(np.isfinite(age), age, aj)
                sex = np.where(np.isfinite(sex), sex, sj)
        dxf = dx.astype(float) if dx is not None else None
        if dxf is not None:
            dxf[dxf < 0] = np.nan
        out[tag] = run(E, site, age, sex, dxf)
        r = out[tag]
        print(f"{tag:10s} site raw={r['site_decodability_raw']:.3f} -> "
              f"pop-adjusted={r['site_decodability_pop_adjusted']:.3f} "
              f"(retained {100*r['retained_fraction']:.0f}%; adjusted for {r['covariates_adjusted']})")
    with open("population_adjust_results.json", "w") as f:
        json.dump(out, f, indent=2)


def main_sandbox():
    """Site is driven by a STRONG pure-acquisition offset plus a WEAK component
    that correlates with age; residualizing age should remove only the weak part,
    so site stays high -- demonstrating the analysis distinguishes acquisition
    from population."""
    print("[sandbox] population-adjusted site decodability")
    rng = np.random.RandomState(0)
    d, nps, ns = 48, 120, 6
    site = np.array([s for s in range(ns) for _ in range(nps)])
    age = np.array([40 + s * 3 for s in site], float) + rng.randn(len(site)) * 4  # age ~ site
    sex = rng.randint(0, 2, len(site)).astype(float)
    acq = rng.randn(ns, d) * 2.0                       # strong per-site acquisition
    age_dir = rng.randn(d)
    E = (np.stack([acq[s] for s in site])              # pure acquisition (dominant)
         + ((age - age.mean()) / 10)[:, None] * age_dir[None, :] * 0.6   # weak age component
         + rng.randn(len(site), d) * 0.6)
    r = run(E, site, age, sex, None)
    print(f"  site raw={r['site_decodability_raw']:.3f} -> adjusted={r['site_decodability_pop_adjusted']:.3f}")
    ok = r["site_decodability_pop_adjusted"] > 0.5 and r["retained_fraction"] > 0.6
    print("SANDBOX", "PASS" if ok else "FAIL", "(site survives population adjustment)")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    a = ap.parse_args()
    sys.exit(main_sandbox() if a.sandbox else (main_real() or 0))
