#!/usr/bin/env python
"""scanner_dominance.py -- the "scanner > brain" matrix.

For each frozen encoder (multilayer_<tag>.npz from extract_multilayer.py) and each
layer, quantify how decodable ACQUISITION SITE is versus CLINICAL/DEMOGRAPHIC
variables (age, sex, ASD) from the pooled embedding. The scanner-dominance ratio
= site decodability / best clinical decodability. A ratio >> 1 means the frozen
representation carries more information about the scanner than about the patient.

Reuses the validated probes in confound_audit. CPU-only; runs in seconds.
Self-test (no data):  python scanner_dominance.py --sandbox
"""
from __future__ import annotations

import os
import re
import sys
import glob
import json
import argparse
import numpy as np

from confound_audit import probe_decodability, probe_r2, _standardize


def _join_dx(subject_id):
    """ASD label (1) / control (0) / missing (-1) per subject, from ABIDE pheno."""
    root = os.environ.get("ABIDE_ROOT")
    if not root:
        return None
    pheno_name = os.environ.get("PHENO_CSV", "Phenotypic_V1_0b.csv")
    csv = pheno_name if os.path.isabs(pheno_name) else os.path.join(root, pheno_name)
    if not os.path.exists(csv):
        return None
    import pandas as pd
    df = pd.read_csv(csv, encoding="latin-1")
    df.columns = df.columns.str.strip()
    dxmap = {}
    for r in df.itertuples():
        try:
            if int(r.DX_GROUP) in (1, 2):
                dxmap[int(r.SUB_ID)] = int(r.DX_GROUP)
        except Exception:
            pass
    out = []
    for s in subject_id:
        m = re.search(r"sub-0*(\d+)", str(s))
        g = dxmap.get(int(m.group(1)), 0) if m else 0
        out.append(1 if g == 1 else (0 if g == 2 else -1))
    return np.array(out)


def _join_pheno(subject_id):
    """Join (age, sex 0/1) from the phenotype by subject id, so clinical labels
    can be attached at ANALYSIS time (no re-extraction needed if the embeddings
    were extracted site-only). Returns (age[], sex[]) or (None, None)."""
    root = os.environ.get("ABIDE_ROOT")
    if not root:
        return None, None
    pheno_name = os.environ.get("PHENO_CSV", "Phenotypic_V1_0b.csv")
    csv = pheno_name if os.path.isabs(pheno_name) else os.path.join(root, pheno_name)
    if not os.path.exists(csv) or os.path.getsize(csv) == 0:
        return None, None
    import pandas as pd
    df = pd.read_csv(csv, encoding="latin-1")
    df.columns = df.columns.str.strip()               # ABIDE-II has 'AGE_AT_SCAN '
    amap = {}
    for r in df.itertuples():
        try:
            amap[int(r.SUB_ID)] = (float(r.AGE_AT_SCAN), int(r.SEX))
        except Exception:
            pass
    age, sex = [], []
    for s in subject_id:
        m = re.search(r"sub-0*(\d+)", str(s))
        a, x = amap.get(int(m.group(1)), (np.nan, 0)) if m else (np.nan, 0)
        age.append(a if np.isfinite(a) else np.nan)
        sex.append((x - 1) if x in (1, 2) else np.nan)
    return np.array(age, float), np.array(sex, float)


def decodabilities(E, site, age, sex, dx):
    """All chance-corrected decodabilities for one embedding matrix. Clinical
    targets with missing (NaN) labels are masked per-target and reported as NaN
    (so site always computes even when the phenotype is unavailable)."""
    Es, _, _ = _standardize(E)
    row = {"site": probe_decodability(Es, site)}
    ms = np.isfinite(sex)
    row["sex"] = (probe_decodability(Es[ms], sex[ms].astype(int))
                  if ms.sum() > 40 and len(np.unique(sex[ms])) > 1 else float("nan"))
    ma = np.isfinite(age)
    row["age_r2"] = probe_r2(Es[ma], age[ma]) if ma.sum() > 40 else float("nan")
    if dx is not None and (dx >= 0).sum() > 40:
        m = dx >= 0
        row["asd"] = probe_decodability(Es[m], dx[m].astype(int))
    clin = [v for v in [row["sex"], row["age_r2"], row.get("asd", float("nan"))]
            if np.isfinite(v)]
    row["clinical_peak"] = float(max(clin)) if clin else float("nan")
    row["scanner_dominance"] = (float(row["site"] / (row["clinical_peak"] + 1e-3))
                                if clin else float("nan"))
    return row


def analyze_file(path):
    d = np.load(path, allow_pickle=True)
    layer_keys = sorted([k for k in d.files if k.startswith("emb_L")],
                        key=lambda k: int(k.split("L")[1]))
    site = d["site"]; age = d["age"].astype(float); sex = d["sex"].astype(float)
    dx = _join_dx(d["subject_id"]) if "subject_id" in d.files else None
    if "subject_id" in d.files:                       # fill NaN clinical from pheno
        aj, sj = _join_pheno(d["subject_id"])
        if aj is not None:
            age = np.where(np.isfinite(age), age, aj)
            sex = np.where(np.isfinite(sex), sex, sj)
    mask = _site_mask(site)                            # optional N_SITES matching
    site, age, sex = site[mask], age[mask], sex[mask]
    dx = dx[mask] if dx is not None else None
    out = {"n": int(mask.sum()), "n_sites": int(len(np.unique(site))), "layers": {}}
    for k in layer_keys:
        E = d[k].astype(np.float64)[mask]
        out["layers"][k] = {"dim": int(E.shape[1]),
                            **decodabilities(E, site, age, sex, dx)}
    return out


def _site_mask(site):
    """Optional matching: N_SITES keeps the N largest sites (for an apples-to-
    apples k-way site-decodability comparison across cohorts with different site
    counts); N_PER_SITE subsamples each kept site to a fixed n. Default: keep all."""
    from collections import Counter
    n_sites = int(os.environ.get("N_SITES", "0"))
    n_per = int(os.environ.get("N_PER_SITE", "0"))
    keep_sites = ([s for s, _ in Counter(site).most_common(n_sites)]
                  if n_sites and len(np.unique(site)) > n_sites else list(np.unique(site)))
    mask = np.zeros(len(site), bool)
    rng = np.random.RandomState(0)
    for s in keep_sites:
        idx = np.where(site == s)[0]
        if n_per and len(idx) > n_per:
            idx = rng.choice(idx, n_per, replace=False)
        mask[idx] = True
    return mask


def main_real():
    tag_files = {}
    for p in sorted(glob.glob(os.environ.get("GLOB", "multilayer_*.npz"))):
        tag = os.path.basename(p)[len("multilayer_"):-len(".npz")]
        tag_files[tag] = p
    if not tag_files:
        sys.exit("no multilayer_*.npz found (run extract_multilayer.py first)")
    results = {}
    for tag, p in tag_files.items():
        print(f"\n### model: {tag}  ({p})")
        r = analyze_file(p); results[tag] = r
        print(f"  {'layer':7s} {'dim':>5s} {'site':>6s} {'sex':>6s} "
              f"{'age_r2':>7s} {'asd':>6s} {'scanner/clinical':>17s}")
        for k, row in r["layers"].items():
            print(f"  {k:7s} {row['dim']:5d} {row['site']:6.3f} {row['sex']:6.3f} "
                  f"{row['age_r2']:7.3f} {row.get('asd', float('nan')):6.3f} "
                  f"{row['scanner_dominance']:17.1f}")
    with open("scanner_dominance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[done] wrote scanner_dominance_results.json")


def main_sandbox():
    """Synthetic 2-model, 2-layer check: an embedding where site is strongly
    encoded and the clinical signal is weak must yield scanner_dominance > 1."""
    print("[sandbox] scanner-dominance matrix")
    rng = np.random.RandomState(0)
    n, ns = 480, 6
    site = np.array([s for s in range(ns) for _ in range(n // ns)])
    age = 40 + rng.randn(n) * 8
    sex = rng.randint(0, 2, n).astype(float)
    dx = rng.randint(0, 2, n)
    # strong per-site offsets (scanner), weak sex signal, no age/asd signal
    d = 48
    conf = rng.randn(d, 4)
    w_sex = rng.randn(d) * 0.25
    off = rng.randn(ns, 4) * 2.0
    E = (sex[:, None] * w_sex[None, :]
         + np.stack([off[s] for s in site]) @ conf.T + rng.randn(n, d) * 0.6)
    row = decodabilities(E, site, age, sex, dx)
    print(f"  site={row['site']:.3f}  sex={row['sex']:.3f}  age_r2={row['age_r2']:.3f}  "
          f"asd={row.get('asd', float('nan')):.3f}  dominance={row['scanner_dominance']:.1f}")
    ok = row["site"] > 0.5 and row["scanner_dominance"] > 1.0
    print("SANDBOX", "PASS" if ok else "FAIL", "(scanner more decodable than clinical)")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    args = ap.parse_args()
    sys.exit(main_sandbox() if args.sandbox else (main_real() or 0))
