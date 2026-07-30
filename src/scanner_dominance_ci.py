#!/usr/bin/env python
"""scanner_dominance_ci.py -- the scanner-vs-clinical matrix with CONFIDENCE
INTERVALS (repeated stratified splits) and a NONLINEAR (MLP) probe alongside the
linear one. Hardens the "scanner > brain" figure for a journal:
  * every decodability gets a mean + 90% interval over repeated holdouts;
  * an MLP probe shows the linear numbers are a floor (nonlinear >= linear).

CPU-only; reuses the multilayer_*.npz caches from extract_multilayer.py.
Self-test:  python scanner_dominance_ci.py --sandbox
"""
from __future__ import annotations

import os
import re
import sys
import glob
import json
import argparse
import warnings
import numpy as np

from scanner_dominance import _join_dx, _join_pheno, _site_mask
from confound_audit import _standardize

try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:
    pass


def _cat_once(E, labels, probe, seed):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    classes = np.unique(labels)
    if len(classes) < 2 or min(np.bincount(labels.astype(int))) < 4:
        return float("nan")
    Xtr, Xte, ytr, yte = train_test_split(E, labels, test_size=0.3,
                                          random_state=seed, stratify=labels)
    clf = (LogisticRegression(max_iter=1000, C=1.0) if probe == "linear"
           else MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=300,
                              random_state=seed))
    clf.fit(Xtr, ytr)
    ba = balanced_accuracy_score(yte, clf.predict(Xte))
    return max(0.0, (ba - 1.0 / len(classes)) / (1.0 - 1.0 / len(classes)))


def _reg_once(E, y, probe, seed):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    Xtr, Xte, ytr, yte = train_test_split(E, y, test_size=0.3, random_state=seed)
    m = (Ridge(alpha=1.0) if probe == "linear"
         else MLPRegressor(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=300,
                          random_state=seed))
    m.fit(Xtr, ytr); p = m.predict(Xte)
    ss_res = float(((yte - p) ** 2).sum()); ss_tot = float(((yte - yte.mean()) ** 2).sum()) + 1e-12
    return max(0.0, 1.0 - ss_res / ss_tot)


def _ci(once_fn, n_rep):
    vals = np.array([once_fn(s) for s in range(n_rep)], float)
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return {"mean": float(vals.mean()), "lo": float(np.percentile(vals, 5)),
            "hi": float(np.percentile(vals, 95))}


def decodabilities_ci(E, site, age, sex, dx, n_rep, probe):
    Es, _, _ = _standardize(E)
    nan_ci = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    row = {"site": _ci(lambda s: _cat_once(Es, site_int(site), probe, s), n_rep)}
    ms = np.isfinite(sex)
    row["sex"] = (_ci(lambda s: _cat_once(Es[ms], sex[ms].astype(int), probe, s), n_rep)
                  if ms.sum() > 40 and len(np.unique(sex[ms])) > 1 else nan_ci)
    ma = np.isfinite(age)
    row["age_r2"] = (_ci(lambda s: _reg_once(Es[ma], age[ma], probe, s), n_rep)
                     if ma.sum() > 40 else nan_ci)
    if dx is not None and (dx >= 0).sum() > 40:
        m = dx >= 0
        row["asd"] = _ci(lambda s: _cat_once(Es[m], dx[m].astype(int), probe, s), n_rep)
    clin = [v["mean"] for v in [row["sex"], row["age_r2"], row.get("asd", nan_ci)]
            if np.isfinite(v["mean"])]
    peak = float(max(clin)) if clin else float("nan")
    row["clinical_peak"] = peak
    row["scanner_dominance"] = (float(row["site"]["mean"] / (peak + 1e-3))
                                if clin else float("nan"))
    return row


def site_int(site):
    _, inv = np.unique(site, return_inverse=True)
    return inv


def analyze_file(path, n_rep):
    d = np.load(path, allow_pickle=True)
    lk = sorted([k for k in d.files if k.startswith("emb_L")],
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
    for k in lk:
        E = d[k].astype(np.float64)[mask]
        out["layers"][k] = {"dim": int(E.shape[1]),
                            "linear": decodabilities_ci(E, site, age, sex, dx, n_rep, "linear"),
                            "mlp": decodabilities_ci(E, site, age, sex, dx, n_rep, "mlp")}
    return out


def main_real():
    n_rep = int(os.environ.get("N_REP", "50"))
    files = {os.path.basename(p)[len("multilayer_"):-len(".npz")]: p
             for p in sorted(glob.glob(os.environ.get("GLOB", "multilayer_*.npz")))}
    if not files:
        sys.exit("no multilayer_*.npz found")
    results = {}
    for tag, p in files.items():
        print(f"\n### model: {tag}  (n_rep={n_rep})")
        r = analyze_file(p, n_rep); results[tag] = r
        print(f"  {'layer':6s} {'site(lin)':>16s} {'site(mlp)':>16s} "
              f"{'clin_peak(lin)':>15s} {'dom(lin)':>9s}")
        for k, row in r["layers"].items():
            sl, sm = row["linear"]["site"], row["mlp"]["site"]
            print(f"  {k:6s} {sl['mean']:.2f}[{sl['lo']:.2f},{sl['hi']:.2f}]   "
                  f"{sm['mean']:.2f}[{sm['lo']:.2f},{sm['hi']:.2f}]   "
                  f"{row['linear']['clinical_peak']:>13.2f}   "
                  f"{row['linear']['scanner_dominance']:>7.1f}")
    with open("scanner_dominance_ci_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[done] wrote scanner_dominance_ci_results.json")


def main_sandbox():
    print("[sandbox] scanner-dominance with CIs + nonlinear probe")
    rng = np.random.RandomState(0)
    n, ns, d = 480, 6, 48
    site = np.array([s for s in range(ns) for _ in range(n // ns)])
    age = 40 + rng.randn(n) * 8
    sex = rng.randint(0, 2, n).astype(float)
    dx = rng.randint(0, 2, n)
    conf = rng.randn(d, 4); w_sex = rng.randn(d) * 0.25
    off = rng.randn(ns, 4) * 2.0
    E = (sex[:, None] * w_sex[None, :]
         + np.stack([off[s] for s in site]) @ conf.T + rng.randn(n, d) * 0.6)
    lin = decodabilities_ci(E, site, age, sex, dx, n_rep=20, probe="linear")
    mlp = decodabilities_ci(E, site, age, sex, dx, n_rep=20, probe="mlp")
    print(f"  site linear={lin['site']['mean']:.3f}[{lin['site']['lo']:.2f},{lin['site']['hi']:.2f}]"
          f"  mlp={mlp['site']['mean']:.3f}  clin_peak={lin['clinical_peak']:.3f}"
          f"  dominance={lin['scanner_dominance']:.1f}")
    ok = (lin["site"]["lo"] > 0.5 and lin["scanner_dominance"] > 1.0
          and np.isfinite(mlp["site"]["mean"]))
    print("SANDBOX", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    args = ap.parse_args()
    sys.exit(main_sandbox() if args.sandbox else (main_real() or 0))
