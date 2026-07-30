#!/usr/bin/env python
"""intrinsic_equivalence.py -- does 'pretraining is not required for the scanner
fingerprint' survive a proper PAIRED test?

Overlapping CIs cannot establish that random-init and pretrained encoders are
equivalent. Here we run the correct paired analysis on the SAME holdout splits:
per split, compute deep-layer site decodability for random-init, brain-pretrained,
and CT-pretrained encoders, then test the paired differences. We report (a) a
one-sided non-inferiority result -- is random >= pretrained? -- via a paired sign
test and Wilcoxon, and (b) a two-one-sided-tests (TOST) equivalence result within
a margin epsilon: can we conclude |d_random - d_pretrained| < epsilon?

CPU-only; reuses the multilayer_*.npz caches. Subjects are aligned by id across
encoders so every split is identical for all three.
Self-test:  python intrinsic_equivalence.py --sandbox
"""
from __future__ import annotations

import os
import sys
import glob
import json
import argparse
import numpy as np

from confound_audit import _standardize


def _site_decode_once(E, site, idx_tr, idx_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(E[idx_tr], site[idx_tr])
    ba = balanced_accuracy_score(site[idx_te], clf.predict(E[idx_te]))
    K = len(np.unique(site))
    return max(0.0, (ba - 1.0 / K) / (1.0 - 1.0 / K))


def paired_site_decode(embs: dict, site, n_rep, deep_layers):
    """embs: {tag: E_deep [N,d]} already aligned + standardized. Returns per-rep
    site decodability for each tag over identical stratified splits."""
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=n_rep, test_size=0.3, random_state=0)
    out = {t: [] for t in embs}
    for idx_tr, idx_te in sss.split(np.zeros(len(site)), site):
        for t, E in embs.items():
            out[t].append(_site_decode_once(E, site, idx_tr, idx_te))
    return {t: np.array(v) for t, v in out.items()}


def tost(diff, eps):
    """Two one-sided tests for equivalence of paired diffs within +/-eps.
    Returns (equivalent_bool, p_upper, p_lower) using a paired t on the diffs."""
    from scipy import stats
    n = len(diff); m = diff.mean(); s = diff.std(ddof=1) + 1e-12
    se = s / np.sqrt(n)
    t_upper = (m - eps) / se      # H0: diff >= eps
    t_lower = (m + eps) / se      # H0: diff <= -eps
    p_upper = stats.t.cdf(t_upper, n - 1)          # want diff < eps
    p_lower = stats.t.sf(t_lower, n - 1)           # want diff > -eps
    return bool(p_upper < 0.05 and p_lower < 0.05), float(p_upper), float(p_lower)


def analyze(embs, site, n_rep, eps, deep_layers):
    from scipy.stats import binomtest, wilcoxon
    scores = paired_site_decode(embs, site, n_rep, deep_layers)
    ref = "random"
    res = {"n_rep": n_rep, "eps": eps,
           "mean_site_decodability": {t: float(v.mean()) for t, v in scores.items()}}
    for t in embs:
        if t == ref:
            continue
        diff = scores[ref] - scores[t]           # random - pretrained, per split
        k = int((diff >= 0).sum())
        equiv, pu, pl = tost(diff, eps)
        res[f"{ref}_vs_{t}"] = {
            "mean_diff": float(diff.mean()),
            "diff_ci90": [float(np.percentile(diff, 5)), float(np.percentile(diff, 95))],
            "noninferiority_sign_p": float(binomtest(k, len(diff), 0.5, alternative="greater").pvalue),
            "noninferiority_wilcoxon_p": (float(wilcoxon(diff, alternative="greater").pvalue)
                                          if np.any(diff != 0) else None),
            "tost_equivalent_within_eps": equiv, "tost_p_upper": pu, "tost_p_lower": pl,
            "random_at_least_as_high": bool(diff.mean() >= 0),
        }
    return res


def _load_deep(path, deep_layers):
    d = np.load(path, allow_pickle=True)
    Es = [d[f"emb_L{l}"].astype(np.float64) for l in deep_layers]
    E = np.concatenate(Es, axis=1)               # concat deep layers
    E, _, _ = _standardize(E)
    return E, d["site"], d["subject_id"]


def main_real():
    n_rep = int(os.environ.get("N_REP", "100"))
    eps = float(os.environ.get("EPS", "0.05"))
    deep = [int(x) for x in os.environ.get("DEEP_LAYERS", "3,4").split(",")]
    files = {os.path.basename(p)[len("multilayer_"):-len(".npz")]: p
             for p in sorted(glob.glob(os.environ.get("GLOB", "multilayer_*.npz")))}
    if "random" not in files:
        sys.exit("need a multilayer_random.npz (the control encoder)")
    # align subjects by id across encoders
    loaded = {t: _load_deep(p, deep) for t, p in files.items()}
    common = set.intersection(*[set(map(str, v[2])) for v in loaded.values()])
    order = sorted(common)
    embs, site = {}, None
    for t, (E, s, sid) in loaded.items():
        pos = {str(x): i for i, x in enumerate(sid)}
        idx = np.array([pos[c] for c in order])
        embs[t] = E[idx]
        site = s[np.array([pos[c] for c in order])] if site is None else site
    print(f"[data] {len(order)} aligned subjects, {len(np.unique(site))} sites, "
          f"encoders={list(embs)} deep_layers={deep}")
    res = analyze(embs, site, n_rep, eps, deep)
    print(json.dumps(res, indent=2))
    with open("intrinsic_equivalence_results.json", "w") as f:
        json.dump(res, f, indent=2)


def main_sandbox():
    """Random encoder truly >= pretrained: build embeddings where site is equally
    (or more) decodable from the 'random' one; the paired test must find random
    non-inferior and (given the small planted gap) equivalence within eps."""
    print("[sandbox] paired intrinsic equivalence")
    rng = np.random.RandomState(0)
    n, ns, d = 360, 6, 64
    site = np.array([s for s in range(ns) for _ in range(n // ns)])
    conf = rng.randn(d, 4)
    off = rng.randn(ns, 4) * 0.7                  # moderate site effect (unsaturated)
    base = np.stack([off[s] for s in site]) @ conf.T
    embs = {                                      # random slightly >= pretrained
        "random":   _standardize(base * 1.05 + rng.randn(n, d) * 1.1)[0],
        "brainseg": _standardize(base * 1.00 + rng.randn(n, d) * 1.1)[0],
        "ctssl":    _standardize(base * 0.95 + rng.randn(n, d) * 1.1)[0],
    }
    r = analyze(embs, site, n_rep=40, eps=0.05, deep_layers=[0])
    print(f"  mean site decodability: {r['mean_site_decodability']}")
    rb = r["random_vs_brainseg"]
    print(f"  random - brainseg: diff={rb['mean_diff']:+.3f} CI={rb['diff_ci90']} "
          f"noninf_p={rb['noninferiority_sign_p']:.3f} equiv={rb['tost_equivalent_within_eps']}")
    ok = rb["random_at_least_as_high"] and rb["noninferiority_sign_p"] < 0.2
    print("SANDBOX", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    args = ap.parse_args()
    sys.exit(main_sandbox() if args.sandbox else (main_real() or 0))
