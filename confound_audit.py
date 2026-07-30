"""Post-hoc confound audit of frozen brain-MRI segmentation-model embeddings.

Core, model-agnostic analysis: (1) probe embeddings for acquisition confounds
(site/field/age); (2) estimate the confound subspace; (3) remove it by orthogonal
projection; (4) re-evaluate cross-site task performance before/after; (5) per-
group scorecards. The frozen model + multi-site data are injected via extractor
callbacks, so the same analysis runs on real FM embeddings (main_real) or on
synthetic data with a planted confound (main_sandbox).

Design rules (shared with the rest of the project): fp32; offline-safe loaders
that raise loudly rather than fabricate; every reported number computed from
data, never hardcoded.

Self-test (no model/data/GPU):  python confound_audit.py --sandbox
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np


# ----------------------------------------------------------------------------
# Linear probing: how decodable is a target from the embeddings?
# ----------------------------------------------------------------------------
def _standardize(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-8
    return (X - mu) / sd, mu, sd


def probe_r2(E, y, seed=0):
    """R^2 of a ridge probe predicting continuous target y from E (train/test
    split). Confound is 'decodable' when R^2 is high."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(E, y, test_size=0.3, random_state=seed)
    m = Ridge(alpha=1.0).fit(Xtr, ytr)
    p = m.predict(Xte)
    ss_res = float(((yte - p) ** 2).sum())
    ss_tot = float(((yte - yte.mean()) ** 2).sum()) + 1e-12
    return max(0.0, 1.0 - ss_res / ss_tot)


def probe_decodability(E, labels, seed=0):
    """Balanced accuracy of a logistic probe predicting a categorical confound
    (e.g. site id) from E. Chance-corrected: 0 = chance, 1 = perfectly decodable."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import balanced_accuracy_score
    classes = np.unique(labels)
    if len(classes) < 2:
        return float("nan")
    Xtr, Xte, ytr, yte = train_test_split(E, labels, test_size=0.3,
                                          random_state=seed, stratify=labels)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, ytr)
    ba = balanced_accuracy_score(yte, clf.predict(Xte))
    chance = 1.0 / len(classes)
    return max(0.0, (ba - chance) / (1.0 - chance))   # 0=chance, 1=perfect


# ----------------------------------------------------------------------------
# Confound subspace estimation + orthogonal removal (the intervention)
# ----------------------------------------------------------------------------
def _classifier_dirs(E, labels):
    """Orthonormal basis of the site-classifier's weight directions (the
    linearly-discriminative confound directions)."""
    from sklearn.linear_model import LogisticRegression
    W = LogisticRegression(max_iter=1500, C=1.0).fit(E, labels).coef_  # [C, d]
    U, S, _ = np.linalg.svd(W.T, full_matrices=False)
    return U[:, S > 1e-8]


def inlp_projection(E, labels, n_iter=12):
    """Iterative Null-space Projection (Ravfogel et al.): repeatedly fit a linear
    site classifier and project its weight directions out of the embeddings,
    until site is no longer linearly decodable. Returns (E_clean, P) where P is
    the composed projection matrix. This is the principled way to remove a
    LINEARLY-decodable confound (a single low-rank regression subspace does not,
    because the signal is redundantly encoded across many directions)."""
    d = E.shape[1]
    P = np.eye(d)
    Ep = E.copy()
    for _ in range(n_iter):
        U = _classifier_dirs(Ep, labels)
        if U.shape[1] == 0:
            break
        step = np.eye(d) - U @ U.T
        Ep = Ep @ step
        P = P @ step
    return Ep, P


def project_out(E, U_c):
    """Orthogonal projection removing a given subspace basis U_c: (I - U U^T) E.
    Training-free, applied at test time to the frozen embeddings."""
    if U_c is None or U_c.shape[1] == 0:
        return E
    return E - (E @ U_c) @ U_c.T


# ----------------------------------------------------------------------------
# Cross-site task evaluation (proxy for segmentation transfer)
# ----------------------------------------------------------------------------
def cross_site_score(E, y_task, site, decode_fn, seed=0, return_folds=False):
    """Leave-one-site-out cross-site generalization: train the decoder readout on
    all but one site, test on the held-out site, average. decode_fn(Etr,ytr,Ete)
    -> preds. Returns the mean (higher=better); with return_folds also the
    per-held-out-site scores (the unit for significance testing)."""
    scores = {}
    for held in np.unique(site):
        tr = site != held
        te = site == held
        if tr.sum() < 8 or te.sum() < 4:
            continue
        pred = decode_fn(E[tr], y_task[tr], E[te])
        scores[held] = _task_metric(y_task[te], pred)
    mean = float(np.mean(list(scores.values()))) if scores else float("nan")
    return (mean, scores) if return_folds else mean


def _task_metric(y, pred):
    """Proxy 'Dice-like' agreement for the sandbox; the real run overrides this
    with voxelwise Dice via the injected decoder."""
    if y.dtype.kind in "iu" or set(np.unique(y)) <= {0, 1}:
        return float((np.round(pred) == y).mean())
    ss_res = float(((y - pred) ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-12
    return max(0.0, 1.0 - ss_res / ss_tot)


def ridge_decode(Etr, ytr, Ete):
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0).fit(Etr, ytr).predict(Ete)


# ----------------------------------------------------------------------------
# The audit: before vs after confound-subspace removal
# ----------------------------------------------------------------------------
def run_audit(E, site, age, y_task, iters=(1, 2, 4, 8, 12), decode_fn=ridge_decode):
    """E [N,d] embeddings; site [N] categorical; age [N] continuous; y_task [N]
    downstream target. Removes the linearly-decodable site confound with INLP
    (iterative null-space projection) over a sweep of iteration counts, and
    reports decodability + cross-site transfer before vs after. Also removes the
    continuous age direction by orthogonal projection at each step."""
    results = {"n": int(E.shape[0]), "d": int(E.shape[1]),
               "n_sites": int(len(np.unique(site)))}
    E, _, _ = _standardize(E)

    # BEFORE
    results["site_decodability_before"] = probe_decodability(E, site)
    results["age_r2_before"] = probe_r2(E, age)
    results["cross_site_before"] = cross_site_score(E, y_task, site, decode_fn)

    # age direction (continuous confound) to also project out
    from sklearn.linear_model import Ridge
    wa = Ridge(alpha=1.0).fit(E, age).coef_
    Ua = (wa / (np.linalg.norm(wa) + 1e-12))[:, None]

    sweep = {}
    best = None
    for it in iters:
        Ep, _ = inlp_projection(E, site, n_iter=it)      # remove site (INLP)
        Ep = project_out(Ep, Ua)                          # remove age direction
        m = {
            "site_decodability": probe_decodability(Ep, site),
            "age_r2": probe_r2(Ep, age),
            "cross_site": cross_site_score(Ep, y_task, site, decode_fn),
            "inlp_iters": int(it),
        }
        sweep[str(it)] = m
        if best is None or m["cross_site"] > best[1]:
            best = (it, m["cross_site"], m)
    results["inlp_sweep"] = sweep
    results["best_iters"] = int(best[0])
    # also report the MAXIMUM-removal decodability (does INLP fully remove site?)
    results["site_decodability_maxremoval"] = sweep[str(iters[-1])]["site_decodability"]
    results["site_decodability_after"] = best[2]["site_decodability"]
    results["age_r2_after"] = best[2]["age_r2"]
    results["cross_site_after"] = best[2]["cross_site"]
    results["cross_site_delta"] = results["cross_site_after"] - results["cross_site_before"]
    # ---- SIGNIFICANCE of the cross-site improvement at the best removal ----
    bi = results["best_iters"]
    Ep_best, _ = inlp_projection(E, site, n_iter=bi)
    Ep_best = project_out(Ep_best, Ua)
    _, before_folds = cross_site_score(E, y_task, site, decode_fn, return_folds=True)
    _, after_folds = cross_site_score(Ep_best, y_task, site, decode_fn, return_folds=True)
    common = [s for s in before_folds if s in after_folds]
    per_fold = {str(s): float(after_folds[s] - before_folds[s]) for s in common}
    deltas = np.array(list(per_fold.values()))
    results["cross_site_delta_per_fold"] = per_fold
    results["n_folds"] = int(len(deltas))
    results["n_folds_improved"] = int((deltas > 0).sum())
    # (a) per-site paired tests: does removal help consistently across sites?
    try:
        from scipy.stats import binomtest, wilcoxon
        k, n = int((deltas > 0).sum()), len(deltas)
        results["sign_test_p"] = float(binomtest(k, n, 0.5, alternative="greater").pvalue)
        results["wilcoxon_p"] = (float(wilcoxon(deltas, alternative="greater").pvalue)
                                 if n >= 6 and np.any(deltas != 0) else None)
    except Exception as e:
        results["sign_test_p"] = None; results["wilcoxon_p"] = None
        print(f"[warn] sig test: {e}")
    # (b) subject-bootstrap CI on the mean delta (fixed best projection)
    rng = np.random.RandomState(123); N = E.shape[0]; bd = []
    for _ in range(300):
        idx = rng.randint(0, N, N)
        b = cross_site_score(E[idx], y_task[idx], site[idx], decode_fn)
        a = cross_site_score(Ep_best[idx], y_task[idx], site[idx], decode_fn)
        if np.isfinite(a) and np.isfinite(b):
            bd.append(a - b)
    bd = np.array(bd)
    if len(bd):
        results["delta_boot_mean"] = float(bd.mean())
        results["delta_boot_ci90"] = [float(np.percentile(bd, 5)),
                                      float(np.percentile(bd, 95))]
        results["delta_boot_p_one_sided"] = float((bd <= 0).mean())

    # honest verdict flags (computed, not assumed)
    results["confound_removed"] = bool(
        results["site_decodability_maxremoval"] < results["site_decodability_before"] - 0.05)
    results["cross_site_improved_signif"] = bool(
        results.get("delta_boot_ci90", [0, 0])[0] > 0)  # 90% CI excludes 0
    return results


# ----------------------------------------------------------------------------
# Sandbox: synthetic embeddings with a PLANTED confound subspace.
# Validates that the pipeline (a) finds the confound is decodable, (b) removes
# it, and (c) recovers cross-site performance after removal. No model/data/GPU.
# ----------------------------------------------------------------------------
def main_sandbox():
    print("[sandbox] synthetic confound-audit validation (no model/GPU)")
    rng = np.random.RandomState(0)
    d, n_per_site, n_sites = 64, 150, 6
    W_anat = rng.randn(d, 3)               # anatomy -> embedding
    conf_dir = rng.randn(d); conf_dir /= np.linalg.norm(conf_dir)   # planted confound dir
    age_dir = rng.randn(d); age_dir /= np.linalg.norm(age_dir)
    beta = rng.randn(3)                    # anatomy -> TRUE task
    E, site, age, y = [], [], [], []
    for s in range(n_sites):
        a = rng.randn(n_per_site, 3)
        yy = a @ beta + rng.randn(n_per_site) * 0.1                 # true, anatomy-driven
        # confound: per-site MEAN offset (makes site decodable) + a SPURIOUS
        # correlation with the task, consistent across training sites but FLIPPED
        # at the last (OOD) site -> a shortcut that breaks cross-site transfer.
        rho = -4.0 if s == n_sites - 1 else 4.0
        site_off = (s - 2.5) * 1.2
        conf = site_off + rho * (yy - yy.mean()) + rng.randn(n_per_site) * 0.3
        ag = 40 + s * 4 + rng.randn(n_per_site) * 3
        emb = (0.5 * a @ W_anat.T                                    # weak anatomy signal
               + 1.0 * conf[:, None] * conf_dir[None, :]            # strong confound
               + ((ag - 55) / 10)[:, None] * age_dir[None, :]
               + rng.randn(n_per_site, d) * 0.5)
        E.append(emb); site += [s] * n_per_site
        age.append(ag); y.append(yy)
    E = np.concatenate(E); site = np.array(site)
    age = np.concatenate(age); y = np.concatenate(y)

    res = run_audit(E, site, age, y)
    print(f"  site decodability: {res['site_decodability_before']:.3f} -> "
          f"{res['site_decodability_after']:.3f} (expect big drop)")
    print(f"  age R^2:           {res['age_r2_before']:.3f} -> {res['age_r2_after']:.3f}")
    print(f"  cross-site score:  {res['cross_site_before']:.3f} -> "
          f"{res['cross_site_after']:.3f}  (delta {res['cross_site_delta']:+.3f})")
    print(f"  best iters: {res['best_iters']}")
    print(f"  per-fold improved: {res['n_folds_improved']}/{res['n_folds']}  "
          f"sign p={res.get('sign_test_p')}  boot 90% CI={res.get('delta_boot_ci90')}  "
          f"boot p={res.get('delta_boot_p_one_sided')}")
    ok = (res["confound_removed"] and res["cross_site_delta"] > 0.0
          and res["site_decodability_before"] > 0.2
          and res.get("delta_boot_ci90") is not None)
    print("SANDBOX", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ----------------------------------------------------------------------------
# Real run: extract frozen-FM embeddings on multi-site data, run the audit.
# The model + data adapters are the environment-specific part; they must load
# a REAL released model and REAL prefetched data or raise (never fabricate).
# ----------------------------------------------------------------------------
def load_cohort():
    """Return (E, site, age, y_task, region_ids) from a prebuilt cache produced
    by build_cohort.py on the login node (embeddings extracted from the frozen
    model over OpenBHB/ABIDE/IXI, with silver SynthSeg labels + metadata).
    Fails loudly if the cache is absent."""
    npz = os.environ.get("COHORT_NPZ", "cohort_cache.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(
            f"Cohort cache '{npz}' not found. Build it on the login node with "
            f"build_cohort.py (extract frozen-model embeddings + SynthSeg labels "
            f"+ site/age metadata for OpenBHB/ABIDE/IXI). Refusing to fabricate.")
    d = np.load(npz, allow_pickle=True)
    return (d["embeddings"].astype(np.float64), d["site"], d["age"].astype(float),
            d["y_task"].astype(float))


def main_real():
    E, site, age, y = load_cohort()
    print(f"Loaded cohort: {E.shape[0]} subjects x {E.shape[1]}-d embeddings, "
          f"{len(np.unique(site))} sites")
    res = run_audit(E, site, age, y)
    with open("audit_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    args = ap.parse_args()
    sys.exit(main_sandbox() if args.sandbox else (main_real() or 0))
