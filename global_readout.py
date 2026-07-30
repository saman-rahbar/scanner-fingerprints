#!/usr/bin/env python
"""global_readout.py -- does confound removal help GLOBAL (pooled-embedding)
cross-site prediction? This is the readout where the confound is NOT fused to a
dense spatial signal, so removal has room to help.

Leave-one-site-out clinical classification from the frozen BrainSegFounder GAP
embeddings (reusing the SAME cohort_cache_bsf.npz the audit built), for tasks
ABIDE actually labels -- ASD diagnosis (DX_GROUP) and sex -- comparing:
  * raw embeddings
  * INLP site-subspace removal (ours)
  * ComBat harmonization (baseline)
on identical LOSO balanced accuracy, with per-fold sign test + subject bootstrap.

Reuses inlp_projection/project_out (confound_audit) and combat (combat_baseline).
Self-test (no data):  python global_readout.py --sandbox
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np

from confound_audit import inlp_projection, project_out, _standardize
from combat_baseline import combat


# ----------------------------------------------------------------------------
# Leave-one-site-out balanced accuracy (the cross-site generalization metric)
# ----------------------------------------------------------------------------
def loso_balacc(E, y, site, return_folds=False, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    scores = {}
    for held in np.unique(site):
        tr = site != held
        te = site == held
        if tr.sum() < 20 or te.sum() < 8:
            continue
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0,
                                 class_weight="balanced").fit(E[tr], y[tr])
        scores[held] = balanced_accuracy_score(y[te], clf.predict(E[te]))
    mean = float(np.mean(list(scores.values()))) if scores else float("nan")
    return (mean, scores) if return_folds else mean


def _site_subspace(E_std, site, n_iter):
    _, P = inlp_projection(E_std, site, n_iter=n_iter)
    R = np.eye(P.shape[0]) - P
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    return U[:, S > 1e-6]


def run_task(E, y, site, age, task, n_iter=8):
    """before / INLP / ComBat LOSO balanced accuracy for one task, with the INLP
    vs before and INLP vs ComBat significance."""
    E0, _, _ = _standardize(E)
    out = {"task": task, "n": int(len(y)),
           "class_balance": {int(c): int((y == c).sum()) for c in np.unique(y)}}

    # variants
    U_c = _site_subspace(E0, site, n_iter)
    E_inlp = project_out(E0, U_c)
    covars = np.stack([age.astype(float), y.astype(float)], axis=1)
    E_cb, _, _ = _standardize(combat(E0, site, covars, eb=True))

    m_raw, f_raw = loso_balacc(E0, y, site, return_folds=True)
    m_inlp, f_inlp = loso_balacc(E_inlp, y, site, return_folds=True)
    m_cb, f_cb = loso_balacc(E_cb, y, site, return_folds=True)
    out.update({"balacc_raw": m_raw, "balacc_inlp": m_inlp, "balacc_combat": m_cb,
                "inlp_removed_rank": int(U_c.shape[1])})

    common = [s for s in f_raw if s in f_inlp and s in f_cb]
    d_inlp = np.array([f_inlp[s] - f_raw[s] for s in common])
    d_vs_cb = np.array([f_inlp[s] - f_cb[s] for s in common])
    out["n_folds"] = int(len(common))
    out["inlp_vs_raw_per_fold"] = {str(s): float(f_inlp[s] - f_raw[s]) for s in common}
    out["inlp_folds_improved"] = int((d_inlp > 0).sum())
    try:
        from scipy.stats import binomtest
        out["inlp_vs_raw_sign_p"] = float(binomtest((d_inlp > 0).sum(), len(d_inlp),
                                                     0.5, alternative="greater").pvalue)
        out["inlp_vs_combat_sign_p"] = float(binomtest((d_vs_cb > 0).sum(), len(d_vs_cb),
                                                       0.5, alternative="greater").pvalue)
    except Exception as ex:
        print(f"[warn] sig: {ex}")
    # subject bootstrap on the mean LOSO delta (INLP - raw)
    rng = np.random.RandomState(0); N = len(y); bd = []
    for _ in range(300):
        idx = rng.randint(0, N, N)
        a = loso_balacc(E_inlp[idx], y[idx], site[idx])
        b = loso_balacc(E0[idx], y[idx], site[idx])
        if np.isfinite(a) and np.isfinite(b):
            bd.append(a - b)
    if bd:
        bd = np.array(bd)
        out["inlp_delta_boot_mean"] = float(bd.mean())
        out["inlp_delta_boot_ci90"] = [float(np.percentile(bd, 5)),
                                       float(np.percentile(bd, 95))]
    out["inlp_beats_raw"] = bool(m_inlp > m_raw)
    out["inlp_beats_combat"] = bool(m_inlp > m_cb)
    return out


# ----------------------------------------------------------------------------
# Load embeddings + join ABIDE labels (DX_GROUP, sex) by subject id
# ----------------------------------------------------------------------------
def load_data():
    npz = os.environ.get("COHORT_NPZ", "cohort_cache_bsf.npz")
    if not os.path.exists(npz):
        raise FileNotFoundError(f"{npz} not found (the audit's embedding cache).")
    d = np.load(npz, allow_pickle=True)
    E = d["embeddings"].astype(np.float64)
    site = d["site"]; age = d["age"].astype(float)
    sex = d["y_task"].astype(int)                       # 0/1 (sex-1) from build_cohort
    sid = d["subject_id"]                               # "SITE/sub-<ID>"
    # join DX_GROUP from ABIDE phenotype (1=ASD, 2=control) -> 1/0
    dx = None
    root = os.environ.get("ABIDE_ROOT")
    if root and os.path.exists(os.path.join(root, "Phenotypic_V1_0b.csv")):
        import pandas as pd, re
        df = pd.read_csv(os.path.join(root, "Phenotypic_V1_0b.csv"))
        dxmap = {int(r.SUB_ID): int(r.DX_GROUP) for r in df.itertuples()
                 if int(r.DX_GROUP) in (1, 2)}
        dx = np.array([dxmap.get(int(re.search(r"sub-0*(\d+)", str(s)).group(1)), -1)
                       for s in sid])
        dx = np.where(dx == 1, 1, np.where(dx == 2, 0, -1))   # ASD=1, control=0, missing=-1
    return E, site, age, sex, dx


def main_real():
    E, site, age, sex, dx = load_data()
    print(f"Loaded {E.shape[0]} subjects x {E.shape[1]}-d, {len(np.unique(site))} sites")
    results = []
    # sex task
    results.append(run_task(E, sex, site, age, "sex"))
    # ASD task (if labels joined)
    if dx is not None and (dx >= 0).sum() > 40:
        m = dx >= 0
        results.append(run_task(E[m], dx[m], site[m], age[m], "asd_dx"))
    else:
        print("[info] DX_GROUP not joined (set ABIDE_ROOT); skipping ASD task")
    for r in results:
        print(f"\n== {r['task']} (n={r['n']}, balance={r['class_balance']}) ==")
        print(f"  balacc  raw={r['balacc_raw']:.3f}  inlp={r['balacc_inlp']:.3f}  "
              f"combat={r['balacc_combat']:.3f}")
        print(f"  inlp vs raw: folds {r['inlp_folds_improved']}/{r['n_folds']}, "
              f"sign p={r.get('inlp_vs_raw_sign_p')}, "
              f"boot90={r.get('inlp_delta_boot_ci90')}")
        print(f"  beats raw={r['inlp_beats_raw']}  beats combat={r['inlp_beats_combat']}")
    with open("global_readout_results.json", "w") as f:
        json.dump(results, f, indent=2)


def main_sandbox():
    """Site-imbalance confounding (the realistic ABIDE-ASD case): the label rate
    varies by site, so a classifier latches onto the strongly-encoded site signal
    and fails leave-one-site-out. ComBat -- told to preserve the label covariate --
    strips site while keeping the genuine signal and recovers cross-site accuracy;
    blind INLP removes the whole site subspace, which OVERLAPS the confounded label,
    so it does not. This validates the pipeline AND the honest method contrast."""
    print("[sandbox] global-readout LOSO under site-imbalance confounding")
    rng = np.random.RandomState(1)
    d, nps, ns, cdim = 60, 120, 6, 3
    wg = rng.randn(d); wg /= np.linalg.norm(wg)      # genuine signal direction
    conf = rng.randn(d, cdim)                         # site subspace
    ps = [0.15, 0.3, 0.45, 0.55, 0.7, 0.85]          # per-site label rate = confound
    offs = rng.randn(ns, cdim) * 2.5
    E, site, y, age = [], [], [], []
    for s in range(ns):
        yy = (rng.rand(nps) < ps[s]).astype(int)
        emb = (1.0 * (yy - 0.5))[:, None] * wg[None, :] \
            + (offs[s][None, :] + rng.randn(nps, cdim) * 0.3) @ conf.T \
            + rng.randn(nps, d) * 0.7
        E.append(emb); site += [s] * nps; y.append(yy); age.append(40 + rng.randn(nps) * 8)
    E = np.concatenate(E); site = np.array(site); y = np.concatenate(y); age = np.concatenate(age)
    r = run_task(E, y, site, age, "sandbox", n_iter=3)
    print(f"  balacc raw={r['balacc_raw']:.3f}  inlp={r['balacc_inlp']:.3f}  "
          f"combat={r['balacc_combat']:.3f}  (inlp rank {r['inlp_removed_rank']})")
    ok = r["balacc_combat"] > r["balacc_raw"] + 0.02
    print("SANDBOX", "PASS" if ok else "FAIL",
          "(covariate-preserving ComBat recovers confounded cross-site accuracy)")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    args = ap.parse_args()
    sys.exit(main_sandbox() if args.sandbox else (main_real() or 0))
