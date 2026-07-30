#!/usr/bin/env python
"""combat_baseline.py -- ComBat harmonization baseline for the confound audit.

A natural first question about "we remove the site confound by projection" is:
is that better than just running ComBat on the embeddings? This module provides
the head-to-head. It implements parametric empirical-Bayes ComBat (Johnson et al.,
Biostatistics 2007) applied to the frozen-FM bottleneck GAP embeddings, with site
as the batch variable and age + the downstream task PRESERVED as biological
covariates, then re-scores site decodability and leave-one-site-out cross-site
transfer the SAME way confound_audit.py does -- so RAW vs ComBat vs our INLP
projection are directly comparable on identical metrics.

Self-test (no model/data/GPU):  python combat_baseline.py --sandbox
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np

from confound_audit import (probe_decodability, probe_r2, cross_site_score,
                            ridge_decode, inlp_projection, project_out,
                            _standardize, load_cohort)


# ----------------------------------------------------------------------------
# Parametric empirical-Bayes ComBat (Johnson 2007), features treated independently.
# E: [N, d] samples x features; batch: [N] categorical site; covars: [N, p] design
# of PRESERVED biological covariates (already includes an intercept column).
# ----------------------------------------------------------------------------
def _aprior(g):     # method-of-moments hyperparameters for the inverse-gamma prior
    m, s2 = g.mean(), g.var()
    return (2 * s2 + m ** 2) / s2


def _bprior(g):
    m, s2 = g.mean(), g.var()
    return (m * s2 + m ** 3) / s2


def _it_eb(Zb, gamma_hat, delta_hat, g_bar, t2, a, b, tol=1e-4, maxit=200):
    """Iterative EB fixed point for one batch: shrink location gamma_star and
    scale delta_star toward their priors (parametric ComBat)."""
    n = Zb.shape[0]  # samples in this batch (per feature same n)
    g_old, d_old = gamma_hat.copy(), delta_hat.copy()
    for _ in range(maxit):
        # postmean uses the FIXED original gamma_hat (not the iterated g_old) with
        # the CURRENT delta estimate -- the neuroCombat fixed-point. Using g_old
        # here compounds shrinkage every iteration and under-corrects location.
        g_new = (t2 * n * gamma_hat + d_old * g_bar) / (t2 * n + d_old)
        sig = (Zb - g_new[None, :]) ** 2
        d_new = (0.5 * sig.sum(0) + b) / (n / 2.0 + a - 1.0)
        if (max(np.max(np.abs(g_new - g_old) / (np.abs(g_old) + 1e-8)),
                np.max(np.abs(d_new - d_old) / (np.abs(d_old) + 1e-8))) < tol):
            g_old, d_old = g_new, d_new
            break
        g_old, d_old = g_new, d_new
    return g_old, d_old


def combat(E, batch, covars, eb=True):
    """Return ComBat-harmonized embeddings [N, d]. Removes additive+multiplicative
    site effects while preserving the covariate signal in `covars`."""
    E = np.asarray(E, float)
    N, d = E.shape
    sites = np.unique(batch)
    # batch design (one-hot, no intercept) + preserved covariates
    B = np.stack([(batch == s).astype(float) for s in sites], axis=1)   # [N, nb]
    X = np.concatenate([B, covars], axis=1)                             # [N, nb+p]
    # OLS fit; grand mean uses batch-size-weighted average of batch intercepts
    beta, *_ = np.linalg.lstsq(X, E, rcond=None)                        # [nb+p, d]
    nb = B.shape[1]
    grand = (B.mean(0) @ beta[:nb])                                     # [d]
    cov_contrib = covars @ beta[nb:]                                    # [N, d]
    stand_mean = grand[None, :] + cov_contrib
    var_pooled = ((E - X @ beta) ** 2).mean(0)                          # [d]
    sd = np.sqrt(var_pooled) + 1e-8
    Z = (E - stand_mean) / sd

    E_adj = np.empty_like(E)
    for k, s in enumerate(sites):
        idx = batch == s
        Zb = Z[idx]
        gamma_hat = Zb.mean(0)                                          # [d]
        delta_hat = Zb.var(0) + 1e-8                                    # [d]
        if eb:
            g_bar, t2 = gamma_hat.mean(), gamma_hat.var()
            a, b = _aprior(delta_hat), _bprior(delta_hat)
            gamma_star, delta_star = _it_eb(Zb, gamma_hat, delta_hat, g_bar, t2, a, b)
        else:
            gamma_star, delta_star = gamma_hat, delta_hat
        Z_adj = (Zb - gamma_star[None, :]) / np.sqrt(delta_star)[None, :]
        E_adj[idx] = Z_adj * sd[None, :] + stand_mean[idx]
    return E_adj


# ----------------------------------------------------------------------------
# Head-to-head: RAW vs ComBat vs INLP (ours) on identical metrics.
# ----------------------------------------------------------------------------
def run_compare(E, site, age, y_task, inlp_iters=8, decode_fn=ridge_decode):
    E0, _, _ = _standardize(E)
    # preserved biological covariates: age + task (NO intercept -- the batch
    # one-hot columns in combat() already provide the per-batch intercepts; adding
    # an intercept here makes the design collinear and corrupts the batch fit).
    covars = np.stack([age.astype(float), y_task.astype(float)], axis=1)

    def score(Emat):
        return {"site_decodability": probe_decodability(Emat, site),
                "age_r2": probe_r2(Emat, age),
                "cross_site": cross_site_score(Emat, y_task, site, decode_fn)}

    out = {"n": int(E.shape[0]), "d": int(E.shape[1]),
           "n_sites": int(len(np.unique(site)))}
    out["raw"] = score(E0)

    # ComBat baseline
    E_cb = combat(E0, site, covars, eb=True)
    E_cb, _, _ = _standardize(E_cb)
    out["combat"] = score(E_cb)

    # ours: INLP site removal + age direction removal (matches run_audit)
    Ep, _ = inlp_projection(E0, site, n_iter=inlp_iters)
    from sklearn.linear_model import Ridge
    wa = Ridge(alpha=1.0).fit(E0, age).coef_
    Ua = (wa / (np.linalg.norm(wa) + 1e-12))[:, None]
    Ep = project_out(Ep, Ua)
    out["inlp_ours"] = score(Ep)

    # deltas vs raw (cross-site up = better; site decodability down = better)
    for k in ("combat", "inlp_ours"):
        out[k + "_delta_cross_site"] = out[k]["cross_site"] - out["raw"]["cross_site"]
        out[k + "_delta_site_dec"] = out[k]["site_decodability"] - out["raw"]["site_decodability"]
    out["ours_beats_combat_cross_site"] = bool(
        out["inlp_ours"]["cross_site"] >= out["combat"]["cross_site"])
    out["ours_beats_combat_site_removal"] = bool(
        out["inlp_ours"]["site_decodability"] <= out["combat"]["site_decodability"])
    return out


def main_sandbox():
    """Validate the ComBat implementation on data matching ComBat's own model:
    strong per-site LOCATION + SCALE batch effects across all features, on top of a
    preserved biological signal (anatomy->task, plus age). A correct ComBat drives
    site decodability toward chance WHILE preserving the task (cross-site transfer).
    Real embeddings had site decodability ~0.94, so a strong-batch-effect synthetic
    is the faithful unit test (a single weak confound direction is not)."""
    print("[sandbox] ComBat batch-effect removal (no model/GPU)")
    rng = np.random.RandomState(0)
    d, n_per_site, n_sites = 64, 150, 6
    W_anat = rng.randn(d, 3)                      # anatomy -> embedding
    beta = rng.randn(3)                           # anatomy -> task
    age_dir = rng.randn(d); age_dir /= np.linalg.norm(age_dir)
    E, site, age, y = [], [], [], []
    for s in range(n_sites):
        # strong per-site batch effect: multivariate location shift + per-feature
        # scale change -- exactly ComBat's additive+multiplicative model.
        gamma_s = rng.randn(d) * 2.5                          # location (per site)
        delta_s = np.exp(rng.randn(d) * 0.5)                 # noise scale (per site)
        a = rng.randn(n_per_site, 3)
        yy = a @ beta + rng.randn(n_per_site) * 0.1          # biological task
        ag = 40 + rng.randn(n_per_site) * 8                  # age INDEPENDENT of site
        bio = (a @ W_anat.T + ((ag - 40) / 10)[:, None] * age_dir[None, :])
        # ComBat model: bio signal + per-site location + per-site-SCALED noise
        emb = bio + gamma_s[None, :] + delta_s[None, :] * rng.randn(n_per_site, d) * 0.6
        E.append(emb); site += [s] * n_per_site; age.append(ag); y.append(yy)
    E = np.concatenate(E); site = np.array(site)
    age = np.concatenate(age); y = np.concatenate(y)

    r = run_compare(E, site, age, y)
    for k in ("raw", "combat", "inlp_ours"):
        print(f"  {k:10s}  site_dec={r[k]['site_decodability']:.3f}  "
              f"cross_site={r[k]['cross_site']:.3f}")
    ok = (r["raw"]["site_decodability"] > 0.5                     # strong batch effect present
          and r["combat"]["site_decodability"] < r["raw"]["site_decodability"] - 0.2
          and r["combat"]["cross_site"] > r["raw"]["cross_site"] - 0.05)  # task preserved
    print("SANDBOX", "PASS" if ok else "FAIL",
          "(ComBat removes site batch effect while preserving task)")
    return 0 if ok else 1


def main_real():
    E, site, age, y = load_cohort()
    print(f"Loaded cohort: {E.shape[0]} subjects x {E.shape[1]}-d, "
          f"{len(np.unique(site))} sites")
    r = run_compare(E, site, age, y)
    with open("combat_compare.json", "w") as f:
        json.dump(r, f, indent=2)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    args = ap.parse_args()
    sys.exit(main_sandbox() if args.sandbox else (main_real() or 0))
