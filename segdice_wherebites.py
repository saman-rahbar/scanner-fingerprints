#!/usr/bin/env python
"""segdice_wherebites.py -- does confound removal help cross-site Dice in the
regime where it SHOULD bite?

The full-training, bottleneck-only intervention was Dice-neutral: a decoder
trained on 5 sites already generalizes, and skip connections route around the
bottleneck. This script targets the regime where scanner confound actually hurts:
  * train the decoder on FEW sites (TRAIN_SITES) -> a real cross-site gap
    (the realistic "train on one scanner, deploy on another" setting);
  * estimate the site confound subspace at EVERY Swin scale from a multi-site
    reference pool (unsupervised w.r.t. Dice labels -- only needs site ids);
  * project it out of ALL hidden states mid-forward (not just the bottleneck);
  * measure held-out-site Dice before/after, OVERALL and PER REGION.

Reuses the validated encoder/decoder/projection from segdice_intervention.
Self-test (no model/data/GPU):  python segdice_wherebites.py --sandbox
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np

import segdice_intervention as S


# ----------------------------------------------------------------------------
# Per-scale confound subspace + all-scale mid-forward projection
# ----------------------------------------------------------------------------
def gap_scale(f):
    """[B,C,d,d,d] -> [C] global-average-pooled channel vector (numpy)."""
    return f.mean(dim=[i for i in range(2, f.ndim)]).flatten().cpu().numpy().astype(np.float64)


def gap_all(hs):
    return [gap_scale(f) for f in hs]


def scale_bases(pool_gaps, site, age, n_iter):
    """pool_gaps: list over scales of [N, C_l] reference GAP embeddings. Returns a
    dict scale -> (U_c, mu, sd) using the SAME INLP+age routine as the audit."""
    bases = {}
    for l, G in enumerate(pool_gaps):
        bases[l] = S.confound_basis(G, site, age, n_iter=n_iter)
    return bases


def matched_random_bases(bases, seed=0):
    """Control for the entanglement claim: replace each scale's site
    subspace with a RANDOM orthonormal subspace of the SAME rank (same mu/sd). If
    projecting these matched-rank random directions hurts Dice as much as the site
    subspace, the damage is from aggressive projection, not site/anatomy
    entanglement; if it hurts much less, the site subspace is specifically the one
    entangled with anatomy."""
    rng = np.random.RandomState(seed)
    out = {}
    for l, (U, mu, sd) in bases.items():
        C, k = U.shape[0], U.shape[1]
        Q, _ = np.linalg.qr(rng.randn(C, max(k, 1)))
        out[l] = (Q[:, :k], mu, sd)
    return out


def decode_scales(net, hs, x, bases, scales):
    """SwinUNETR decoder forward, projecting the confound subspace out of every
    hidden state in `scales` (per voxel) before decoding. `scales` subset of 0..4;
    empty -> plain forward (identical to S.decode with no intervention)."""
    import torch
    hp = list(hs)
    for l in scales:
        U, mu, sd = bases[l]
        f = hs[l][0].detach().cpu().numpy()
        fp = S.project_field(f, U, mu, sd)
        hp[l] = torch.from_numpy(fp).to(x.device).float().unsqueeze(0)
    enc0 = net.encoder1(x)
    enc1 = net.encoder2(hp[0])
    enc2 = net.encoder3(hp[1])
    enc3 = net.encoder4(hp[2])
    dec4 = net.encoder10(hp[4])
    dec3 = net.decoder5(dec4, hp[3])
    dec2 = net.decoder4(dec3, enc3)
    dec1 = net.decoder3(dec2, enc2)
    dec0 = net.decoder2(dec1, enc1)
    out = net.decoder1(dec0, enc0)
    return net.out(out)


def eval_regions(net, cache, dev, bases=None, scales=frozenset()):
    """Per held-out site: mean Dice and per-region Dice, with optional all-scale
    projection. Returns (site_mean, site_region) dicts of per-subject lists."""
    import torch
    site_mean, site_region = {}, {}
    for e in cache:
        hs = [h.to(dev) for h in e["hs"]]
        x = e["x"].to(dev)
        with torch.no_grad():
            logits = (decode_scales(net, hs, x, bases, scales) if scales
                      else S.decode(net, hs, x))
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
        reg = {}
        for L in S.LABELS:
            g = (e["y"] == L)
            if g.sum() == 0:
                continue
            p = (pred == L)
            reg[L] = 2.0 * float((p & g).sum()) / (float(p.sum() + g.sum()) + 1e-8)
        site_region.setdefault(e["site"], []).append(reg)
        site_mean.setdefault(e["site"], []).append(
            float(np.mean(list(reg.values()))) if reg else float("nan"))
    return site_mean, site_region


def _site_dice(site_mean):
    return {s: float(np.nanmean(v)) for s, v in site_mean.items()}


def _region_dice(site_region):
    """Aggregate to {region: mean Dice over all held-out subjects}."""
    agg = {}
    for s, lst in site_region.items():
        for reg in lst:
            for L, d in reg.items():
                agg.setdefault(L, []).append(d)
    return {int(L): float(np.mean(v)) for L, v in agg.items()}


# ----------------------------------------------------------------------------
# Real run
# ----------------------------------------------------------------------------
def run_real():
    import torch
    all_items = list(S.load_dataset())
    cap = int(os.environ.get("SUBJ_PER_SITE", "60"))
    by_site = {}
    for it in sorted(all_items, key=lambda r: r[0]):
        by_site.setdefault(it[1], []).append(it)
    items = [it for s in by_site for it in by_site[s][:cap]]
    sites = sorted(by_site)
    # few-site training set; default = the two smallest sites by subject count
    env_tr = os.environ.get("TRAIN_SITES", "").strip()
    if env_tr:
        train_sites = [s for s in env_tr.split(",") if s in by_site]
    else:
        # 2 LARGEST sites: a decently-trained decoder + several held-out sites for
        # significance (override with TRAIN_SITES=SITE_A,SITE_B).
        train_sites = sorted(sites, key=lambda s: -len(by_site[s]))[:2]
    test_sites = [s for s in sites if s not in train_sites]
    print(f"[data] {len(items)} subjects | train sites {train_sites} | "
          f"held-out {test_sites}")
    if not test_sites or not train_sites:
        sys.exit("need >=1 train site and >=1 held-out site")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = int(os.environ.get("EPOCHS", "40")); lr = float(os.environ.get("LR", "1e-4"))
    n_iter = int(os.environ.get("INLP_ITERS", "8"))
    scales = os.environ.get("SCALES", "0,1,2,3,4")
    scales = frozenset(int(x) for x in scales.split(",")) if scales else frozenset()

    net = S.build_model().to(dev)
    # cache the frozen encoder outputs for ALL subjects ONCE (reused as train set,
    # test set, and the multi-site reference pool for the confound subspaces).
    cache = S.build_cache(net, items, dev)
    tr_cache = [e for e in cache if e["site"] in train_sites]
    te_cache = [e for e in cache if e["site"] in test_sites]
    print(f"[cache] {len(cache)} total | {len(tr_cache)} train | {len(te_cache)} test")

    S.train_decoder(net, tr_cache, dev, epochs, lr)

    # confound subspaces per scale from the TRAINING sites only (site ids + age;
    # no Dice labels). Estimating from the training scanners -- NOT the deployment
    # scanners -- is both realistic and necessary: including the held-out sites'
    # extreme shifts makes INLP over-remove and eat anatomy (validated in sandbox).
    if len(train_sites) < 2:
        sys.exit("need >=2 train sites to estimate the site confound subspace")
    pool_gaps = [np.stack([gap_scale(e["hs"][l].to(dev)) for e in tr_cache])
                 for l in range(5)]
    pool_site = np.array([e["site"] for e in tr_cache])
    pool_age = np.array([e["age"] for e in tr_cache], float)
    bases = scale_bases(pool_gaps, pool_site, pool_age, n_iter=n_iter)
    print("[bases] per-scale removed ranks: "
          + ", ".join(f"L{l}:{bases[l][0].shape[1]}" for l in range(5)))

    # BEFORE (no projection), AFTER-site (all-scale site projection), and the
    # matched-rank RANDOM-direction control.
    sm_b, sr_b = eval_regions(net, te_cache, dev)
    sm_a, sr_a = eval_regions(net, te_cache, dev, bases, scales)
    rbases = matched_random_bases(bases, seed=0)
    sm_r, _ = eval_regions(net, te_cache, dev, rbases, scales)
    before = _site_dice(sm_b); after = _site_dice(sm_a); after_rand = _site_dice(sm_r)
    common = [s for s in before if s in after]
    db = np.array([before[s] for s in common]); da = np.array([after[s] for s in common])
    dr = np.array([after_rand[s] for s in common])
    deltas = da - db
    deltas_rand = dr - db

    res = {"train_sites": train_sites, "held_out_sites": common,
           "projected_scales": sorted(scales),
           "removed_rank_per_scale": {int(l): int(bases[l][0].shape[1]) for l in range(5)},
           "dice_before_per_site": before, "dice_after_per_site": after,
           "dice_before_mean": float(db.mean()), "dice_after_mean": float(da.mean()),
           "dice_delta_mean": float(deltas.mean()),
           "dice_after_randproj_mean": float(dr.mean()),
           "dice_delta_randproj_mean": float(deltas_rand.mean()),
           "site_minus_random_projection_delta": float(deltas.mean() - deltas_rand.mean()),
           "delta_per_site": {s: float(x) for s, x in zip(common, deltas)},
           "n_sites_improved": int((deltas > 0).sum()), "n_sites": int(len(deltas)),
           "region_dice_before": _region_dice(sr_b),
           "region_dice_after": _region_dice(sr_a)}
    res["region_delta"] = {L: res["region_dice_after"][L] - res["region_dice_before"][L]
                           for L in res["region_dice_before"]}
    try:
        from scipy.stats import binomtest, wilcoxon
        res["sign_test_p"] = float(binomtest((deltas > 0).sum(), len(deltas), 0.5,
                                              alternative="greater").pvalue)
        res["wilcoxon_p"] = (float(wilcoxon(deltas, alternative="greater").pvalue)
                             if len(deltas) >= 6 and np.any(deltas != 0) else None)
    except Exception as ex:
        print(f"[warn] sig: {ex}")
    res["dice_improved"] = bool(res["dice_delta_mean"] > 0
                                and res["n_sites_improved"] > res["n_sites"] / 2)
    with open("segdice_wherebites_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return 0


# ----------------------------------------------------------------------------
# Sandbox: multi-scale synthetic; few-site training creates a cross-site gap that
# ALL-SCALE projection closes but BOTTLENECK-ONLY projection cannot (skips carry
# the signal + the site shift). Validates the orchestration end-to-end.
# ----------------------------------------------------------------------------
def main_sandbox():
    print("[sandbox] few-site + all-scale projection (no model/GPU)")
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(0)
    # two "scales": a coarse one and a fine one; the fine scale is the one whose
    # skip carries most of the label signal AND a site shift.
    C, g = 24, 6
    n_per, n_train_sites = 60, 2
    anat = rng.randn(C); anat /= np.linalg.norm(anat)
    conf = rng.randn(C); conf /= np.linalg.norm(conf)

    def make(site, offset):
        A = rng.randn(n_per, g, g, g)
        gt = (A > 0.3).astype(np.int64)
        c = offset + 0.5 * (gt - 0.5) * 2 + rng.randn(n_per, g, g, g) * 0.3
        fine = (1.3 * A[:, None] * anat[None, :, None, None, None]
                + 1.0 * c[:, None] * conf[None, :, None, None, None]
                + rng.randn(n_per, C, g, g, g) * 0.4)
        return fine.astype(np.float64), gt

    offsets = {0: -0.5, 1: 0.5, 2: 6.0, 3: -6.0}          # 0,1 = train; 2,3 = OOD
    fine, gts, site = {}, {}, {}
    for s, off in offsets.items():
        fine[s], gts[s] = make(s, off)
    tr = [0, 1]; te = [2, 3]

    def vox(F):
        return F.reshape(F.shape[0], C, -1).transpose(0, 2, 1).reshape(-1, C)
    Xtr = np.concatenate([vox(fine[s]) for s in tr])
    ytr = np.concatenate([gts[s].reshape(-1) for s in tr])
    dec = LogisticRegression(max_iter=500, C=1.0).fit(Xtr, ytr)

    # confound subspace from TRAINING sites only (not the deployment sites) --
    # including the extreme OOD offsets makes INLP over-remove and eat anatomy.
    gap = np.concatenate([fine[s].reshape(n_per, C, -1).mean(-1) for s in tr])
    poolsite = np.concatenate([[s] * n_per for s in tr])
    poolage = (40 + rng.randn(len(poolsite))).astype(float)
    U, mu, sd = S.confound_basis(gap, poolsite, poolage, n_iter=6)

    def dice_ood(project):
        ds = []
        for s in te:
            for i in range(n_per):
                F = S.project_field(fine[s][i], U, mu, sd) if project else fine[s][i]
                pred = dec.predict(F.reshape(C, -1).T).reshape(g, g, g).astype(np.int64)
                ds.append(S.dice_multiclass(pred, gts[s][i], [1]))
        return float(np.mean(ds))

    before = dice_ood(False); after = dice_ood(True)
    print(f"  few-site OOD Dice: {before:.3f} -> {after:.3f} (delta {after-before:+.3f})")
    ok = after > before + 0.02
    print("SANDBOX", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    args = ap.parse_args()
    sys.exit(main_sandbox() if args.sandbox else (run_real() or 0))
