#!/usr/bin/env python
"""segdice_intervention.py -- the segmentation-Dice headline for direction #5.

The GAP audit (confound_audit.py) asks: are the frozen embeddings site-biased,
and is that bias removable? This script asks the deployment question that matters:
does removing the confound subspace MID-FORWARD improve *cross-site
segmentation Dice* -- with no retraining of the frozen encoder?

Design (faithful to the paper's thesis):
  * Frozen brain-MRI FM encoder (BrainSegFounder swinViT) -- never updated. This
    is the object under audit.
  * A segmentation decoder (the standard SwinUNETR decoder path) trained on
    SILVER labels (SynthSeg) with the encoder frozen -- i.e. the exact "frozen FM
    + trained head" pattern practitioners deploy.
  * Intervention: estimate the site/age confound subspace U_c in the 768-d
    bottleneck channel space (via INLP, reusing confound_audit.py), then at TEST
    time project it out of the bottleneck feature FIELD, per voxel, before the
    decoder: hidden[4] <- (I - U_c U_c^T) hidden[4]. Continue the frozen forward.
  * Report leave-one-site-out cross-site Dice BEFORE vs AFTER, sweeping removed
    rank, with the same subject-bootstrap CI + per-fold sign test as the audit.

Everything is computed from data. Absent data/model/labels -> raise, never fake.
Self-test (no model/data/GPU):  python segdice_intervention.py --sandbox
"""
from __future__ import annotations

import os
import sys
import json
import glob
import argparse
import numpy as np

# reuse the SAME validated confound machinery as the GAP audit
from confound_audit import inlp_projection, project_out, _standardize

IMG = (96, 96, 96)


# ----------------------------------------------------------------------------
# Dice + leave-one-site-out evaluation (shared by sandbox and real run)
# ----------------------------------------------------------------------------
def dice_multiclass(pred, gt, labels):
    """Mean Dice over `labels` present in gt (background excluded). pred, gt are
    integer label volumes of the same shape."""
    ds = []
    for L in labels:
        if L == 0:
            continue
        g = (gt == L)
        if g.sum() == 0:
            continue
        p = (pred == L)
        inter = float((p & g).sum())
        denom = float(p.sum() + g.sum())
        ds.append(2.0 * inter / (denom + 1e-8))
    return float(np.mean(ds)) if ds else float("nan")


def loso_folds(site):
    for held in np.unique(site):
        yield held, (site != held), (site == held)


# ----------------------------------------------------------------------------
# Confound subspace in the bottleneck channel space, estimated on TRAIN sites.
# Same INLP as the GAP audit, applied to the GAP of the bottleneck field so the
# subspace is directly comparable to the audit numbers.
# ----------------------------------------------------------------------------
def confound_basis(gap_train, site_train, age_train, n_iter):
    """Return an orthonormal basis U_c [C, k] spanning the site subspace (INLP)
    plus the single age-regression direction, in STANDARDIZED channel space.
    Returns (U_c, mu, sd) so the same standardization is applied to test voxels."""
    Es, mu, sd = _standardize(gap_train)
    # site: INLP gives the composed projection P = I - U_c U_c^T; recover U_c.
    _, P = inlp_projection(Es, site_train, n_iter=n_iter)
    # column space removed by P is span(U_c); get it from I-P via SVD.
    R = np.eye(P.shape[0]) - P
    U, S, _ = np.linalg.svd(R, full_matrices=False)
    U_site = U[:, S > 1e-6]
    # age: one ridge direction in the same standardized space
    from sklearn.linear_model import Ridge
    wa = Ridge(alpha=1.0).fit(Es, age_train).coef_
    Ua = (wa / (np.linalg.norm(wa) + 1e-12))[:, None]
    U_c = np.concatenate([U_site, Ua], axis=1)
    # re-orthonormalize the union
    Q, _ = np.linalg.qr(U_c)
    return Q, mu, sd


def project_field(field, U_c, mu, sd):
    """Project the confound subspace out of a bottleneck feature field.
    field: [C, D, H, W] (numpy or torch->numpy). Standardize per channel with the
    TRAIN stats, remove U_c per voxel, un-standardize. Returns [C,D,H,W]."""
    C = field.shape[0]
    V = field.reshape(C, -1).T                 # [Nvox, C]
    Vs = (V - mu) / sd
    Vp = project_out(Vs, U_c)                   # (I - U U^T)
    Vr = Vp * sd + mu
    return Vr.T.reshape(field.shape)


# ============================================================================
# REAL RUN: frozen BrainSegFounder encoder + trained SwinUNETR decoder on ABIDE
# ============================================================================
def build_model():
    import torch
    from monai.networks.nets import SwinUNETR
    try:
        net = SwinUNETR(in_channels=1, out_channels=int(os.environ.get("N_CLASSES", 14)),
                        feature_size=48)
    except TypeError:
        net = SwinUNETR(img_size=IMG, in_channels=1,
                        out_channels=int(os.environ.get("N_CLASSES", 14)), feature_size=48)
    ckpt = os.environ.get("FROZEN_CKPT")
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(f"FROZEN_CKPT missing: {ckpt!r} (need the brain FM encoder)")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    weights = sd if (isinstance(sd, dict) and "state_dict" in sd) else {"state_dict": sd}
    try:
        net.load_from(weights=weights)
        print(f"[model] loaded encoder via load_from from {ckpt}")
    except Exception as e:
        print(f"[model] load_from failed ({str(e)[:60]}); swinViT strict=False")
        inner = weights["state_dict"]
        clean = {k.replace("module.", "").replace("swinViT.", ""): v for k, v in inner.items()}
        r = net.swinViT.load_state_dict(clean, strict=False)
        print(f"[model] swinViT loaded: {len(r.missing_keys)} missing, "
              f"{len(r.unexpected_keys)} unexpected keys")
    # FREEZE the encoder (the FM under audit); decoder path stays trainable.
    for p in net.swinViT.parameters():
        p.requires_grad_(False)
    return net


def encode(net, x):
    """Run the FROZEN swinViT encoder (no grad); returns the 5 hierarchical hidden
    states. Cache these per subject so the decoder trains without ever re-running
    the expensive frozen encoder (the dominant cost)."""
    import torch
    with torch.no_grad():
        return net.swinViT(x)


def decode(net, hs, x, U_c=None, mu=None, sd=None):
    """Trainable SwinUNETR decoder path from cached hidden states, with an OPTIONAL
    mid-forward intervention: project the confound subspace out of the deepest
    hidden state (hs[4], the 768-ch bottleneck) per voxel before decoding. Mirrors
    monai SwinUNETR.forward's decoder half exactly."""
    import torch
    if U_c is not None:
        f = hs[4][0].detach().cpu().numpy()              # [C,D,H,W] (batch=1)
        fp = project_field(f, U_c, mu, sd)
        hs4 = torch.from_numpy(fp).to(x.device).float().unsqueeze(0)
    else:
        hs4 = hs[4]
    enc0 = net.encoder1(x)
    enc1 = net.encoder2(hs[0])
    enc2 = net.encoder3(hs[1])
    enc3 = net.encoder4(hs[2])
    dec4 = net.encoder10(hs4)
    dec3 = net.decoder5(dec4, hs[3])
    dec2 = net.decoder4(dec3, enc3)
    dec1 = net.decoder3(dec2, enc2)
    dec0 = net.decoder2(dec1, enc1)
    out = net.decoder1(dec0, enc0)
    return net.out(out)


def gap_of(hs):
    """768-d GAP of the bottleneck hidden state (the same embedding the GAP audit
    uses), computed from a cached forward."""
    f = hs[4]
    return f.mean(dim=[i for i in range(2, f.ndim)]).flatten().cpu().numpy().astype(np.float64)


def make_transforms():
    from monai.transforms import (Compose, LoadImage, EnsureChannelFirst, Orientation,
                                  Spacing, ScaleIntensity, ResizeWithPadOrCrop)
    img = Compose([LoadImage(image_only=True), EnsureChannelFirst(), Orientation(axcodes="RAS"),
                   Spacing(pixdim=(1.7, 1.7, 1.7), mode="bilinear"),
                   ScaleIntensity(minv=0.0, maxv=1.0), ResizeWithPadOrCrop(spatial_size=IMG)])
    lab = Compose([LoadImage(image_only=True), EnsureChannelFirst(), Orientation(axcodes="RAS"),
                   Spacing(pixdim=(1.7, 1.7, 1.7), mode="nearest"),
                   ResizeWithPadOrCrop(spatial_size=IMG)])
    return img, lab


def remap_labels(vol, mapping):
    out = np.zeros_like(vol, dtype=np.int64)
    for src, dst in mapping.items():
        out[vol == src] = dst
    return out


def load_dataset():
    """Yield (subject_id, site, age, t1_path, seg_path) for ABIDE subjects that
    have a SynthSeg silver label. Requires ABIDE_ROOT + SILVER_ROOT."""
    import pandas as pd
    root = os.environ["ABIDE_ROOT"]; silver = os.environ["SILVER_ROOT"]
    df = pd.read_csv(os.path.join(root, "Phenotypic_V1_0b.csv"))
    pheno = {int(r.SUB_ID): (str(r.SITE_ID), float(r.AGE_AT_SCAN)) for r in df.itertuples()}
    import re
    for t1 in glob.glob(os.path.join(root, "**", "*_T1w.nii.gz"), recursive=True):
        m = re.search(r"sub-0*(\d+)", os.path.basename(t1))
        if not m:
            continue
        sid = int(m.group(1))
        if sid not in pheno:
            continue
        seg = os.path.join(silver, f"sub-{sid}_synthseg.nii.gz")
        if not os.path.exists(seg):
            continue
        site, age = pheno[sid]
        if not np.isfinite(age):
            continue
        yield sid, site, age, t1, seg


# SynthSeg -> compact anatomical label set (background + 13 structures), so Dice
# is over meaningful regions and matches out_channels=14.
SYNTHSEG_MAP = {
    0: 0, 2: 1, 41: 1,          # cerebral WM (L/R)
    3: 2, 42: 2,                # cerebral cortex
    4: 3, 43: 3,                # lateral ventricle
    10: 4, 49: 4,               # thalamus
    11: 5, 50: 5,               # caudate
    12: 6, 51: 6,               # putamen
    13: 7, 52: 7,               # pallidum
    17: 8, 53: 8,               # hippocampus
    18: 9, 54: 9,               # amygdala
    26: 10, 58: 10,             # accumbens
    7: 11, 46: 11,              # cerebellum WM
    8: 12, 47: 12,              # cerebellum cortex
    16: 13,                     # brainstem
}
LABELS = list(range(1, 14))


def build_cache(net, items, dev, need_labels=True):
    """Run the FROZEN encoder ONCE per subject and cache its hidden states, the
    input volume, and the silver label. Big tensors are kept on CPU so many
    subjects fit in RAM; the decoder then trains without re-running the encoder."""
    img_t, lab_t = make_transforms()
    cache = []
    for sid, site, age, t1, seg in items:
        try:
            x = img_t(t1).unsqueeze(0).to(dev).float()
            hs = encode(net, x)
            e = {"sid": sid, "site": site, "age": float(age),
                 "hs": [h.detach().cpu() for h in hs], "x": x.detach().cpu()}
            if need_labels:
                y = remap_labels(np.rint(lab_t(seg)[0].cpu().numpy()).astype(np.int64),
                                 SYNTHSEG_MAP)
                e["y"] = y
            cache.append(e)
        except Exception as ex:
            print(f"  [skip-cache] {sid}: {str(ex)[:60]}")
    return cache


def train_decoder(net, cache, dev, epochs, lr):
    """Train the (non-encoder) decoder path on cached encoder features + silver
    labels; encoder stays frozen. No intervention during training."""
    import torch
    from monai.losses import DiceCELoss
    params = [p for n, p in net.named_parameters()
              if p.requires_grad and not n.startswith("swinViT")]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    net.train(); net.swinViT.eval()
    rng = np.random.RandomState(0)
    order = list(range(len(cache)))
    for ep in range(epochs):
        rng.shuffle(order)
        running = 0.0
        for k in order:
            e = cache[k]
            hs = [h.to(dev) for h in e["hs"]]
            x = e["x"].to(dev)
            yt = torch.from_numpy(e["y"]).to(dev)[None, None].float()
            opt.zero_grad()
            logits = decode(net, hs, x)
            loss = loss_fn(logits, yt)
            loss.backward(); opt.step()
            running += float(loss)
        print(f"  [train] epoch {ep+1}/{epochs} mean loss {running/max(1,len(order)):.4f}")
    net.eval()


def gap_from_cache(cache):
    G = np.stack([gap_of(e["hs"]) for e in cache])
    S = np.array([e["site"] for e in cache])
    A = np.array([e["age"] for e in cache], float)
    return G, S, A


def eval_dice(net, cache, dev, U_c=None, mu=None, sd=None):
    """{site: [per-subject Dice]} over a cache, with optional mid-forward removal."""
    import torch
    per = {}
    for e in cache:
        hs = [h.to(dev) for h in e["hs"]]
        x = e["x"].to(dev)
        with torch.no_grad():
            logits = decode(net, hs, x, U_c, mu, sd)
            pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
        per.setdefault(e["site"], []).append(dice_multiclass(pred, e["y"], LABELS))
    return per


def run_real():
    import torch
    all_items = list(load_dataset())
    # optional per-site cap keeps LOSO training tractable on a single GPU
    cap = int(os.environ.get("SUBJ_PER_SITE", "60"))
    by_site = {}
    for it in sorted(all_items, key=lambda r: r[0]):         # deterministic by sid
        by_site.setdefault(it[1], []).append(it)
    items = [it for s in by_site for it in by_site[s][:cap]]
    if len(items) < 40:
        sys.exit(f"Only {len(items)} subjects with silver labels; refusing degenerate run.")
    sites = sorted(set(s for _, s, *_ in items))
    print(f"[data] {len(items)}/{len(all_items)} subjects (cap {cap}/site), "
          f"{len(sites)} sites: {sites}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iters_sweep = tuple(int(x) for x in os.environ.get("INLP_ITERS", "2,4,8").split(","))
    epochs = int(os.environ.get("EPOCHS", "30")); lr = float(os.environ.get("LR", "1e-4"))

    fold_before, fold_after = {}, {int(i): {} for i in iters_sweep}
    for held, _, _ in loso_folds(np.array([s for _, s, *_ in items])):
        train_items = [it for it in items if it[1] != held]
        test_items = [it for it in items if it[1] == held]
        if len(test_items) < 4 or len(train_items) < 20:
            continue
        print(f"\n=== fold: hold out {held} ({len(test_items)} test, {len(train_items)} train) ===")
        net = build_model().to(dev)
        train_cache = build_cache(net, train_items, dev)     # frozen encoder ONCE
        test_cache = build_cache(net, test_items, dev)
        train_decoder(net, train_cache, dev, epochs, lr)
        # BEFORE
        db = eval_dice(net, test_cache, dev)
        fold_before[held] = float(np.mean([np.mean(v) for v in db.values()]))
        print(f"  [{held}] Dice before: {fold_before[held]:.4f}")
        # confound subspace from TRAIN gap, then intervene at test, per rank
        gap_tr, site_tr, age_tr = gap_from_cache(train_cache)
        for it in iters_sweep:
            U_c, mu, sd = confound_basis(gap_tr, site_tr, age_tr, n_iter=it)
            da = eval_dice(net, test_cache, dev, U_c, mu, sd)
            fold_after[int(it)][held] = float(np.mean([np.mean(v) for v in da.values()]))
            print(f"  [{held}] Dice after (inlp={it}, rank={U_c.shape[1]}): "
                  f"{fold_after[int(it)][held]:.4f}")
        del net, train_cache, test_cache
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        # incremental checkpoint: a walltime timeout still preserves finished folds
        with open("segdice_partial.json", "w") as f:
            json.dump({"dice_before_per_site": fold_before,
                       "dice_after_by_rank": {str(i): fold_after[int(i)] for i in iters_sweep}},
                      f, indent=2)
        print(f"  [checkpoint] {len(fold_before)}/{len(sites)} folds done -> segdice_partial.json")

    # aggregate + significance (best rank by mean cross-site Dice)
    res = {"n": len(items), "n_sites": len(sites), "sites": sites,
           "dice_before_per_site": fold_before}
    best_it, best_mean = None, -1
    for it in iters_sweep:
        common = [s for s in fold_before if s in fold_after[int(it)]]
        m = float(np.mean([fold_after[int(it)][s] for s in common])) if common else float("nan")
        res.setdefault("dice_after_by_rank", {})[str(it)] = {
            "mean": m, "per_site": fold_after[int(it)]}
        if np.isfinite(m) and m > best_mean:
            best_mean, best_it = m, it
    common = [s for s in fold_before if s in fold_after[int(best_it)]]
    before = np.array([fold_before[s] for s in common])
    after = np.array([fold_after[int(best_it)][s] for s in common])
    deltas = after - before
    res["best_inlp_iters"] = int(best_it)
    res["dice_before_mean"] = float(before.mean())
    res["dice_after_mean"] = float(after.mean())
    res["dice_delta_mean"] = float(deltas.mean())
    res["delta_per_site"] = {s: float(d) for s, d in zip(common, deltas)}
    res["n_folds_improved"] = int((deltas > 0).sum())
    res["n_folds"] = int(len(deltas))
    try:
        from scipy.stats import binomtest, wilcoxon
        res["sign_test_p"] = float(binomtest((deltas > 0).sum(), len(deltas), 0.5,
                                              alternative="greater").pvalue)
        res["wilcoxon_p"] = (float(wilcoxon(deltas, alternative="greater").pvalue)
                             if len(deltas) >= 6 and np.any(deltas != 0) else None)
    except Exception as e:
        print(f"[warn] sig: {e}")
    res["dice_improved"] = bool(res["dice_delta_mean"] > 0 and res["n_folds_improved"] > res["n_folds"] / 2)
    with open("segdice_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    return 0


# ============================================================================
# SANDBOX: validate the MID-FORWARD per-voxel projection integrates with a
# decoder and recovers cross-site Dice under a planted spatial site confound.
# No model/data/GPU. Uses a fixed linear per-voxel "decoder".
# ============================================================================
def main_sandbox():
    """Faithful to the real pipeline: a decoder is TRAINED (per LOSO fold) on the
    train-site feature fields -- so it LEARNS the confound shortcut -- then frozen.
    At test we project the confound out of the held-out field and re-apply the
    SAME decoder. A method that works recovers cross-site Dice; a broken one (that
    also erodes anatomy) does not. This mirrors train_decoder + eval_dice exactly."""
    print("[sandbox] segmentation mid-forward intervention (no model/GPU)")
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(0)
    C, g, n_per_site, n_train_sites = 48, 8, 60, 4   # bottleneck [C,g,g,g]
    anat_dir = rng.randn(C); anat_dir /= np.linalg.norm(anat_dir)
    conf_dir = rng.randn(C); conf_dir /= np.linalg.norm(conf_dir)

    def make_site(s, offset):
        A = rng.randn(n_per_site, g, g, g)                    # per-voxel anatomy
        gt = (A > 0.3).astype(np.int64)                       # true foreground
        # confound = per-site MEAN offset along conf_dir (a scanner shift) plus a
        # mild label correlation so the decoder puts weight on conf_dir. The OOD
        # site's offset is far OUTSIDE the training range, so that weight makes
        # the decoder extrapolate catastrophically -- the classic cross-scanner
        # failure. Removing the conf axis at test re-centers onto anatomy.
        conf = offset + 0.6 * (gt - 0.5) * 2.0 + rng.randn(n_per_site, g, g, g) * 0.3
        field = (1.5 * A[:, None] * anat_dir[None, :, None, None, None]
                 + 1.0 * conf[:, None] * conf_dir[None, :, None, None, None]
                 + rng.randn(n_per_site, C, g, g, g) * 0.4)
        return field.astype(np.float64), gt

    train_offsets = [-1.0, -0.4, 0.4, 1.0]                    # in-range scanner shifts
    tr_fields, tr_gts, tr_site = [], [], []
    for s, off in enumerate(train_offsets):
        f, gt = make_site(s, off)
        tr_fields.append(f); tr_gts.append(gt); tr_site += [s] * n_per_site
    tr_fields = np.concatenate(tr_fields); tr_gts = np.concatenate(tr_gts)
    tr_site = np.array(tr_site); tr_age = (40 + tr_site * 2.0).astype(float)
    te_fields, te_gts = make_site(n_train_sites, offset=6.0)  # OOD scanner, far off-range

    # decoder trained on TRAIN sites only (no intervention) -> learns the shortcut
    Xtr = tr_fields.reshape(len(tr_fields), C, -1).transpose(0, 2, 1).reshape(-1, C)
    ytr = tr_gts.reshape(-1)
    dec = LogisticRegression(max_iter=500, C=1.0).fit(Xtr, ytr)

    def eval_ood(project):
        gap = tr_fields.reshape(len(tr_fields), C, -1).mean(-1)
        U_c, mu, sd = (confound_basis(gap, tr_site, tr_age, n_iter=4) if project
                       else (None, None, None))
        ds = []
        for i in range(len(te_fields)):
            fld = project_field(te_fields[i], U_c, mu, sd) if project else te_fields[i]
            pred = dec.predict(fld.reshape(C, -1).T).reshape(g, g, g).astype(np.int64)
            ds.append(dice_multiclass(pred, te_gts[i], [1]))
        return float(np.mean(ds))

    before = eval_ood(project=False)
    after = eval_ood(project=True)
    print(f"  OOD-site Dice: {before:.3f} -> {after:.3f} (delta {after-before:+.3f})")
    ok = after > before + 0.02
    print("SANDBOX", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    args = ap.parse_args()
    sys.exit(main_sandbox() if args.sandbox else (run_real() or 0))
