#!/usr/bin/env python
"""build_cohort.py -- ABIDE + MONAI Swin-UNETR encoder. LOGIN/GPU node.

Builds cohort_cache.npz for confound_audit.py: for each ABIDE subject, extract a
frozen Swin-UNETR-encoder embedding from the T1w volume, with site + age as
confounds and SEX as the anatomy-driven downstream task (all from the ABIDE
phenotypic CSV -- no SynthSeg needed for v1). Fails loudly if data/model absent;
never fabricates.

Setup:
  export ABIDE_ROOT=~/data/abide                 # contains Phenotypic_V1_0b.csv + <SITE>/sub-*/anat/*_T1w.nii.gz
  export FROZEN_CKPT=~/models/model_swinvit.pt    # OPTIONAL MONAI SSL weights; omit -> init-weights encoder
  export COHORT_NPZ=$SCRATCH/cohort_cache.npz
  python build_cohort.py
"""
import os
import sys
import glob
import re
import numpy as np

_S = int(os.environ.get("IMG_SIZE", "96"))   # input cube side; set 128 for the
IMG = (_S, _S, _S)                            # higher-resolution ablation


def load_phenotype():
    import pandas as pd
    root = os.environ.get("ABIDE_ROOT")
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(f"ABIDE_ROOT missing: {root!r}")
    # PHENO_CSV lets the same loader serve ABIDE-I (Phenotypic_V1_0b.csv) and
    # ABIDE-II (ABIDEII_Composite_Phenotypic.csv) -- same column names.
    pheno_name = os.environ.get("PHENO_CSV", "Phenotypic_V1_0b.csv")
    csv = pheno_name if os.path.isabs(pheno_name) else os.path.join(root, pheno_name)
    if not os.path.exists(csv):
        raise FileNotFoundError(f"{pheno_name} not found (looked at {csv})")
    try:
        df = pd.read_csv(csv, encoding="latin-1")
    except Exception:
        df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()               # ABIDE-II has 'AGE_AT_SCAN '
    df = df[["SUB_ID", "SITE_ID", "AGE_AT_SCAN", "SEX"]].copy()
    out = {}
    for r in df.itertuples():
        try:
            out[int(r.SUB_ID)] = (str(r.SITE_ID), float(r.AGE_AT_SCAN), int(r.SEX))
        except Exception:
            pass
    return out


def iter_t1(root):
    """Yield (sub_id, t1_path) for every T1w under ABIDE_ROOT/<SITE>/sub-*/anat/."""
    for t1 in glob.glob(os.path.join(root, "**", "*_T1w.nii.gz"), recursive=True):
        m = re.search(r"sub-0*(\d+)", os.path.basename(t1))
        if m:
            yield int(m.group(1)), t1


def make_transform():
    from monai.transforms import (Compose, LoadImage, EnsureChannelFirst,
                                  Orientation, Spacing, ScaleIntensity,
                                  ResizeWithPadOrCrop)
    mm = float(os.environ.get("SPACING_MM", "1.7"))   # 1.0 + IMG_SIZE=160 = hi-res
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        Spacing(pixdim=(mm, mm, mm), mode="bilinear"),
        ScaleIntensity(minv=0.0, maxv=1.0),
        ResizeWithPadOrCrop(spatial_size=IMG),
    ])


def load_frozen_encoder():
    """MONAI Swin-UNETR; use its swinViT encoder as a frozen feature extractor.
    Loads SSL weights from FROZEN_CKPT if given, else init weights (still a valid
    frozen encoder for the method; swap in real FM weights for headline results)."""
    import torch
    from monai.networks.nets import SwinUNETR
    try:
        net = SwinUNETR(in_channels=1, out_channels=14, feature_size=48)
    except TypeError:
        net = SwinUNETR(img_size=IMG, in_channels=1, out_channels=14, feature_size=48)
    ckpt = os.environ.get("FROZEN_CKPT")
    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        weights = sd if (isinstance(sd, dict) and "state_dict" in sd) else {"state_dict": sd}
        try:
            net.load_from(weights=weights)
            print(f"[model] loaded SSL weights via load_from from {ckpt}")
        except Exception as e:
            print(f"[model] load_from failed ({str(e)[:80]}); trying swinViT strict=False")
            inner = weights["state_dict"]
            clean = {k.replace("module.", "").replace("swinViT.", ""): v
                     for k, v in inner.items()}
            r = net.swinViT.load_state_dict(clean, strict=False)
            print(f"[model] swinViT loaded: {len(r.missing_keys)} missing, "
                  f"{len(r.unexpected_keys)} unexpected keys")
    else:
        print("[model] FROZEN_CKPT not set -> using init-weights encoder "
              "(swap in real FM weights for headline results)")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.float().to(dev).eval()
    return net, dev


@np.errstate(all="ignore")
def embed(net, dev, vol):
    import torch
    with torch.no_grad():
        x = vol.unsqueeze(0).to(dev).float()           # [1,1,D,H,W]
        hidden = net.swinViT(x)                         # list of hierarchical feats
        feat = hidden[-1] if isinstance(hidden, (list, tuple)) else hidden
        emb = feat.mean(dim=[i for i in range(2, feat.ndim)])  # GAP over spatial
        return emb.flatten().cpu().numpy().astype(np.float64)


def main():
    out = os.environ.get("COHORT_NPZ", "cohort_cache.npz")
    root = os.environ.get("ABIDE_ROOT")
    pheno = load_phenotype()
    tfm = make_transform()
    net, dev = load_frozen_encoder()

    embs, sites, ages, ys, ids = [], [], [], [], []
    subs = list(iter_t1(root))
    print(f"[data] {len(subs)} T1w volumes found")
    for k, (sid, t1) in enumerate(subs):
        if sid not in pheno:
            continue
        site, age, sex = pheno[sid]
        if not np.isfinite(age) or sex not in (1, 2):
            continue
        try:
            vol = tfm(t1)
            e = embed(net, dev, vol)
        except Exception as ex:
            print(f"  [skip] sub-{sid}: {str(ex)[:80]}")
            continue
        if not np.isfinite(e).all():
            continue
        embs.append(e); sites.append(site); ages.append(age)
        ys.append(sex - 1)  # 0/1
        ids.append(f"{site}/sub-{sid}")
        if (k + 1) % 25 == 0:
            print(f"  ...{k+1}/{len(subs)} done ({len(embs)} kept)")

    if len(embs) < 40:
        sys.exit(f"Only {len(embs)} usable subjects; check ABIDE_ROOT/model. "
                 f"Refusing to write a degenerate cohort.")
    E = np.stack(embs)
    np.savez(out, embeddings=E, site=np.array(sites), age=np.array(ages, float),
             y_task=np.array(ys, float), subject_id=np.array(ids, dtype=object))
    print(f"[done] wrote {out}: {E.shape[0]} subjects x {E.shape[1]}-d, "
          f"{len(set(sites))} sites")


if __name__ == "__main__":
    main()
