#!/usr/bin/env python
"""extract_multilayer.py -- per-LAYER frozen-encoder GAP embeddings for ABIDE.

For the "scanner > brain" audit: extract the global-average-pooled embedding at
EVERY Swin scale (not just the bottleneck) from a frozen encoder, so we can map
how decodable acquisition site vs clinical variables is across depth AND across
models (brain-pretrained / CT-pretrained / random-init). Reuses the validated
loaders in build_cohort.py; the only change is dumping all 5 hidden states.

Env:
  ABIDE_ROOT       ABIDE root (T1w + Phenotypic_V1_0b.csv)   [required]
  FROZEN_CKPT      encoder checkpoint; UNSET -> random-init encoder (a baseline)
  MULTILAYER_NPZ   output path                                [required]
"""
import os
import re
import sys
import numpy as np

import build_cohort as B


def _site_from_path(t1, root):
    """Site = first path component under ABIDE_ROOT (e.g. ABIDEII-NYU_1). Robust
    for cohorts whose site is in the folder tree rather than only the phenotype."""
    rel = os.path.relpath(t1, root)
    return rel.split(os.sep)[0]


def embed_all(net, dev, vol):
    import torch
    with torch.no_grad():
        x = vol.unsqueeze(0).to(dev).float()
        hs = net.swinViT(x)                       # 5 hierarchical feature maps
        return [h.mean(dim=[i for i in range(2, h.ndim)]).flatten()
                .cpu().numpy().astype(np.float64) for h in hs]


def main():
    out = os.environ.get("MULTILAYER_NPZ")
    root = os.environ.get("ABIDE_ROOT")
    if not out or not root:
        sys.exit("set ABIDE_ROOT and MULTILAYER_NPZ")
    # phenotype is OPTIONAL: site always comes from the folder path; age/sex are
    # joined from the phenotype when available (else NaN, and clinical probes for
    # that variable are simply skipped downstream).
    try:
        pheno = B.load_phenotype()
    except Exception as ex:
        print(f"[warn] no phenotype ({str(ex)[:60]}); site-only (age/sex = NaN)")
        pheno = {}
    tfm = B.make_transform()
    net, dev = B.load_frozen_encoder()            # honors FROZEN_CKPT (unset=random)

    subs = sorted(B.iter_t1(root), key=lambda x: x[1])   # deterministic order
    cap = int(os.environ.get("SUBJ_PER_SITE", "0"))      # 0 = all; else cap per site
    print(f"[data] {len(subs)} T1w volumes; {len(pheno)} phenotype rows; "
          f"cap={cap or 'none'}/site")
    layers = None
    sites, ages, sexes, ids = [], [], [], []
    persite = {}
    for k, (sid, t1) in enumerate(subs):
        site = _site_from_path(t1, root)
        if cap and persite.get(site, 0) >= cap:
            continue
        age, sex = (np.nan, np.nan)
        if sid in pheno:
            _, a, s = pheno[sid]
            age = a if np.isfinite(a) else np.nan
            sex = (s - 1) if s in (1, 2) else np.nan
        try:
            embs = embed_all(net, dev, tfm(t1))
        except Exception as ex:
            print(f"  [skip] sub-{sid}: {str(ex)[:70]}"); continue
        if not all(np.isfinite(e).all() for e in embs):
            continue
        if layers is None:
            layers = [[] for _ in embs]
        for l, e in enumerate(embs):
            layers[l].append(e)
        sites.append(site); ages.append(age); sexes.append(sex)
        ids.append(f"{site}/sub-{sid}")
        persite[site] = persite.get(site, 0) + 1
        if (k + 1) % 25 == 0:
            print(f"  ...{k+1}/{len(subs)} ({len(sites)} kept)")

    if not sites or len(sites) < 40:
        sys.exit(f"only {len(sites)} usable subjects; refusing degenerate output")
    data = {f"emb_L{l}": np.stack(layers[l]) for l in range(len(layers))}
    data.update(site=np.array(sites), age=np.array(ages, float),
                sex=np.array(sexes, float), subject_id=np.array(ids, dtype=object))
    np.savez(out, **data)
    dims = ",".join(str(np.stack(layers[l]).shape[1]) for l in range(len(layers)))
    print(f"[done] wrote {out}: {len(sites)} subjects; per-layer dims [{dims}]")


if __name__ == "__main__":
    main()
