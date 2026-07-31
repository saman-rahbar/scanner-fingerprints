#!/usr/bin/env python
"""multiseed_intrinsic.py -- is the intrinsic site fingerprint stable across the
random weight-init seed, or a fluke of one draw?

The "intrinsic" claim (a randomly initialized encoder is already a ~0.9 site
classifier) was measured with a single weight-init seed per architecture. A skeptic
can ask whether the reported intervals reflect seed variance or only holdout
resampling. They only reflect holdout resampling. This script closes that gap: for a
given architecture we build SEVERAL random-init encoders (different seeds), read the
deep-layer GAP embedding from each, and decode acquisition site. If deep-layer site
decodability is stable across seeds (small spread), the fingerprint is a property of
the architecture, not of one lucky initialization.

Key efficiency: each T1w volume is preprocessed ONCE and cached in memory, then all
(seed) encoders are applied to it -- so N seeds cost ~1x the data-loading time, not
Nx. Reuses build_cohort loaders and confound_audit probes.

Env:
  ABIDE_ROOT      ABIDE root (T1w tree)                         [required]
  ARCH            swin | vit | resnet                           [default swin]
  SEEDS           comma-separated seeds                         [default 0,1,2]
  SUBJ_PER_SITE   cap per site (0 = all)                        [default 60]
  MULTISEED_JSON  output path                          [default multiseed_<arch>.json]
Self-test: python multiseed_intrinsic.py --sandbox
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np

from confound_audit import _standardize, probe_decodability


def _deep_layers_swin(seed, img, dev, vols):
    """Build a random-init SwinUNETR (seeded), return per-layer GAP embeddings
    [n_subj, dim] for all 5 hierarchical stages."""
    import torch
    from monai.networks.nets import SwinUNETR
    torch.manual_seed(seed)
    try:
        net = SwinUNETR(in_channels=1, out_channels=14, feature_size=48)
    except TypeError:
        net = SwinUNETR(img_size=img, in_channels=1, out_channels=14, feature_size=48)
    net = net.float().to(dev).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    per_layer = None
    with torch.no_grad():
        for v in vols:
            x = v.unsqueeze(0).to(dev).float()
            hs = net.swinViT(x)
            embs = [h.mean(dim=[i for i in range(2, h.ndim)]).flatten()
                    .cpu().numpy().astype(np.float64) for h in hs]
            if per_layer is None:
                per_layer = [[] for _ in embs]
            for l, e in enumerate(embs):
                per_layer[l].append(e)
    return [np.stack(c) for c in per_layer]


def _deep_layers_other(arch, seed, img, dev, vols):
    """Random-init ViT / ResNet (seeded) per-layer GAP embeddings."""
    import torch
    import arch_encoders as A
    net = A.build_encoder(arch, img, seed=seed).to(dev).eval()
    per_layer = None
    for v in vols:
        x = v.unsqueeze(0).to(dev).float()
        embs = A.encode_layers(net, arch, x)
        if per_layer is None:
            per_layer = [[] for _ in embs]
        for l, e in enumerate(embs):
            per_layer[l].append(e)
    return [np.stack(c) for c in per_layer]


def _site_decodability_by_layer(layers, site):
    """Chance-corrected site decodability at each layer + deep summaries."""
    per = [probe_decodability(_standardize(layers[l])[0], site)
           for l in range(len(layers))]
    return {"per_layer": [float(x) for x in per],
            "L4": float(per[-1]),
            "L3L4_mean": float(np.mean(per[-2:]))}


def main_real():
    import torch
    import build_cohort as B
    root = os.environ["ABIDE_ROOT"]
    arch = os.environ.get("ARCH", "swin")
    seeds = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
    cap = int(os.environ.get("SUBJ_PER_SITE", "60"))
    out = os.environ.get("MULTISEED_JSON", f"multiseed_{arch}.json")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = B.make_transform()
    subs = sorted(B.iter_t1(root), key=lambda x: x[1])
    vols, sites, persite = [], [], {}
    for sid, t1 in subs:
        rel = os.path.relpath(t1, root); site = rel.split(os.sep)[0]
        if cap and persite.get(site, 0) >= cap:
            continue
        try:
            vols.append(tfm(t1))                       # preprocess ONCE
        except Exception as ex:
            print(f"  [skip] {sid}: {str(ex)[:60]}"); continue
        sites.append(site); persite[site] = persite.get(site, 0) + 1
        if len(vols) % 50 == 0:
            print(f"  ...loaded {len(vols)} volumes")
    if len(sites) < 40:
        sys.exit(f"only {len(sites)} usable subjects")
    site = np.array(sites)
    print(f"[data] {len(site)} volumes cached; arch={arch}; seeds={seeds}; "
          f"{len(persite)} sites; dev={dev}")

    per_seed = {}
    for sd in seeds:
        if arch == "swin":
            layers = _deep_layers_swin(sd, B.IMG, dev, vols)
        else:
            layers = _deep_layers_other(arch, sd, B.IMG, dev, vols)
        r = _site_decodability_by_layer(layers, site)
        per_seed[sd] = r
        print(f"[seed {sd}] {arch}: L4 site decodability = {r['L4']:.3f}  "
              f"(L3-L4 mean {r['L3L4_mean']:.3f})")

    l4 = np.array([per_seed[s]["L4"] for s in seeds])
    deep = np.array([per_seed[s]["L3L4_mean"] for s in seeds])
    summary = {
        "arch": arch, "seeds": seeds, "n": int(len(site)), "n_sites": len(persite),
        "per_seed": {str(s): per_seed[s] for s in seeds},
        "L4_mean": float(l4.mean()), "L4_std": float(l4.std(ddof=0)),
        "L4_min": float(l4.min()), "L4_max": float(l4.max()),
        "L4_range": float(l4.max() - l4.min()),
        "L3L4_mean_mean": float(deep.mean()), "L3L4_mean_std": float(deep.std(ddof=0)),
    }
    print(json.dumps(summary, indent=2))
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] {arch}: L4 site decodability {l4.mean():.3f} +/- {l4.std(ddof=0):.3f} "
          f"across seeds {seeds} (range {l4.max()-l4.min():.3f}) -> {out}")


def main_sandbox():
    """Different seeds must give a STABLE, high site decodability when the site
    signal lives in the input: emulate per-seed random projections of a shared
    per-site low-level pattern and check the across-seed spread is small."""
    print("[sandbox] multi-seed intrinsic-stability logic")
    rng = np.random.RandomState(0)
    ns, nps, d = 6, 100, 128
    site = np.array([s for s in range(ns) for _ in range(nps)])
    base = rng.randn(ns, d) * 1.5                          # per-site input pattern
    X = np.stack([base[s] for s in site]) + rng.randn(ns * nps, d) * 1.0
    vals = []
    for seed in (0, 1, 2):                                 # "random encoders"
        r = np.random.RandomState(1000 + seed)
        W = r.randn(d, d)                                  # random-init projection
        emb = _standardize(X @ W)[0]
        vals.append(probe_decodability(emb, site))
    vals = np.array(vals)
    spread = vals.max() - vals.min()
    print(f"  per-seed site decodability = {np.round(vals,3).tolist()}")
    print(f"  across-seed spread = {spread:.3f}")
    ok = (vals.min() > 0.5) and (spread < 0.25)            # high AND stable
    print("SANDBOX", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    a = ap.parse_args()
    sys.exit(main_sandbox() if a.sandbox else (main_real() or 0))
