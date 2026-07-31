#!/usr/bin/env python
"""arch_encoders.py -- random-initialized encoders of DIFFERENT architectures, to
test whether the intrinsic site fingerprint is specific to the SwinUNETR family or
holds across architectures. Since the finding is that a RANDOM encoder already
separates sites, no pretrained checkpoint is needed: we just build a random-init
ViT (non-hierarchical transformer) and a random-init 3-D ResNet (pure CNN) and
read per-stage global-average-pooled features, exactly as for the Swin encoder.

Used by extract_multilayer.py when ARCH is 'vit' or 'resnet' (default 'swin' keeps
the original SwinUNETR path). Self-test (needs torch+monai):
    python arch_encoders.py --sandbox
"""
from __future__ import annotations

import os
import sys
import numpy as np


def _gap(f):
    """[B,C,...] -> [C] global-average-pool over all non-channel spatial dims."""
    return f.mean(dim=[i for i in range(2, f.ndim)]).flatten().cpu().numpy().astype(np.float64)


def build_encoder(arch, img, seed=0):
    """Return a frozen, eval-mode, random-initialized encoder for `arch`."""
    import torch
    torch.manual_seed(seed)
    if arch == "vit":
        from monai.networks.nets import ViT
        net = ViT(in_channels=1, img_size=img, patch_size=(16, 16, 16),
                  hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
                  classification=False, spatial_dims=3)
    elif arch == "resnet":
        from monai.networks.nets import resnet18
        net = resnet18(spatial_dims=3, n_input_channels=1, num_classes=2)
    else:
        raise ValueError(f"arch_encoders handles 'vit'/'resnet', not {arch!r}")
    net = net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _vit_layers(net, x):
    import torch
    with torch.no_grad():
        _, hs = net(x)                       # hs: list of [B, N, hidden] per block
    idx = np.linspace(0, len(hs) - 1, 5).astype(int)   # 5 evenly-spaced depths
    return [hs[i][0].mean(dim=0).cpu().numpy().astype(np.float64) for i in idx]


def _resnet_layers(net, x):
    import torch
    outs, handles = {}, []
    for name in ("maxpool", "layer1", "layer2", "layer3", "layer4"):
        m = getattr(net, name)
        handles.append(m.register_forward_hook(
            lambda mod, i, o, n=name: outs.__setitem__(n, o.detach())))
    with torch.no_grad():
        net(x)
    for h in handles:
        h.remove()
    return [_gap(outs[n]) for n in ("maxpool", "layer1", "layer2", "layer3", "layer4")]


def encode_layers(net, arch, x):
    """x: [1,1,D,H,W] -> list of 5 per-stage GAP embeddings (numpy)."""
    if arch == "vit":
        return _vit_layers(net, x)
    if arch == "resnet":
        return _resnet_layers(net, x)
    raise ValueError(arch)


def main_sandbox():
    import torch
    print("[sandbox] random-init ViT + ResNet per-layer extraction")
    img = (96, 96, 96)
    x = torch.randn(1, 1, *img)
    for arch in ("vit", "resnet"):
        try:
            net = build_encoder(arch, img)
            embs = encode_layers(net, arch, x)
            dims = [e.shape[0] for e in embs]
            finite = all(np.isfinite(e).all() for e in embs)
            print(f"  {arch}: {len(embs)} layers, dims {dims}, finite={finite}")
        except Exception as e:
            print(f"  {arch}: FAILED ({str(e)[:80]})")
            return 1
    print("SANDBOX PASS")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sandbox", action="store_true")
    a = ap.parse_args()
    sys.exit(main_sandbox() if a.sandbox else 0)
