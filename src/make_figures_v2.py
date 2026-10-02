#!/usr/bin/env python
"""Figures added in version 2 of the paper, built from the bundled results.

fig1_targets      site, sex, age and diagnosis on separate panels, by encoder and
                  depth (results/abide1_dominance_50splits_1p0mm.json)
fig4_raw_inputs   scores from inputs that pass through no trained model: the
                  downsampled T1 image (raw_voxel_baseline.py, ABIDE-I, n=331,
                  one split) and fMRI connectivity (results/fmri_dominance.json)

    python src/make_figures_v2.py            # writes figures/fig1_targets.* and fig4_raw_inputs.*
"""
import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "DejaVu Sans", "pdf.fonttype": 42,
})
OK = {"site": "#0072B2", "sex": "#E69F00", "age_r2": "#009E73", "asd": "#CC79A7"}
LAB = {"site": "Site", "sex": "Sex", "age_r2": "Age ($R^2$)", "asd": "Diagnosis"}
LAYERS = ["L0", "L1", "L2", "L3", "L4"]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
# Raw-voxel control, ABIDE-I 12^3 voxels (raw_voxel_baseline.py output).
RAW_T1 = {"site": 0.948, "sex": 0.330, "age_r2": 0.067}


def bars(ax, vals, title, floor=None):
    keys = list(vals)
    x = np.arange(len(keys))
    for i, k in enumerate(keys):
        ax.bar(i, vals[k], 0.62, color=OK[k], edgecolor="white" if k != "age_r2" else OK[k],
               hatch="///" if k == "age_r2" else None,
               facecolor=OK[k] if k != "age_r2" else "white", linewidth=1.2)
        ax.text(i, vals[k] + 0.02, f"{vals[k]:.2f}", ha="center", fontsize=9)
    if floor is not None:
        ax.axhline(floor, color="#666666", lw=1, ls="--")
        ax.text(len(keys) - 0.6, 0.97, f"dashed line: site permutation floor ({floor:.2f})",
                fontsize=8, color="#555555", ha="right", va="top")
    ax.set_xticks(x)
    ax.set_xticklabels([LAB[k] for k in keys])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="x", visible=False)
    ax.set_title(title, fontsize=10.5, loc="left")


def fig_raw():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True,
                             gridspec_kw={"width_ratios": [3, 4]})
    fmri = json.loads((ROOT / "results" / "fmri_dominance.json").read_text())["fmri-raw-cc200"]["layers"]["emb_L0"]
    bars(axes[0], RAW_T1, "(a) Structural MRI, $12^3$ voxels")
    bars(axes[1], {k: fmri[k] for k in ("site", "age_r2", "asd", "sex")},
         "(b) Resting-state connectivity")
    axes[0].set_ylabel("Decodability")
    fig.tight_layout()
    save(fig, "fig4_raw_inputs")


def load(path):
    """Accepts both result formats: scanner_dominance_ci (values are dicts with
    mean/lo/hi under 'linear') and scanner_dominance (plain floats)."""
    res = json.loads(Path(path).read_text())
    out = {}
    for tag, r in res.items():
        rows = {}
        for layer, row in r["layers"].items():
            src = row.get("linear", row)
            rows[layer] = {k: (src[k]["mean"] if isinstance(src.get(k), dict) else src.get(k, np.nan))
                           for k in ("site", "sex", "age_r2", "asd")}
        out[tag] = rows
    return out


def model_order(res):
    def rank(t):
        t = t.lower()
        return 0 if "brain" in t else 1 if "ct" in t else 2
    names = {0: "Brain-pretrained", 1: "CT-pretrained", 2: "Untrained"}
    tags = sorted(res, key=rank)
    return tags, [names[rank(t)] for t in tags]


def fig_targets(cohorts):
    targets = ["site", "sex", "age_r2", "asd"]
    fig, axes = plt.subplots(len(cohorts), 4, figsize=(11.5, 2.6 * len(cohorts)),
                             gridspec_kw={"wspace": 0.08, "hspace": 0.45})
    axes = np.atleast_2d(axes)
    im = None
    for r, (cname, res) in enumerate(cohorts):
        tags, labels = model_order(res)
        for c, k in enumerate(targets):
            ax = axes[r, c]
            M = np.array([[res[t][f"emb_{l}"][k] for l in LAYERS] for t in tags], float)
            im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.grid(False)
            ax.set_xticks(range(5)); ax.set_xticklabels(LAYERS)
            ax.set_yticks(range(len(tags)))
            ax.set_yticklabels(labels if c == 0 else [])
            ax.set_title(LAB[k] if len(cohorts) == 1 else f"{cname}: {LAB[k]}", fontsize=10.5)
            for i in range(M.shape[0]):
                for j in range(5):
                    v = M[i, j]
                    ax.text(j, i, "n/a" if not np.isfinite(v) else f"{v:.2f}", ha="center",
                            va="center", fontsize=8, color="white" if (np.isfinite(v) and v < 0.6) else "black")
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("Balanced accuracy (chance-corrected) or $R^2$ for age")
    save(fig, "fig1_targets")


def fig_depth(cohorts):
    fig, axes = plt.subplots(1, len(cohorts), figsize=(9.0, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(5)
    for ax, (cname, res) in zip(axes, cohorts):
        tags, _ = model_order(res)
        for k in ("site", "sex", "age_r2", "asd"):
            A = np.array([[res[t][f"emb_{l}"][k] for t in tags] for l in LAYERS], float)
            if not np.isfinite(A).any():
                continue
            ax.fill_between(x, np.nanmin(A, 1), np.nanmax(A, 1), color=OK[k], alpha=0.15, lw=0)
            ax.plot(x, np.nanmean(A, 1), color=OK[k], lw=2, marker="o", ms=4,
                    ls="--" if k == "age_r2" else "-", label=LAB[k])
        ax.set_xticks(x); ax.set_xticklabels(LAYERS)
        ax.set_title(cname, fontsize=11); ax.set_xlabel("Encoder depth")
        ax.set_ylim(-0.02, 1.0)
    axes[0].set_ylabel("Decodability (mean of 3 encoders)")
    axes[-1].legend(frameon=False, fontsize=8.5, loc="center right")
    save(fig, "fig2_depth_targets")


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    print(f"[fig] {OUT / name}.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--abide1", type=Path, default=ROOT / "results" / "abide1_dominance_50splits_1p0mm.json")
    a = ap.parse_args()
    fig_raw()
    fig_targets([("ABIDE-I", load(a.abide1))])
