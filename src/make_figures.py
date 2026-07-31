#!/usr/bin/env python
"""Paper figures for "Frozen Brain-MRI Foundation Models Are Site Fingerprints".
Numbers are the verified outputs of scanner_dominance_ci (ABIDE-I) and
scanner_dominance N_SITES=6 (ABIDE-II). Colorblind-safe: Okabe-Ito categoricals +
viridis (perceptually uniform) for the magnitude heatmap. Saves PNG + PDF."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})
OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73"}
LAYERS = ["L0", "L1", "L2", "L3", "L4"]
DIMS = [48, 96, 192, 384, 768]
MODELS = ["brainseg", "ctssl", "random"]
MLAB = {"brainseg": "Brain-pretrained", "ctssl": "CT-pretrained", "random": "Random-init"}

# ---- ABIDE-I (repeated-holdout means; site with 90% CIs) --------------------
A1 = {
    "brainseg": dict(site=[.707, .825, .904, .947, .948],
                     lo=[.66, .75, .87, .91, .91], hi=[.75, .88, .95, .98, .99],
                     clin=[.275, .369, .405, .240, .254]),
    "ctssl":    dict(site=[.732, .912, .918, .938, .926],
                     lo=[.68, .87, .87, .89, .89], hi=[.78, .96, .96, .98, .97],
                     clin=[.262, .368, .392, .158, .187]),
    "random":   dict(site=[.680, .845, .917, .947, .956],
                     lo=[.63, .79, .86, .91, .92], hi=[.74, .90, .96, .98, .99],
                     clin=[.276, .365, .403, .298, .275]),
}
# ---- ABIDE-II (matched 6-way; repeated-holdout means + 90% intervals) --------
A2 = {
    "brainseg": dict(site=[.65, .80, .88, .87, .87], lo=[.57, .71, .82, .81, .81],
                     hi=[.75, .89, .96, .93, .93], clin=[.21, .49, .55, .49, .38]),
    "ctssl":    dict(site=[.65, .90, .86, .81, .79], lo=[.52, .82, .78, .73, .69],
                     hi=[.75, .96, .91, .91, .87], clin=[.21, .52, .53, .39, .25]),
    "random":   dict(site=[.66, .88, .90, .88, .88], lo=[.54, .80, .84, .80, .82],
                     hi=[.76, .96, .96, .93, .93], clin=[.25, .50, .49, .49, .44]),
}
OUT = "figures"
import os
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    print(f"[fig] {OUT}/{name}.png")


# ============================================================================
# FIG 1 -- decodability heatmap: site vs clinical, both cohorts (viridis 0..1)
# ============================================================================
def fig_heatmap():
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0),
                             gridspec_kw={"wspace": 0.12, "hspace": 0.35})
    panels = [("ABIDE-I", A1, "site", "Acquisition site"),
              ("ABIDE-I", A1, "clin", "Best clinical (sex/age/ASD)"),
              ("ABIDE-II", A2, "site", "Acquisition site"),
              ("ABIDE-II", A2, "clin", "Best clinical (sex/age/ASD)")]
    im = None
    for idx, (ax, (cohort, D, key, sub)) in enumerate(zip(axes.ravel(), panels)):
        M = np.array([D[m][key] for m in MODELS])          # models x layers
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(5)); ax.set_xticklabels(LAYERS)
        if idx % 2 == 0:                                   # only left column labels rows
            ax.set_yticks(range(3)); ax.set_yticklabels([MLAB[m] for m in MODELS])
        else:
            ax.set_yticks([])
        ax.set_title(f"{cohort} — {sub}", fontsize=10.5, pad=6)
        ax.grid(False)
        for i in range(3):
            for j in range(5):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=8.5)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("Decodability (chance-corrected)")
    save(fig, "fig1_decodability_heatmap")


# ============================================================================
# FIG 2 -- depth curves: site vs clinical peak by layer, both cohorts
# ============================================================================
def fig_depth():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    x = np.arange(5)
    for ax, (cohort, D) in zip(axes, [("ABIDE-I", A1), ("ABIDE-II", A2)]):
        site = np.array([[D[m]["site"][j] for m in MODELS] for j in range(5)])
        clin = np.array([[D[m]["clin"][j] for m in MODELS] for j in range(5)])
        for arr, col, lab in [(site, OK["blue"], "Acquisition site"),
                              (clin, OK["orange"], "Best clinical")]:
            mu = arr.mean(1); lo = arr.min(1); hi = arr.max(1)
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, linewidth=0)
            ax.plot(x, mu, color=col, lw=2, marker="o", ms=5, label=lab)
        ax.axhline(0, color="#BBBBBB", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([f"{l}\n{d}d" for l, d in zip(LAYERS, DIMS)])
        ax.set_title(cohort, fontsize=11)
        ax.set_xlabel("Encoder depth (Swin scale)")
        ax.set_ylim(-0.02, 1.0)
    axes[0].set_ylabel("Decodability (mean over 3 encoders;\nband = min–max)")
    # direct labels on the right panel
    axes[1].text(4.05, A2["random"]["site"][4], "site", color=OK["blue"],
                 va="center", fontsize=10, fontweight="bold")
    axes[1].text(4.05, np.mean([A2[m]["clin"][4] for m in MODELS]), "clinical",
                 color=OK["orange"], va="center", fontsize=10, fontweight="bold")
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    save(fig, "fig2_depth_curves")


# ============================================================================
# FIG 3 -- the headline: random ≈ pretrained (deep-layer site decodability)
# ============================================================================
def fig_random():
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    deep = [3, 4]                                          # L3,L4 average
    cohorts = ["ABIDE-I", "ABIDE-II"]
    data = {"ABIDE-I": A1, "ABIDE-II": A2}
    w = 0.25; x = np.arange(len(cohorts))
    for k, m in enumerate(MODELS):
        vals, errs = [], []
        for c in cohorts:
            D = data[c]
            vals.append(np.mean([D[m]["site"][j] for j in deep]))
            if "lo" in D[m]:
                lo = np.mean([D[m]["lo"][j] for j in deep])
                hi = np.mean([D[m]["hi"][j] for j in deep])
                errs.append([vals[-1] - lo, hi - vals[-1]])
            else:
                errs.append([0, 0])
        errs = np.array(errs).T
        bars = ax.bar(x + (k - 1) * w, vals, w, yerr=errs, capsize=3,
                      color=[OK["blue"], OK["orange"], OK["green"]][k],
                      label=MLAB[m], edgecolor="white", linewidth=0.5)
        for xi, v, e in zip(x + (k - 1) * w, vals, errs[1]):
            ax.text(xi, v + e + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(0.9, color="#888888", lw=1, ls="--")
    ax.text(0.5, 0.9, "≈0.9", color="#555555", fontsize=8.5, va="center", ha="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    ax.set_xticks(x); ax.set_xticklabels(cohorts)
    ax.set_ylabel("Deep-layer site decodability\n(L3–L4 mean; bars = 90% interval)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=3, loc="lower center", fontsize=9,
              bbox_to_anchor=(0.5, -0.28))
    save(fig, "fig3_random_vs_pretrained")


if __name__ == "__main__":
    fig_heatmap(); fig_depth(); fig_random()
    print("done")
