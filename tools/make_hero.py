#!/usr/bin/env python
"""Animated header for the README (assets/hero.gif).

Bars show the deep-layer (L3 and L4 averaged) ABIDE-I scores reported in the
paper: acquisition site against the highest clinical or demographic target, for
two pretrained encoders, an untrained encoder and the raw image with no model.

    python tools/make_hero.py            # writes assets/hero.gif
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / "assets" / "hero.gif"

# Deep-layer means from the paper (Table 1 and the raw-image control).
ROWS = [("Brain-pretrained", 0.948, 0.247),
        ("CT-pretrained", 0.932, 0.173),
        ("Untrained (random weights)", 0.952, 0.287),
        ("Raw image, no model", 0.948, 0.330)]
SITE, CLIN = "#0072B2", "#E69F00"
INK, MUTED, BG, CARD, LINE = "#1b2430", "#5b6675", "#ffffff", "#f7f9fc", "#e3e8ef"

fams = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in fams), "DejaVu Sans")

W, H, DPI = 9.6, 3.6, 150
N_TITLE, N_GROW, N_CAPTION = 10, 34, 12


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def frame(i):
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 37.5); ax.axis("off")
    ax.add_patch(plt.Rectangle((1.2, 1.2), 97.6, 35.1, fc=CARD, ec=LINE, lw=1.2, zorder=0))
    a_title = ease(i / N_TITLE)
    ax.text(4, 32.4, "Site is decodable before pretraining", fontsize=19, weight="bold",
            color=INK, alpha=a_title, va="center")
    ax.text(4, 29.0, "How well a frozen brain-MRI model's output predicts the scanning site, "
            "against the best clinical target (ABIDE-I, deep layers)", fontsize=10.5, color=MUTED,
            alpha=a_title, va="center")
    g = ease((i - N_TITLE) / N_GROW)
    x0, x1 = 30, 88
    ax.add_patch(plt.Rectangle((x0, 24.6), 1.4, 1.1, fc=SITE))
    ax.text(x0 + 2.2, 25.15, "acquisition site", fontsize=9.5, color=INK, va="center")
    ax.add_patch(plt.Rectangle((x0 + 19, 24.6), 1.4, 1.1, fc=CLIN))
    ax.text(x0 + 21.2, 25.15, "best clinical or demographic target", fontsize=9.5, color=INK, va="center")
    for k, (name, site, clin) in enumerate(ROWS):
        y = 20.6 - k * 4.6
        ax.text(x0 - 1.5, y, name, ha="right", va="center", fontsize=10.5, color=INK,
                weight="bold" if k >= 2 else "normal")
        for dy, val, col in ((0.15, site, SITE), (-1.7, clin, CLIN)):
            ax.add_patch(plt.Rectangle((x0, y + dy), (x1 - x0), 1.55, fc=LINE, ec="none"))
            ax.add_patch(plt.Rectangle((x0, y + dy), (x1 - x0) * val * g, 1.55, fc=col, ec="none"))
        if g > 0:
            ax.text(x0 + (x1 - x0) * site * g + 0.8, y + 0.95, f"{site * g:.2f}", va="center",
                    fontsize=9.5, color=SITE, weight="bold")
            ax.text(x0 + (x1 - x0) * clin * g + 0.8, y - 0.9, f"{clin * g:.2f}", va="center",
                    fontsize=9.5, color="#a86f00")
    c = ease((i - N_TITLE - N_GROW) / N_CAPTION)
    ax.text(50, 3.6, "The same score with or without pretraining: the site is already in the scan.",
            fontsize=11.5, color=INK, alpha=c, weight="bold", ha="center", va="center")
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return img


def main():
    n = N_TITLE + N_GROW + N_CAPTION
    frames = [frame(i) for i in range(n + 1)]
    durations = [70] * n + [4200]
    pal = frames[-1].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    q[0].save(OUT, save_all=True, append_images=q[1:], duration=durations, loop=0, optimize=True, disposal=1)
    frames[-1].save(OUT.with_name("hero_still.png"))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
