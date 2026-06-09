"""Fig 2 — System diagram: FBS-Mix + MTRCNN + DANN + DiCL.

Pure matplotlib — no external diagram tools required.

Usage (from repo root, .venv active):
    python scripts/paper/fig2_system_diagram.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
OUT  = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CB = sns.color_palette("colorblind")

plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    8.5,
    "figure.dpi":   150,
})

fig, ax = plt.subplots(figsize=(7.0, 2.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3.5)
ax.axis("off")


def box(ax, x, y, w, h, label, sublabel=None, color="#DDEEFF",
        edgecolor="#3366AA", fontsize=8, bold=False):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor=edgecolor, linewidth=1.1)
    ax.add_patch(rect)
    kw = {"ha": "center", "va": "center", "fontsize": fontsize,
          "fontweight": "bold" if bold else "normal"}
    if sublabel:
        ax.text(x + w/2, y + h*0.62, label, **kw)
        ax.text(x + w/2, y + h*0.28, sublabel,
                ha="center", va="center", fontsize=fontsize - 1,
                color="#555555")
    else:
        ax.text(x + w/2, y + h/2, label, **kw)


def arrow(ax, x1, y1, x2, y2, color="#444444", lw=1.4, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"))


# ── Input spectrogram ────────────────────────────────────────────────────────
box(ax, 0.1, 0.9, 1.2, 1.7,
    "Log-mel\nspectrogram",
    sublabel="64 bins × T frames",
    color="#F5F5F5", edgecolor="#888888")

# ── FBS-Mix block ────────────────────────────────────────────────────────────
# Domain band (low)
fbs_x = 1.65
box(ax, fbs_x, 1.95, 0.9, 0.55,
    "Mix\nbins 0–8",
    color="#FFE0CC", edgecolor=CB[3], fontsize=7.5)
# Wingbeat band (protected)
box(ax, fbs_x, 1.25, 0.9, 0.60,
    "Protect\nbins 9–35",
    color="#CCE8FF", edgecolor=CB[0], fontsize=7.5)
# Mixed band
box(ax, fbs_x, 0.9, 0.9, 0.28,
    "Pass  36–63",
    color="#F0F0F0", edgecolor="#888888", fontsize=6.5)

# FBS-Mix outer label
ax.text(fbs_x + 0.45, 2.72, "FBS-Mix", ha="center", fontsize=8.5,
        fontweight="bold", color=CB[3])
ax.add_patch(mpatches.FancyBboxPatch(
    (fbs_x - 0.05, 0.82), 1.02, 2.0,
    boxstyle="round,pad=0.05",
    facecolor="none", edgecolor=CB[3], linewidth=1.0, linestyle="--"))

# ── MTRCNN ───────────────────────────────────────────────────────────────────
mtrcnn_x = 3.0
box(ax, mtrcnn_x, 0.85, 1.55, 1.8,
    "MTRCNN backbone",
    sublabel="3 branches: k=3,5,7\n→ masked mean+max pool\n→ z  (32-dim)",
    color="#E8F4E8", edgecolor="#228B22", fontsize=7.5)

# ── Projection MLP (for DiCL) ────────────────────────────────────────────────
proj_x = 4.9
box(ax, proj_x, 1.3, 0.8, 0.75,
    "Proj MLP",
    sublabel="32→128→128",
    color="#FFF5CC", edgecolor="#BB8800", fontsize=7)

# ── Embedding circle ─────────────────────────────────────────────────────────
emb_x = 5.95
emb_y = 1.75
circle = plt.Circle((emb_x, emb_y), 0.28, color="#E8E8F8",
                     ec="#5555AA", linewidth=1.2, zorder=3)
ax.add_patch(circle)
ax.text(emb_x, emb_y, "z", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#5555AA", zorder=4)
ax.text(emb_x, emb_y - 0.48, "32-dim", ha="center", fontsize=6.5,
        color="#5555AA")

# ── Species head ─────────────────────────────────────────────────────────────
sp_x = 7.1
box(ax, sp_x, 2.1, 1.2, 0.65,
    "Species head",
    sublabel="9-class softmax",
    color="#E0F0FF", edgecolor="#0055BB", fontsize=7)

# ── Domain head + GRL ────────────────────────────────────────────────────────
box(ax, sp_x, 1.0, 1.2, 0.65,
    "GRL  →  Domain",
    sublabel="5-class softmax",
    color="#FFE8E8", edgecolor="#BB0000", fontsize=7)

# ── DiCL loss node ───────────────────────────────────────────────────────────
dicl_x = 7.6
box(ax, dicl_x, 0.2, 0.95, 0.55,
    r"$\mathcal{L}_\mathrm{DiCL}$",
    sublabel="cross-domain\ncontrast",
    color="#FFF0CC", edgecolor="#BB8800", fontsize=7.5)

# ── Arrows ───────────────────────────────────────────────────────────────────
arrow(ax, 1.3,  1.75, 1.65, 1.75)       # spec → FBS-Mix
arrow(ax, 2.55, 1.75, 3.0,  1.75)       # FBS-Mix → MTRCNN
arrow(ax, 4.55, 1.75, 4.9,  1.75)       # MTRCNN → z / proj
arrow(ax, 5.73, 1.75, 7.1,  2.42)       # z → species head
arrow(ax, 5.73, 1.75, 7.1,  1.32)       # z → domain head

# Projection arrow (from proj to DiCL)
arrow(ax, 5.7,  1.67, 7.6,  0.75,
      color="#BB8800", lw=1.0)

ax.text(8.35, 2.42, r"$\hat{y}_\mathrm{sp}$",
        ha="center", va="center", fontsize=9, color="#0055BB")
ax.text(8.35, 1.32, r"$\hat{y}_\mathrm{dom}$",
        ha="center", va="center", fontsize=9, color="#BB0000")

fig.tight_layout(pad=0.4)

stem = "fig2_system_diagram"
for ext, kw in [("pdf", {}), ("png", {"dpi": 150}), ("svg", {})]:
    fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", **kw)
print(f"Saved to {OUT}/{stem}.{{pdf,png,svg}}")
plt.close(fig)
