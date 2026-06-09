"""Fig 4 — Ablation horizontal bar chart (presentation / supplementary use).

Shows LODO D1-D4 mean BA_unseen for all ablation methods,
color-coded by method family. Not in the 4-page paper (Table 1 covers
the same content), but useful for talks and posters.

Usage (from repo root, .venv active):
    python scripts/paper/fig4_ablation_bars.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
OUT  = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

CB = sns.color_palette("colorblind")

plt.rcParams.update({
    "font.family":    "sans-serif",
    "font.size":      9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi":     150,
})

# ── Data (from plan.md, seed 42, D1-D4 mean) ────────────────────────────────
# (label, BA_unseen, color_group)
RESULTS = [
    # Proposed / best
    ("Bal + DANN + DiCL\n+proj128 + τ0.2 (Proposed)", 0.378, "proposed"),
    ("Bal + DANN + DiCL + proj128",                    0.371, "proposed"),
    # Balanced + DANN variants
    ("Bal + DANN",                                     0.368, "dann"),
    ("Bal + DANN + DiCL",                              0.340, "dann"),
    ("Bal + DANN + FBS-Mix",                           0.345, "dann"),
    ("Bal + DANN + Wingbeat feat.",                    0.339, "dann"),
    # Balanced only
    ("Balanced + Species-only",                        0.324, "balance"),
    ("Balanced + DiCL",                                0.321, "balance"),
    ("Balanced",                                       0.321, "balance"),
    ("Balanced + MixStyle",                            0.262, "balance"),  # ~= balanced alone
    # Baselines
    ("DANN (unbalanced)",                              0.169, "baseline"),
    ("MixStyle (unbalanced)",                          0.167, "baseline"),
    ("MTRCNN baseline",                                0.165, "baseline"),
]

# Sort ascending by BA_unseen
RESULTS = sorted(RESULTS, key=lambda x: x[1])

GROUP_COLORS = {
    "proposed": CB[2],
    "dann":     CB[0],
    "balance":  CB[4],
    "baseline": "lightgray",
}

labels   = [r[0] for r in RESULTS]
values   = [r[1] for r in RESULTS]
groups   = [r[2] for r in RESULTS]
colors   = [GROUP_COLORS[g] for g in groups]
y_pos    = np.arange(len(RESULTS))

fig, ax = plt.subplots(figsize=(5.5, 4.8))

bars = ax.barh(y_pos, values, color=colors, edgecolor="white",
               linewidth=0.5, height=0.72)

# Baseline dashed line
ax.axvline(0.165, color="gray", lw=0.9, ls="--", zorder=0)
ax.text(0.165 + 0.004, -0.9, "Baseline", color="gray",
        fontsize=7, va="top")

# Value annotations
for i, v in enumerate(values):
    ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=7.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("LODO D1–D4 mean BA_unseen")
ax.set_title("Method ablation (seed 42)")
ax.set_xlim(0.10, 0.43)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
legend_items = [
    mpatches.Patch(color=GROUP_COLORS["proposed"], label="Proposed system"),
    mpatches.Patch(color=GROUP_COLORS["dann"],     label="Balanced + DANN variants"),
    mpatches.Patch(color=GROUP_COLORS["balance"],  label="Balanced only"),
    mpatches.Patch(color=GROUP_COLORS["baseline"], label="Unbalanced baseline"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=7.5,
          framealpha=0.85, borderpad=0.5)

fig.tight_layout(pad=0.6)

stem = "fig4_ablation_bars"
for ext, kw in [("pdf", {}), ("png", {"dpi": 150}), ("svg", {})]:
    fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", **kw)
print(f"Saved to {OUT}/{stem}.{{pdf,png,svg}}")
plt.close(fig)
