"""Fig 5 — embed_dim x contrastive_proj_dim 2D sweep (LODO, seed 42).

Panel A: heatmap of mean test BA_unseen over folds D1-D4 (higher = better).
Panel B: heatmap of mean test DSG over folds D1-D4 (lower = better).
Panel C: per-cell BA_unseen, mean +/- SEM across the 4 folds, sorted. Shows that
         fold-to-fold variance dwarfs the between-cell spread -> the sweep is
         noise-dominated and no cell is significantly best.

Reads metrics straight from the run output directories (job 9831317).

Usage (from repo root, .venv active):
    python scripts/paper/fig5_projdim_sweep.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
OUTPUTS = ROOT / "data" / "Development_data" / "outputs"

EMBED_DIMS = [16, 32, 64, 128]
PROJ_DIMS = [64, 128, 256]
FOLDS = ["D1", "D2", "D3", "D4"]

# ── publication style ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7.5,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi":        150,
})
CB = sns.color_palette("colorblind")

# ── load metrics from run dirs ───────────────────────────────────────────────
pat = re.compile(r"LODO_(D\d)_seed42_.*_baldann_dicl_tau02_e(\d+)_p(\d+)$")
# (embed, proj) -> {fold: {BA_unseen, DSG}}
cells: dict = {}
for d in OUTPUTS.glob("*baldann_dicl_tau02_e*_p*"):
    m = pat.search(d.name)
    if not m:
        continue
    fold, e, p = m.group(1), int(m.group(2)), int(m.group(3))
    metrics_path = d / "best_model_eval" / "test_metrics.json"
    if not metrics_path.exists():
        continue
    met = json.loads(metrics_path.read_text())
    cells.setdefault((e, p), {})[fold] = met

# Build grids [embed, proj]
ba_mean = np.full((len(EMBED_DIMS), len(PROJ_DIMS)), np.nan)
dsg_mean = np.full_like(ba_mean, np.nan)
for i, e in enumerate(EMBED_DIMS):
    for j, p in enumerate(PROJ_DIMS):
        folds = cells.get((e, p), {})
        ba = [folds[f]["BA_unseen"] for f in folds]
        dsg = [folds[f]["DSG"] for f in folds]
        if ba:
            ba_mean[i, j] = np.mean(ba)
            dsg_mean[i, j] = np.mean(dsg)

# Per-cell mean / SEM for panel C
cell_stats = []  # (label, embed, mean, sem)
for e in EMBED_DIMS:
    for p in PROJ_DIMS:
        ba = [cells[(e, p)][f]["BA_unseen"] for f in cells.get((e, p), {})]
        ba = np.array(ba, dtype=float)
        sem = ba.std(ddof=1) / np.sqrt(len(ba)) if len(ba) > 1 else 0.0
        cell_stats.append((f"e{e}/p{p}", e, ba.mean(), sem))
cell_stats.sort(key=lambda r: r[2], reverse=True)

best_idx = np.unravel_index(np.nanargmax(ba_mean), ba_mean.shape)

# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))

# Panel A — BA_unseen heatmap (higher better)
sns.heatmap(
    ba_mean, ax=axes[0], cmap="viridis", annot=True, fmt=".3f",
    xticklabels=PROJ_DIMS, yticklabels=EMBED_DIMS,
    cbar_kws={"label": "BA_unseen", "shrink": 0.8}, linewidths=0.5, linecolor="white",
)
axes[0].add_patch(mpatches.Rectangle(
    (best_idx[1], best_idx[0]), 1, 1, fill=False, edgecolor="red", lw=2.0))
axes[0].set_xlabel("contrastive_proj_dim")
axes[0].set_ylabel("embed_dim")
axes[0].set_title("(a) Mean test BA$_{unseen}$ (higher better)")

# Panel B — DSG heatmap (lower better)
sns.heatmap(
    dsg_mean, ax=axes[1], cmap="rocket_r", annot=True, fmt=".3f",
    xticklabels=PROJ_DIMS, yticklabels=EMBED_DIMS,
    cbar_kws={"label": "DSG", "shrink": 0.8}, linewidths=0.5, linecolor="white",
)
axes[1].set_xlabel("contrastive_proj_dim")
axes[1].set_ylabel("embed_dim")
axes[1].set_title("(b) Mean test DSG (lower better)")

# Panel C — per-cell mean +/- SEM, sorted
labels = [r[0] for r in cell_stats]
means = np.array([r[2] for r in cell_stats])
sems = np.array([r[3] for r in cell_stats])
embed_of = [r[1] for r in cell_stats]
color_map = {e: CB[k] for k, e in enumerate(EMBED_DIMS)}
colors = [color_map[e] for e in embed_of]

y = np.arange(len(labels))
axes[2].errorbar(
    means, y, xerr=sems, fmt="none", ecolor="0.4", elinewidth=1.0, capsize=2.5, zorder=1)
axes[2].scatter(means, y, c=colors, s=36, zorder=2, edgecolor="white", linewidth=0.5)
# grand-mean reference line + fold-noise band annotation
grand = means.mean()
axes[2].axvline(grand, color="0.6", ls="--", lw=0.8, zorder=0)
axes[2].set_yticks(y)
axes[2].set_yticklabels(labels)
axes[2].invert_yaxis()
axes[2].set_xlabel("test BA$_{unseen}$  (mean $\\pm$ SEM over folds)")
axes[2].set_title("(c) Cells overlap within fold noise")
axes[2].spines["top"].set_visible(False)
axes[2].spines["right"].set_visible(False)
legend_items = [mpatches.Patch(color=color_map[e], label=f"embed_dim={e}") for e in EMBED_DIMS]
axes[2].legend(handles=legend_items, loc="lower right", framealpha=0.85)

fig.tight_layout(pad=0.8)

stem = "fig5_projdim_sweep"
for ext, kw in [("pdf", {}), ("png", {"dpi": 150}), ("svg", {})]:
    fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", **kw)
print(f"Saved to {OUT}/{stem}.{{pdf,png,svg}}")
print(f"Best cell: embed={EMBED_DIMS[best_idx[0]]}, proj={PROJ_DIMS[best_idx[1]]}, "
      f"BA_unseen={ba_mean[best_idx]:.3f}")
plt.close(fig)
