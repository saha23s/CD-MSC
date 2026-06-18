"""Fig 6 — Held-out (unseen) confusion matrices, Perch vs DANN+DiCL.

Pools the *unseen* test partition across LODO folds D1-D4 (each from its own
fold model; D5 excluded as degenerate — it removes 99.4%% of training data) and
row-normalises so the diagonal is per-species recall and off-diagonals show
where errors are routed. Two panels: frozen-Perch baseline vs the best MTRCNN
method (Balanced + DANN + DiCL, proj128, tau0.2).

Usage (from repo root, .venv active):
    python scripts/paper/fig6_unseen_confusion.py
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS = ROOT / "data" / "Development_data" / "outputs"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
})

FOLDS = [1, 2, 3, 4]  # D5 excluded (degenerate)

METHODS = [
    ("Frozen Perch",
     lambda d: f"LODO_D{d}_seed42_B128_E100_earlystop_min10_pati10_perch_lodo"),
    ("MTRCNN + DANN + DiCL\n(proj128, $\\tau$0.2)",
     lambda d: f"LODO_D{d}_seed42_B64_E100_earlystop_min10_pati5_balanced_dann_dicl_proj128_tau02"),
]

# Genus-grouped species order with short labels.
SPECIES = [
    ("Aedes aegypti",          "Ae. aegypti"),
    ("Aedes albopictus",       "Ae. albopictus"),
    ("Culex pipiens",          "Cx. pipiens"),
    ("Culex quinquefasciatus", "Cx. quinque."),
    ("Anopheles arabiensis",   "An. arabiensis"),
    ("Anopheles dirus",        "An. dirus"),
    ("Anopheles gambiae",      "An. gambiae"),
    ("Anopheles minimus",      "An. minimus"),
    ("Anopheles stephensi",    "An. stephensi"),
]
FULL = [s[0] for s in SPECIES]
SHORT = [s[1] for s in SPECIES]
IDX = {name: i for i, name in enumerate(FULL)}
N = len(FULL)


def pred_path(run: str):
    base = OUTPUTS / run
    for root, _, files in os.walk(base):
        if "test_predictions.jsonl" in files:
            return Path(root) / "test_predictions.jsonl"
    return None


def pooled_confusion(name_fn):
    """Counts[true, pred] pooled over the unseen partition of folds D1-D4."""
    counts = np.zeros((N, N), dtype=np.int64)
    for d in FOLDS:
        p = pred_path(name_fn(d))
        if p is None:
            raise FileNotFoundError(name_fn(d))
        with open(p) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("evaluation_partition") != "unseen":
                    continue
                counts[IDX[r["true_species_label"]], IDX[r["predicted_species_label"]]] += 1
    return counts


fig, axes = plt.subplots(1, len(METHODS), figsize=(11.0, 5.0), constrained_layout=True)

for ax, (title, name_fn) in zip(axes, METHODS):
    counts = pooled_confusion(name_fn)
    totals = counts.sum(axis=1, keepdims=True)
    recall = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)
    recall[totals[:, 0] == 0] = np.nan  # species absent from unseen folds

    ylabels = [f"{s}  (n={int(t)})" for s, t in zip(SHORT, totals[:, 0])]

    sns.heatmap(
        recall, ax=ax, cmap="rocket_r", vmin=0, vmax=1,
        annot=True, fmt=".0%", annot_kws={"size": 6.5},
        linewidths=0.5, linecolor="white",
        xticklabels=SHORT, yticklabels=ylabels,
        cbar=(ax is axes[-1]),
        cbar_kws={"label": "fraction of true-species clips", "shrink": 0.7},
    )
    # mark the diagonal (correct cell) with a box
    for i in range(N):
        if totals[i, 0] > 0:
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                       edgecolor="#2ca02c", lw=1.6))
    ax.set_title(title)
    ax.set_xlabel("Predicted species")
    ax.set_ylabel("True species" if ax is axes[0] else "")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

fig.suptitle("Held-out (unseen) confusion, pooled over LODO folds D1–D4  "
             "(green = correct; row-normalised)", fontsize=11)

stem = "fig6_unseen_confusion"
for ext, kw in [("pdf", {}), ("png", {"dpi": 150}), ("svg", {})]:
    fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", **kw)
print(f"Saved to {OUT}/{stem}.{{pdf,png,svg}}")
plt.close(fig)
