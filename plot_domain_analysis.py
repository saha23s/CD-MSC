"""Domain analysis visualizations for the BioDCASE 2026 CD-MSC challenge.

Generates four publication-quality figures:
  Fig 1 — Domain sample imbalance (train vs test)
  Fig 2 — Species × Domain co-occurrence heatmap
  Fig 3 — Mean log-mel spectrogram per domain
  Fig 4 — B1 LODO per-fold BA_seen vs BA_unseen + per-domain species recall heatmap

Usage:
    python plot_domain_analysis.py [--out figures/]
"""

import argparse
import collections
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPECIES_SHORT = {
    "Aedes aegypti":           "Ae. aeg",
    "Aedes albopictus":        "Ae. alb",
    "Culex quinquefasciatus":  "Cx. qui",
    "Anopheles gambiae":       "An. gam",
    "Anopheles arabiensis":    "An. ara",
    "Anopheles dirus":         "An. dir",
    "Culex pipiens":           "Cx. pip",
    "Anopheles minimus":       "An. min",
    "Anopheles stephensi":     "An. ste",
}
SPECIES_ORDER = list(SPECIES_SHORT.keys())
DOMAINS       = ["D1", "D2", "D3", "D4", "D5"]

# From split_summary.json / technical report
TRAIN_COUNTS = {"D1": 634,    "D2": 230,    "D3": 364,    "D4": 80,     "D5": 212339}
VAL_COUNTS   = {"D1": 96,     "D2": 30,     "D3": 53,     "D4": 3,      "D5": 30334}
TEST_COUNTS  = {"D1": 3335,   "D2": 524,    "D3": 262,    "D4": 117,    "D5": 22979}

LODO_B1 = {   # corrected BA_seen/BA_unseen (held-out domain)
    "D1": {"BA_seen": 0.589, "BA_unseen": 0.051},
    "D2": {"BA_seen": 0.598, "BA_unseen": 0.246},
    "D3": {"BA_seen": 0.499, "BA_unseen": 0.265},
    "D4": {"BA_seen": 0.625, "BA_unseen": 0.100},
    "D5": {"BA_seen": 0.209, "BA_unseen": 0.067},
}

PALETTE = sns.color_palette("colorblind")
DOMAIN_COLORS = {d: PALETTE[i] for i, d in enumerate(DOMAINS)}

plt.rcParams.update({
    "font.family":   "sans-serif",
    "font.size":     11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi":    150,
})


# ---------------------------------------------------------------------------
# Fig 1 — Domain imbalance
# ---------------------------------------------------------------------------

def fig_domain_imbalance(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    splits = {"Train": TRAIN_COUNTS, "Test": TEST_COUNTS}
    for ax, (split_name, counts) in zip(axes, splits.items()):
        total = sum(counts.values())
        bars = ax.bar(
            DOMAINS,
            [counts[d] for d in DOMAINS],
            color=[DOMAIN_COLORS[d] for d in DOMAINS],
            edgecolor="white", linewidth=0.5,
        )
        # Annotate percentage
        for bar, d in zip(bars, DOMAINS):
            pct = 100 * counts[d] / total
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=8,
            )
        ax.set_title(f"{split_name} split  (N={total:,})")
        ax.set_xlabel("Recording domain")
        ax.set_ylabel("Number of clips")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_ylim(0, max(counts.values()) * 1.18)
        sns.despine(ax=ax)

    fig.suptitle("Domain sample distribution — D5 accounts for 99.4% of training data",
                 fontsize=12, y=1.02)
    path = out / "fig1_domain_imbalance.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Fig 2 — Species × Domain co-occurrence
# ---------------------------------------------------------------------------

def fig_species_domain_matrix(test_items: list, out: Path) -> None:
    # Count clips per (species, domain)
    counts = np.zeros((len(SPECIES_ORDER), len(DOMAINS)), dtype=int)
    for item in test_items:
        r = SPECIES_ORDER.index(item["species"])
        c = DOMAINS.index(item["domain"])
        counts[r, c] += 1

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    sns.heatmap(
        counts,
        ax=ax,
        xticklabels=DOMAINS,
        yticklabels=[SPECIES_SHORT[s] for s in SPECIES_ORDER],
        annot=True, fmt="d",
        cmap="Blues",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Test clips"},
    )
    ax.set_xlabel("Recording domain")
    ax.set_ylabel("Species")
    ax.set_title("Species × Domain co-occurrence (test set)")
    # Mark unseen domain per species (from split_summary)
    unseen_domain_by_species = {
        "Aedes aegypti": "D3", "Aedes albopictus": "D2",
        "Culex quinquefasciatus": "D1", "Anopheles gambiae": "D1",
        "Anopheles arabiensis": "D1", "Anopheles dirus": "D4",
        "Culex pipiens": "D3", "Anopheles minimus": "D2",
        "Anopheles stephensi": "D4",
    }
    for r, sp in enumerate(SPECIES_ORDER):
        ud = unseen_domain_by_species.get(sp)
        if ud and ud in DOMAINS:
            c = DOMAINS.index(ud)
            ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False,
                                       edgecolor="red", lw=2))
    ax.text(1.02, 0.5, "Red border = official unseen domain",
            transform=ax.transAxes, va="center", rotation=90,
            fontsize=8, color="red")

    path = out / "fig2_species_domain_matrix.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Fig 3 — Mean log-mel spectrogram per domain
# ---------------------------------------------------------------------------

def fig_mean_spectrograms(test_items: list, out: Path) -> None:
    hop_length = 80
    sr         = 8000

    # Compute per-domain mean across full clips
    domain_sum   = {d: None for d in DOMAINS}
    domain_count = {d: 0    for d in DOMAINS}
    for item in test_items:
        d = item["domain"]
        feat = item["feature"].astype(np.float32)  # [T, 64]
        if domain_sum[d] is None:
            domain_sum[d] = feat.mean(axis=0)  # mean over time → [64]
        else:
            domain_sum[d] += feat.mean(axis=0)
        domain_count[d] += 1

    domain_means = {
        d: domain_sum[d] / domain_count[d]
        for d in DOMAINS if domain_count[d] > 0
    }

    fig, axes = plt.subplots(1, 5, figsize=(14, 3.5), constrained_layout=True,
                              sharey=True)
    mel_bins = np.arange(64)
    freq_labels = [f"{int(b * sr / 2 / 64 / 1000)}" for b in mel_bins]

    vmin = min(v.min() for v in domain_means.values())
    vmax = max(v.max() for v in domain_means.values())

    for ax, d in zip(axes, DOMAINS):
        mean_spec = domain_means[d]    # [64]
        im = ax.imshow(
            mean_spec[np.newaxis, :].T,  # [64, 1]
            aspect="auto", origin="lower",
            vmin=vmin, vmax=vmax, cmap="magma",
        )
        ax.set_title(f"{d}\n(N={domain_count[d]:,})", fontsize=10)
        ax.set_xticks([])
        ax.set_xlabel("Mean over time")
        if ax == axes[0]:
            ax.set_ylabel("Mel bin")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Log-mel energy")
    fig.suptitle("Mean log-mel spectrum per recording domain (test set)", fontsize=12)

    path = out / "fig3_mean_spectrograms.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Fig 4a — LODO per-fold generalization gap
# ---------------------------------------------------------------------------

def fig_lodo_gap(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    x      = np.arange(len(DOMAINS))
    width  = 0.35
    seen   = [LODO_B1[d]["BA_seen"]   for d in DOMAINS]
    unseen = [LODO_B1[d]["BA_unseen"] for d in DOMAINS]

    bars_seen   = ax.bar(x - width/2, seen,   width, label="BA (seen domains)",
                         color=PALETTE[0], alpha=0.85)
    bars_unseen = ax.bar(x + width/2, unseen, width, label="BA (held-out domain)",
                         color=PALETTE[1], alpha=0.85)

    # DSG annotation
    for i, d in enumerate(DOMAINS):
        dsg = abs(LODO_B1[d]["BA_seen"] - LODO_B1[d]["BA_unseen"])
        ax.annotate(
            f"Δ{dsg:.2f}",
            xy=(i, max(seen[i], unseen[i]) + 0.02),
            ha="center", va="bottom", fontsize=8, color="dimgray",
        )

    ax.axhline(0.146, color="gray", linestyle="--", linewidth=1,
               label=f"Mean BA_unseen = 0.146")
    ax.set_xticks(x)
    ax.set_xticklabels([f"LODO-{d}" for d in DOMAINS])
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0, 0.85)
    ax.set_title("LODO generalization gap — B1 baseline (seed 42)")
    ax.legend(loc="upper right")
    sns.despine(ax=ax)

    path = out / "fig4a_lodo_gap.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Fig 4b — Per-species per-domain recall heatmap (B1)
# ---------------------------------------------------------------------------

def fig_species_domain_recall(pred_path: Path, out: Path) -> None:
    rows = [json.loads(l) for l in pred_path.read_text().splitlines() if l.strip()]

    # recall[species][domain] = correct / total
    correct = collections.defaultdict(lambda: collections.defaultdict(int))
    total   = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in rows:
        sp = r["true_species_label"]
        d  = r["true_domain_label"]
        total[sp][d]   += 1
        correct[sp][d] += int(r["predicted_species_index"] == r["true_species_index"])

    recall = np.full((len(SPECIES_ORDER), len(DOMAINS)), np.nan)
    for ri, sp in enumerate(SPECIES_ORDER):
        for ci, d in enumerate(DOMAINS):
            if total[sp][d] > 0:
                recall[ri, ci] = correct[sp][d] / total[sp][d]

    fig, ax = plt.subplots(figsize=(6.5, 5), constrained_layout=True)
    mask = np.isnan(recall)
    sns.heatmap(
        recall, ax=ax, mask=mask,
        xticklabels=DOMAINS,
        yticklabels=[SPECIES_SHORT[s] for s in SPECIES_ORDER],
        annot=True, fmt=".2f",
        cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": "Per-class recall"},
        annot_kws={"size": 8},
    )
    # Gray out missing combos
    sns.heatmap(
        recall, ax=ax, mask=~mask,
        xticklabels=DOMAINS,
        yticklabels=[SPECIES_SHORT[s] for s in SPECIES_ORDER],
        cmap=["#dddddd"], cbar=False,
        linewidths=0.4, linecolor="white",
    )
    ax.set_xlabel("Recording domain")
    ax.set_ylabel("Species")
    ax.set_title("Per-species recall by domain — B1 baseline LODO-D1 test\n"
                 "(held-out = D1; gray = combination not in test set)")

    path = out / "fig4b_species_domain_recall.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="figures/", help="Output directory")
    parser.add_argument("--feature-pkl", default="Development_data/feature/test_features.pkl")
    parser.add_argument("--pred-jsonl",
        default="outputs/LODO_D1_seed42_B64_E100_earlystop_min10_pati5/best_model_eval/test_predictions.jsonl")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading test features …")
    with open(args.feature_pkl, "rb") as f:
        data = pickle.load(f)
    test_items = data["items"]
    print(f"  {len(test_items):,} test clips loaded")

    print("Generating figures …")
    fig_domain_imbalance(out)
    fig_species_domain_matrix(test_items, out)
    fig_mean_spectrograms(test_items, out)
    fig_lodo_gap(out)
    fig_species_domain_recall(Path(args.pred_jsonl), out)
    print(f"\nAll figures saved to {out}/")


if __name__ == "__main__":
    main()
