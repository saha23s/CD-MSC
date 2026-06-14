"""Fig 3 — t-SNE diagnostics: baseline vs. best system (LODO D1 fold).

Extracts 32-dim embeddings from both checkpoints on the D1 test set (the
held-out domain in fold D1) and renders three views:

  fig3_tsne          — coloured by DOMAIN (domain-collapse story)
  fig3_tsne_species  — coloured by SPECIES (which species cluster / smear)
  fig3_tsne_gambiae  — An. gambiae only: seen domains vs held-out D1
                       (visualises the per-species transfer failure)

REQUIRES: a compute node and LODO D1 checkpoints for both systems.

Usage (on compute node, from repo root, .venv active):
    python scripts/paper/fig3_tsne.py \
        --baseline data/Development_data/outputs/LODO_D1_seed42_B64_E100_earlystop_min10_pati5/model/model_best.pth \
        --best     data/Development_data/outputs/LODO_D1_seed42_B64_E100_earlystop_min10_pati5_balanced_dann_dicl_proj128_tau02/model/model_best.pth \
        --config   configs/lodo_baseline.json \
        --best-config configs/lodo_balanced_dann_dicl_proj128_tau02.json \
        --fold     D1

NOTE: the feature pickle path is derived from each config; ensure D1 fold
      features exist.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from framework.config import load_config
from framework.dataset import MosquitoFeatureDataset, pad_collate_fn
from framework.metadata import DOMAIN_NAMES, SPECIES_NAMES
from framework.utilization import (
    build_model, choose_device, make_loader,
    split_feature_path, training_stats_path,
)

GAMBIAE = SPECIES_NAMES.index("Anopheles gambiae")  # = 3

OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

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
CB_DOM = sns.color_palette("colorblind", 5)
CB_SP  = sns.color_palette("husl", len(SPECIES_NAMES))


def extract_embeddings(model, dataloader, device):
    model.eval()
    embeddings, domain_labels, species_labels = [], [], []

    captured = {}
    def _hook(module, inp, out):
        captured["emb"] = inp[0].detach().cpu()

    handle = model.species_classifier.register_forward_hook(_hook)
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            lengths  = batch["lengths"].to(device)
            model(features, lengths)
            embeddings.append(captured["emb"])
            domain_labels.extend(batch["domain_labels"].numpy())
            species_labels.extend(batch["species_labels"].numpy())
    handle.remove()

    return (
        torch.cat(embeddings).numpy(),
        np.array(domain_labels),
        np.array(species_labels),
    )


def select_subsample(n, species, subsample, seed):
    """Pick <=subsample indices, always keeping every gambiae point so the
    rare malaria-vector class survives subsampling for the overlay panel."""
    rng = np.random.default_rng(seed)
    if n <= subsample:
        return np.arange(n)
    forced = np.flatnonzero(species == GAMBIAE)
    rest = np.flatnonzero(species != GAMBIAE)
    n_fill = max(0, subsample - len(forced))
    fill = rng.choice(rest, min(n_fill, len(rest)), replace=False)
    idx = np.concatenate([forced, fill])
    rng.shuffle(idx)
    return idx


def compute_tsne(embeddings, seed=42):
    return TSNE(n_components=2, perplexity=40,
                random_state=seed, n_jobs=4).fit_transform(embeddings)


def plot_domain_panel(ax, coords, doms, title):
    for i, name in enumerate(DOMAIN_NAMES):
        m = doms == i
        if m.sum() == 0:
            continue
        ax.scatter(coords[m, 0], coords[m, 1], c=[CB_DOM[i]], label=name,
                   s=5, alpha=0.55, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


def plot_species_panel(ax, coords, species, title):
    for i, name in enumerate(SPECIES_NAMES):
        m = species == i
        if m.sum() == 0:
            continue
        ax.scatter(coords[m, 0], coords[m, 1], c=[CB_SP[i]], label=name,
                   s=5, alpha=0.55, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


def plot_gambiae_panel(ax, coords, doms, species, held_idx, title):
    """Grey = everything else; blue = gambiae in SEEN domains;
    red = gambiae in the HELD-OUT domain (the unseen test we fail on)."""
    other = species != GAMBIAE
    g_seen = (species == GAMBIAE) & (doms != held_idx)
    g_unseen = (species == GAMBIAE) & (doms == held_idx)
    ax.scatter(coords[other, 0], coords[other, 1], c="0.82", s=4,
               alpha=0.5, linewidths=0)
    ax.scatter(coords[g_seen, 0], coords[g_seen, 1], c="#2166AC",
               s=18, alpha=0.85, linewidths=0,
               label=f"gambiae seen (n={g_seen.sum()})")
    ax.scatter(coords[g_unseen, 0], coords[g_unseen, 1], c="#B2182B",
               s=18, alpha=0.85, linewidths=0,
               label=f"gambiae held-out (n={g_unseen.sum()})")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="lower right", framealpha=0.85, borderpad=0.4)
    sns.despine(ax=ax, left=True, bottom=True)


def save(fig, stem):
    for ext, kw in [("pdf", {}), ("png", {"dpi": 150}), ("svg", {})]:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", **kw)
    print(f"Saved {OUT}/{stem}.{{pdf,png,svg}}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--best",     required=True)
    ap.add_argument("--config",   default="configs/lodo_baseline.json")
    ap.add_argument("--best-config",
                    default="configs/lodo_balanced_dann_dicl_proj128_tau02.json")
    ap.add_argument("--fold",     default="D1")
    ap.add_argument("--subsample", type=int, default=4000)
    args = ap.parse_args()

    device = choose_device("auto")
    held_idx = DOMAIN_NAMES.index(args.fold)

    systems = [
        (args.baseline, args.config,      "Baseline"),
        (args.best,     args.best_config, "Proposed system"),
    ]

    # Extract + embed each system once; reuse for all three views.
    results = []  # (title, coords, doms, species)
    for ckpt_path, cfg_path, title in systems:
        config = load_config(cfg_path)
        feat_path = split_feature_path(config, "test")
        stats_path = training_stats_path(config)

        dataset = MosquitoFeatureDataset(
            feature_pickle_path=feat_path,
            feature_stats_path=stats_path,
            max_train_frames=None,
            training=False,
            normalize_features=config["normalize_features"],
        )
        loader = make_loader(dataset, 64, False, config["num_workers"],
                             device, pad_collate_fn)

        model = build_model(config, device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        print(f"Extracting embeddings: {title}")
        embs, doms, species = extract_embeddings(model, loader, device)
        idx = select_subsample(len(embs), species, args.subsample, seed=42)
        coords = compute_tsne(embs[idx])
        results.append((title, coords, doms[idx], species[idx]))

    # ── View 1: domain-coloured ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8),
                             gridspec_kw={"wspace": 0.1})
    for ax, (title, coords, doms, _sp) in zip(axes, results):
        plot_domain_panel(ax, coords, doms, title)
    handles = [mpatches.Patch(color=CB_DOM[i], label=d)
               for i, d in enumerate(DOMAIN_NAMES)]
    axes[1].legend(handles=handles, loc="lower right",
                   framealpha=0.85, borderpad=0.4)
    fig.suptitle(f"t-SNE of 32-dim embeddings (LODO {args.fold} test, by domain)",
                 fontsize=9)
    fig.tight_layout(pad=0.5)
    save(fig, "fig3_tsne")

    # ── View 2: species-coloured ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8),
                             gridspec_kw={"wspace": 0.1})
    for ax, (title, coords, _d, species) in zip(axes, results):
        plot_species_panel(ax, coords, species, title)
    handles = [mpatches.Patch(color=CB_SP[i], label=n)
               for i, n in enumerate(SPECIES_NAMES)]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               framealpha=0.85, borderpad=0.4)
    fig.suptitle(f"t-SNE of 32-dim embeddings (LODO {args.fold} test, by species)",
                 fontsize=9)
    fig.tight_layout(pad=0.5)
    save(fig, "fig3_tsne_species")

    # ── View 3: gambiae seen vs held-out ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8),
                             gridspec_kw={"wspace": 0.1})
    for ax, (title, coords, doms, species) in zip(axes, results):
        plot_gambiae_panel(ax, coords, doms, species, held_idx, title)
    fig.suptitle(
        f"An. gambiae embeddings (LODO {args.fold} test): "
        f"seen domains vs held-out {args.fold}", fontsize=9)
    fig.tight_layout(pad=0.5)
    save(fig, "fig3_tsne_gambiae")


if __name__ == "__main__":
    main()
