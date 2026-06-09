"""Fig 3 — Side-by-side t-SNE: baseline vs. best system (LODO D1 fold).

Extracts 32-dim embeddings from both checkpoints on the D1 test set
(the held-out domain in fold D1), then plots domain-coloured t-SNE.

REQUIRES: GPU compute node and LODO D1 checkpoints for both systems.

Usage (on compute node, from repo root, .venv active):
    python scripts/paper/fig3_tsne.py \
        --baseline outputs/LODO_D1_seed42_B64_E100_earlystop_min10_pati5/model/model_best.pth \
        --best     outputs/LODO_D1_seed42_B64_E100_earlystop_min10_pati5_balanced_dann_dicl_proj128_tau02/model/model_best.pth \
        --config   configs/lodo_balanced_dann_dicl_proj128_tau02.json \
        --fold     D1

NOTE: The --config flag should point to the best-system config so that
      resolve_config correctly loads hyperparams. The feature pickle
      path is derived from the config; ensure D1 fold features exist.
"""

import argparse
import json
import pickle
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
CB = sns.color_palette("colorblind", 5)


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


def compute_tsne(embeddings, subsample=4000, seed=42):
    rng = np.random.default_rng(seed)
    if len(embeddings) > subsample:
        idx = rng.choice(len(embeddings), subsample, replace=False)
        return TSNE(n_components=2, perplexity=40,
                    random_state=seed, n_jobs=4).fit_transform(embeddings[idx]), idx
    coords = TSNE(n_components=2, perplexity=40,
                  random_state=seed, n_jobs=4).fit_transform(embeddings)
    return coords, np.arange(len(embeddings))


def plot_panel(ax, coords, labels, title):
    for i, name in enumerate(DOMAIN_NAMES):
        mask = labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[CB[i]], label=name, s=5, alpha=0.55, linewidths=0)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True,
                    help="Baseline LODO D1 checkpoint (.pth)")
    ap.add_argument("--best",     required=True,
                    help="Best-system LODO D1 checkpoint (.pth)")
    ap.add_argument("--config",   default="configs/lodo_baseline.json",
                    help="Config for baseline (used for feature paths)")
    ap.add_argument("--best-config",
                    default="configs/lodo_balanced_dann_dicl_proj128_tau02.json",
                    help="Config for best system")
    ap.add_argument("--fold",     default="D1",
                    help="Which LODO fold test set to use")
    ap.add_argument("--subsample", type=int, default=4000)
    args = ap.parse_args()

    device = choose_device("auto")

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8),
                             gridspec_kw={"wspace": 0.1})

    for ckpt_path, cfg_path, title, ax in [
        (args.baseline, args.config,      "Baseline", axes[0]),
        (args.best,     args.best_config, "Proposed system", axes[1]),
    ]:
        config   = load_config(cfg_path)
        # Override: load the held-out fold test features
        feat_key = f"lodo_{args.fold.lower()}_test"
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
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        print(f"Extracting embeddings: {title}")
        embs, doms, _ = extract_embeddings(model, loader, device)
        coords, idx   = compute_tsne(embs, args.subsample)
        plot_panel(ax, coords, doms[idx], title)

    # Shared legend
    handles = [mpatches.Patch(color=CB[i], label=d)
               for i, d in enumerate(DOMAIN_NAMES)]
    axes[1].legend(handles=handles, loc="lower right",
                   framealpha=0.85, borderpad=0.4)

    fig.suptitle("t-SNE of 32-dim embeddings (LODO D1 test set, coloured by domain)",
                 fontsize=9)
    fig.tight_layout(pad=0.5)

    stem = "fig3_tsne"
    for ext, kw in [("pdf", {}), ("png", {"dpi": 150}), ("svg", {})]:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", **kw)
    print(f"Saved to {OUT}/{stem}.{{pdf,png,svg}}")
    plt.close(fig)


if __name__ == "__main__":
    main()
