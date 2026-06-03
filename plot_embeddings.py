"""Extract penultimate embeddings from B1 checkpoint and plot t-SNE.

Coloured by (a) domain and (b) species on the test set.

Usage (on compute node with GPU):
    python plot_embeddings.py [--checkpoint <path>] [--out figures/]
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE

from framework.config import load_config
from framework.dataset import MosquitoFeatureDataset, pad_collate_fn
from framework.metadata import DOMAIN_NAMES, SPECIES_NAMES
from framework.utilization import build_model, choose_device, make_loader, split_feature_path, training_stats_path

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

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def extract_embeddings(model, dataloader, device):
    """Run forward pass, return 32-dim embeddings before classification heads."""
    model.eval()
    embeddings, domain_labels, species_labels = [], [], []

    # Hook to capture the shared embedding (output of frequency projection)
    captured = {}

    def _hook(module, inp, out):
        captured["emb"] = out.detach().cpu()

    # The shared embedding is the input to both heads.
    # For MTRCNN: species_classifier is an nn.Linear on the 32-dim embedding.
    # Register hook on species_classifier (captures its input = the embedding).
    hook_handle = model.species_classifier.register_forward_hook(_hook)

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            lengths  = batch["lengths"].to(device)
            model(features, lengths)
            embeddings.append(captured["emb"])
            domain_labels.extend(batch["domain_labels"].numpy())
            species_labels.extend(batch["species_labels"].numpy())

    hook_handle.remove()
    return (
        torch.cat(embeddings, dim=0).numpy(),
        np.array(domain_labels),
        np.array(species_labels),
    )


def plot_tsne(embeddings, labels, label_names, title, colors, out_path, subsample=5000):
    rng = np.random.default_rng(42)
    if len(embeddings) > subsample:
        idx = rng.choice(len(embeddings), subsample, replace=False)
        embeddings = embeddings[idx]
        labels     = labels[idx]

    print(f"  Running t-SNE on {len(embeddings)} points …")
    tsne   = TSNE(n_components=2, perplexity=40, random_state=42, n_jobs=4)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for i, name in enumerate(label_names):
        mask = labels == i
        if mask.sum() == 0:
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[i]], label=name,
            s=6, alpha=0.5, linewidths=0,
        )
    ax.legend(markerscale=3, loc="best", framealpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_xticks([]); ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)

    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/default_experiment.json")
    parser.add_argument("--checkpoint", default=
        "outputs/LODO_D1_seed42_B64_E100_earlystop_min10_pati5/model/model_best.pth")
    parser.add_argument("--out",        default="figures/")
    parser.add_argument("--subsample",  type=int, default=5000)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    device = choose_device(config["device"])

    print("Loading dataset …")
    dataset = MosquitoFeatureDataset(
        feature_pickle_path=split_feature_path(config, "test"),
        feature_stats_path=training_stats_path(config),
        max_train_frames=None,
        training=False,
        normalize_features=config["normalize_features"],
    )
    dataloader = make_loader(dataset, 64, False, config["num_workers"], device, pad_collate_fn)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_model(config, device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("Extracting embeddings …")
    embs, domain_labels, species_labels = extract_embeddings(model, dataloader, device)
    print(f"  Embeddings shape: {embs.shape}")

    pal_domain  = sns.color_palette("colorblind", len(DOMAIN_NAMES))
    pal_species = sns.color_palette("tab10",      len(SPECIES_NAMES))

    print("Plotting domain t-SNE …")
    plot_tsne(
        embs, domain_labels, DOMAIN_NAMES,
        title="t-SNE of B1 embeddings — coloured by domain",
        colors=pal_domain,
        out_path=out / "fig5a_tsne_domain.pdf",
        subsample=args.subsample,
    )

    print("Plotting species t-SNE …")
    plot_tsne(
        embs, species_labels,
        [SPECIES_SHORT[s] for s in SPECIES_NAMES],
        title="t-SNE of B1 embeddings — coloured by species",
        colors=pal_species,
        out_path=out / "fig5b_tsne_species.pdf",
        subsample=args.subsample,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
