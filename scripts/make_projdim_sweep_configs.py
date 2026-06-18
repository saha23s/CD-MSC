"""Generate the embed_dim x contrastive_proj_dim 2D sweep configs.

Clones the current best LODO config (balanced + DANN + DiCL tau=0.2) and varies
only two dimensions:

    embed_dim            -- shared representation bottleneck the species head reads
                            (model.py: MTRCNNClassifier). Smaller = stronger
                            regularization, the key knob for unseen-domain
                            generalization.
    contrastive_proj_dim -- size of the auxiliary DiCL projection head (discarded
                            at inference; only shapes the contrastive gradient).

Each cell gets a unique ``experiment_tag`` so train_lodo.py writes to a distinct
output directory. Also emits a manifest (one "<config> <fold>" pair per line)
consumed by sbatch/Q1_projdim_sweep.sbatch as a SLURM array task map.

Usage:
    python scripts/make_projdim_sweep_configs.py
"""

import json
from pathlib import Path

# --- sweep axes ------------------------------------------------------------
EMBED_DIMS = [16, 32, 64, 128]
PROJ_DIMS = [64, 128, 256]
FOLDS = ["D1", "D2", "D3", "D4"]

BASE_CONFIG = Path("configs/lodo_balanced_dann_dicl_proj128_tau02.json")
OUT_DIR = Path("configs/sweep_projdim")
MANIFEST = OUT_DIR / "manifest.txt"


def main() -> None:
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config_paths = []
    for embed_dim in EMBED_DIMS:
        for proj_dim in PROJ_DIMS:
            cfg = dict(base)
            cfg["embed_dim"] = embed_dim
            cfg["contrastive_proj_dim"] = proj_dim
            cfg["experiment_tag"] = f"baldann_dicl_tau02_e{embed_dim}_p{proj_dim}"

            path = OUT_DIR / f"e{embed_dim}_p{proj_dim}.json"
            path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
            config_paths.append(path)

    with MANIFEST.open("w", encoding="utf-8") as handle:
        for path in config_paths:
            for fold in FOLDS:
                handle.write(f"{path} {fold}\n")

    n_cells = len(config_paths)
    print(f"Wrote {n_cells} configs to {OUT_DIR}/")
    print(f"Manifest: {MANIFEST} ({n_cells * len(FOLDS)} runs = {n_cells} cells x {len(FOLDS)} folds)")


if __name__ == "__main__":
    main()
