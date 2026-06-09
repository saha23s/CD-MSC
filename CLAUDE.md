# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

Baseline for the **BioDCASE 2026 Cross-Domain Mosquito Species Classification** task. The model (`MTRCNNClassifier`) jointly classifies mosquito species (9 classes) and recording domain (5 classes) from 8 kHz audio using log-mel spectrograms. The key challenge metric is `species_balanced_accuracy` on unseen domains.

## Data Setup (Mila cluster)

Audio and features live in scratch, not home:

```
data/Development_data/ → actual audio + features (on scratch)
Development_data/      → data/Development_data   (convenience symlink for config paths)
outputs/               → data/Development_data/outputs  (symlink; actual data on scratch)
```

All three are `.gitignore`d. Config paths like `./Development_data/raw_audio` and `output_root: "./data/Development_data/outputs"` resolve correctly through these symlinks.

## Key Commands

```bash
# Install deps
pip install -r requirements.txt

# Extract features (skips if config signature matches)
python extract_features.py --config configs/default_experiment.json

# Train one seed
python train.py --config configs/default_experiment.json

# Evaluate a checkpoint
python evaluate.py --config configs/default_experiment.json \
  --checkpoint data/Development_data/outputs/<run_name>/model/model_best.pth \
  --split test \
  --metrics-out data/Development_data/outputs/<run_name>/manual_test_metrics.json \
  --predictions-out data/Development_data/outputs/<run_name>/manual_test_predictions.jsonl

# Predict a single file
python predict.py --config configs/default_experiment.json \
  --checkpoint data/Development_data/outputs/<run_name>/model/model_best.pth \
  --audio Development_data/raw_audio/<file>.wav

# Run full 10-seed benchmark
python run_multi_seed_experiments.py --config configs/multi_seed_experiment.json
```

Add `--overwrite` to `extract_features.py` or `train.py` to force re-run instead of loading cached results.

## Architecture

### Two-stage pipeline

**Stage 1 — Feature extraction** (`extract_features.py` + `framework/acoustic_feature.py`):  
Converts raw WAV → log-mel spectrogram pickles per split. Skips re-extraction when the config SHA-256 signature matches the stored one. Stats (mean/std for normalization) are computed from training split only.

**Stage 2 — Training** (`train.py`):  
Loads pre-extracted pickles via `MosquitoFeatureDataset`. Training applies random temporal cropping; val/test use full-length sequences with padding masks handled by `pad_collate_fn` → `masked_mean_max` pooling in the model.

### Model (`framework/model.py`)

`MTRCNNClassifier` has three parallel `MTRCNNBranch` instances with kernel sizes 3, 5, 7. Each branch: 3× `ConvStage` (Conv2d → BN → ReLU → AvgPool2d) → masked mean+max pooling over time → frequency projection. Branches are concatenated → 32-dim embedding → two heads: `species_classifier` and `domain_classifier`. Training loss = species CE + domain CE (equal weight, no adversarial objective in baseline).

### Config resolution (`framework/config.py`)

`load_config` calls `resolve_config` which derives `n_mels`, `fmin`, `fmax`, `win_length`, `hop_length`, `n_fft` from `sample_rate`. These derived values are **not** in the JSON — don't add them there. The `config_signature` hash ties feature files to the exact config that produced them; changing any `FEATURE_CONFIG_KEYS` field invalidates cached features.

### Reproducibility mechanism

`train.py` writes `run_context.json` containing hashes of the resolved config and all three split feature files. A run is skipped (outputs loaded) only when this JSON matches exactly. This means changing the config or re-extracting features with different settings will correctly trigger a new run.

### Output structure per run

```
data/Development_data/outputs/<experiment_name>/
├── resolved_config.json, run_context.json, run_summary.json
├── train.log, metrics.csv
├── model/model_best.pth, model_final.pth
├── best_model_eval/{validation,test}_{metrics.json,predictions.jsonl}
└── final_model_eval/{validation,test}_{metrics.json,predictions.jsonl}
```

Checkpoint selection metric: `val_species_balanced_accuracy`.

### Evaluation metrics

- `species_balanced_accuracy`: primary metric (mean per-class recall over species)
- `BA_seen` / `BA_unseen`: balanced accuracy on seen vs. unseen domains (test only, computed in `evaluate.py` using `split_summary.json`)
- `DSG = |BA_unseen - BA_seen|`: domain generalization gap — lower is better

### ID format

Audio IDs follow `S_<speciesID>_D_<domainID>_<clipIndex>`. Parsing is in `framework/metadata.py:parse_file_id`. Species and domain label indices are determined by list order in `SPECIES_NAMES` / `DOMAIN_NAMES` — don't reorder them.
