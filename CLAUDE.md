# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Official baseline for the **BioDCASE 2026 Cross-Domain Mosquito Species Classification (CD-MSC)** challenge. The task is to classify mosquito species from audio under domain shift — models are evaluated separately on seen and unseen recording domains. The primary metric is `BAunseen` (balanced accuracy on unseen domains). The baseline intentionally has a large domain shift gap (DSG ≈ 0.70); improving cross-domain generalisation is the research goal.

## Commands

All scripts take `--config` (default: `configs/default_experiment.json`).

```bash
# 1. Extract log-mel features for all splits (run once; skips if config signature matches)
python extract_features.py --config configs/default_experiment.json

# 2. Train a single run
python train.py --config configs/default_experiment.json

# 3. Evaluate a saved checkpoint on validation or test
python evaluate.py --config configs/default_experiment.json \
  --checkpoint outputs/MTRCNN_seed42_B64_E100_earlystop_min10_pati5/model/model_best.pth \
  --split test \
  --metrics-out outputs/.../manual_test_metrics.json \
  --predictions-out outputs/.../manual_test_predictions.jsonl

# 4. Predict a single audio file
python predict.py --config configs/default_experiment.json \
  --checkpoint outputs/.../model_best.pth \
  --audio Development_data/raw_audio/S_1_D_5_16608.wav

# 5. Run the full 10-seed benchmark
python run_multi_seed_experiments.py --config configs/multi_seed_experiment.json

# 6. Regenerate report assets from saved outputs
python generate_technical_report_assets.py
python plot_official_main_figure.py
```

## Data Setup

Raw audio must be placed at `Development_data/raw_audio/` (not committed — in `.gitignore`). The repo already contains precomputed features in `Development_data/feature/` and split metadata in `Development_data/metadata/`. Re-run `extract_features.py` only if you change feature config keys.

Audio filenames encode both labels: `S_<speciesID>_D_<domainID>_<clipIndex>.wav`. Parsing happens in [framework/metadata.py](framework/metadata.py) via `parse_file_id`.

## Architecture

### Pipeline

```
raw_audio/*.wav
  → extract_features.py            (LogMelSpectrogram, torchlibrosa)
  → Development_data/feature/*.pkl  (pickled list of {file_id, feature, species_label, domain_label, ...})
  → MosquitoFeatureDataset          (random crop to 2s during training, full clip at eval)
  → pad_collate_fn                  (zero-pads batch to longest clip, tracks lengths)
  → MTRCNNClassifier                (forward takes features + lengths)
  → {species_logits, domain_logits}
```

### Model — `framework/model.py`

`MTRCNNClassifier` has three parallel `MTRCNNBranch` modules with kernel sizes 3×3, 5×5, 7×7. Each branch is three `ConvStage` layers (Conv2d → BN → ReLU → AvgPool2d(2,2)) with increasing dilation rates (1, 2, 3) on the time axis only — frequency dilation stays at 1. After the three stages, `masked_mean_max` pools over the time dimension (mean + max of valid frames, ignoring padding), then a `Linear` projects the frequency axis to 1. The three branch outputs (each 64-dim) are concatenated → 192-dim → `Linear(192, 32)` with GELU → two heads: `species_classifier(32 → 9)` and `domain_classifier(32 → 5)`.

Variable-length handling is explicit: `output_lengths` tracks how the valid frame count shrinks through each conv+pool stage, and `masked_mean_max` masks out padding before pooling.

### Loss

`loss = CrossEntropy(species_logits, species_labels) + CrossEntropy(domain_logits, domain_labels)`

The domain head is auxiliary supervision only — it is **not** adversarial and does not enforce domain invariance.

### Config system — `framework/config.py`

`load_config` reads the JSON then calls `resolve_config`, which computes derived keys (`n_mels=64`, `fmin`, `fmax`, `win_length`, `hop_length`, `n_fft`) from `sample_rate`. Several derived keys are fixed regardless of what is in the JSON. Feature files are keyed by a SHA-256 `config_signature` over `FEATURE_CONFIG_KEYS`; `extract_features.py` skips re-extraction when the signature matches. Similarly, `train.py` skips completed runs by comparing `run_context.json` against the current config signature.

### Key `framework/` modules

| File | Responsibility |
|---|---|
| `acoustic_feature.py` | `LogMelSpectrogram` (torchlibrosa wrapper), waveform loading, per-split feature extraction and stats computation |
| `config.py` | Config loading, resolution, and SHA-256 signature helpers |
| `dataset.py` | `MosquitoFeatureDataset` (reads pickles, crops, normalises), `pad_collate_fn` |
| `engine.py` | `train_one_epoch`, `evaluate_model` (returns metrics + optional prediction rows with per-domain breakdowns) |
| `metadata.py` | Species/domain name↔index mappings, filename regex parser |
| `model.py` | `ConvStage`, `MTRCNNBranch`, `masked_mean_max`, `MTRCNNClassifier` |
| `utilization.py` | Device selection, seed setting, output dir management, experiment locking, checkpoint helpers |

## Evaluation Metrics

- `BAseen` / `BAunseen`: mean per-class recall on seen / unseen domain test samples
- `DSG = |BAunseen − BAseen|`: domain shift gap (lower is better)
- `evaluate.py` computes `BAseen`, `BAunseen`, `DSG` by reading `split_summary.json` to identify which test domains are unseen

## Reproducibility Notes

- Feature extraction is skipped when `config_signature` in the pickle matches the current config
- Training is skipped when `run_context.json` in the output dir matches the current resolved config
- If you change `dataset_root`, sampling, or split files, delete the relevant `.pkl` files and re-extract
- 10 fixed seeds: `42, 3407, 1234, 2023, 2024, 1024, 2048, 4096, 8192, 10086`
