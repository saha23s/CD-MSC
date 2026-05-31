# Notes — BioDCASE 2026 CD-MSC

---

## 2026-05-31

- Repo cloned and set up on Mila cluster
- Data lives in `/network/scratch/s/sahas/CD-MSC/Development_data/` (symlinked via `data/` and `Development_data/`)
- Unzipping `Development_data.zip` (3.6 GB, ~271k clips) — in progress as of session start
- Baseline numbers (10-seed, best checkpoint): BA_seen=0.879, BA_unseen=0.185, DSG=0.694
- Key imbalance: D5 has 265k/271k total clips; species 6/8/9 have <700 clips each
- `train_lodo.py` implemented — LODO CV, loads train+val pickles, splits by domain in memory, recomputes stats per fold, saves to `outputs/LODO_{fold}_seed{seed}_...`
- Spectrogram augmentation pipeline implemented (`framework/augmentation.py`): TimeMasking, FrequencyMasking, GaussianNoise, FrequencyShift, Mixup (batch-level). Config-driven via `"augmentation"` block. Wired into both `train.py` and `train_lodo.py`.
- AST implemented (`framework/ast_model.py`): 8×8 patches, d=192, 4 layers, ~2M params, 2D sinusoidal pos embed, attention masking for variable-length. Use `"model_type": "ast"` in config.
- Patch size rationale: wingbeat spans ~20/64 mel bins; 8×8 gives ~2–3 patches across wingbeat region, 200-token sequences at 2s crop — good tradeoff vs. 4×4 (800 tokens).
- AST LR set to 0.0005 (vs 0.001 for MTRCNN) — transformers typically need lower LR from scratch.
- Env not yet set up on cluster (requirements.txt needs Python 3.11+; use compute node).
- DANN implemented (`framework/gradient_reversal.py`): GRL inserts between shared embedding and domain head; λ schedule: `2/(1+exp(-10p))-1` × λ_max, p=epoch/total — starts ~0, saturates near λ_max by epoch 50. Both MTRCNN and AST support `set_grl_lambda()`.
- DANN config keys: `"domain_adversarial": true`, `"grl_lambda_max": 1.0` (default off).
- New configs: `lodo_dann.json` (MTRCNN+DANN), `lodo_ast_dann.json` (AST+DANN+aug).
- Experiment matrix: B1 (baseline LODO) → B2 (aug), D1 (DANN), E1 (AST) as independent ablations → D2 (AST+DANN) only if D1+E1 both win.

