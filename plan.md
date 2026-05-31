# Research Plan — BioDCASE 2026 CD-MSC

**Challenge:** Cross-Domain Mosquito Species Classification  
**Primary metric:** `species_balanced_accuracy` on unseen domains  
**DSG** (`|BA_unseen - BA_seen|`) — lower is better  
**Baseline (10-seed best-ckpt):** BA_seen=0.879, BA_unseen=0.185, DSG=0.694

---

## Goal

Improve cross-domain generalization (reduce DSG) while maintaining or improving BA_unseen.

---

## Experiments

| ID | Description | Status | BA_seen | BA_unseen | DSG | Notes |
|----|-------------|--------|---------|-----------|-----|-------|
| B0 | Baseline MTRCNN (10-seed) | done | 0.879 | 0.185 | 0.694 | released checkpoint |
| B1 | MTRCNN LODO (D1–D5 folds, no aug) | planned | — | — | — | `train_lodo.py --fold Dx` |
| B2 | MTRCNN LODO + SpecAugment + Gaussian noise | planned | — | — | — | `configs/lodo_aug.json` |
| E1 | AST (8×8 patch, d=192, L=4) LODO, no aug | planned | — | — | — | `configs/lodo_ast.json` (aug off) |
| E2 | AST LODO + SpecAugment + Gaussian noise | planned | — | — | — | `configs/lodo_ast.json` |
| D1 | MTRCNN LODO + DANN (GRL, λ_max=1.0) | planned | — | — | — | `configs/lodo_dann.json` |
| D2 | AST LODO + DANN + SpecAugment | planned | — | — | — | `configs/lodo_ast_dann.json` |
| H1 | MTRCNN LODO + HPSS (p=0.5, α_min=0) | planned | — | — | — | `configs/lodo_hpss.json` |
| M1 | MTRCNN LODO + MixStyle (α=0.1) | planned | — | — | — | `use_mixstyle: true` in config |
| M2 | MTRCNN LODO + domain-balanced sampling | planned | — | — | — | `domain_balanced_sampling: true` |
| T1 | B1 + TTBN at test time | planned | — | — | — | `ttbn: true` — zero retraining, apply to B1 ckpt |

---

## Next Steps

- [ ] Wait for Development_data unzip to finish
- [ ] Run `pip install -r requirements.txt` on compute node
- [ ] Feature extraction: `python extract_features.py`
- [ ] Smoke test AST: `python framework/ast_model.py`
- [ ] Submit B1 LODO baseline (all 5 folds)
- [ ] Compare B1 vs B2 (aug effect), B1 vs D1 (DANN effect), E1 vs B1 (AST vs MTRCNN)
- [ ] Run B2, D1, H1, M1, M2 alongside B1 (all independent ablations)
- [ ] T1: re-evaluate B1 checkpoints with ttbn=true (free win, no retraining)
- [ ] Run D2 only after D1 and E2 show individually positive results

---

## Ideas / Directions

- **DANN** (`framework/gradient_reversal.py` implemented) — GRL between shared embedding and domain head; λ annealed 0→λ_max via DANN sigmoid schedule; `"domain_adversarial": true` in config. Works for both MTRCNN and AST.
- Data augmentation for domain shift (SpecAugment, pitch shift, noise)
- **HPSS** (`framework/augmentation.py` implemented) — Fitzgerald 2010 Wiener masking; attenuates percussive component (domain noise) while preserving harmonic (wingbeat); `kernel_harm=17, kernel_perc=9, alpha_min=0.0, p=0.5`; enable with `"hpss": {"enabled": true}` in config.
- Class-balanced sampling to address species imbalance
- **Domain-balanced sampling** (`framework/utilization.py` implemented) — `WeightedRandomSampler` upweights D1–D4 ~200× to equal expected domain frequency per epoch. `"domain_balanced_sampling": true`.
- **MixStyle** (`framework/mixstyle.py` implemented) — mixes instance-norm stats `(μ, σ)` between samples during training. MTRCNN: shared instance after ConvStage 0 across all branches. AST: after patch_embed before flatten. `"use_mixstyle": true, "mixstyle_alpha": 0.1, "mixstyle_p": 0.5`.
- **TTBN** (test-time batch normalisation, implemented in `engine.py`) — at test inference, BN layers use batch stats instead of stored D5 stats. Zero retraining. `"ttbn": true`. Applied to test split only.
- **LODO cross-validation** (`train_lodo.py` implemented) — Leave-One-Domain-Out CV, `--fold D1..D5`, trains on 4 domains, validates on held-out domain; gives true out-of-domain BA per fold. Not yet run.
- **AST** (`framework/ast_model.py` implemented) — 8×8 patch, d=192, 4 transformer layers, ~2M params, 2D sinusoidal pos embed, attention masking for variable-length. Select with `"model_type": "ast"` in config.

---

## Milestones

| Date | Milestone |
|------|-----------|
| — | Baseline reproduced locally |
| — | First improved run > BA_unseen 0.185 |
| — | Challenge submission |
