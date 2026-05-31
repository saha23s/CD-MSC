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

---

## Next Steps

- [ ] Wait for Development_data unzip to finish
- [ ] Run `pip install -r requirements.txt` on compute node
- [ ] Feature extraction: `python extract_features.py`
- [ ] Smoke test AST: `python framework/ast_model.py`
- [ ] Submit B1 LODO baseline (all 5 folds)
- [ ] Compare B1 vs B2 (aug effect) and E1 vs B1 (AST vs MTRCNN)

---

## Ideas / Directions

- Domain-invariant training (adversarial domain head, gradient reversal)
- Data augmentation for domain shift (SpecAugment, pitch shift, noise)
- Class-balanced sampling to address species imbalance
- Domain-balanced sampling (D5 dominates with 265k/271k clips)
- **LODO cross-validation** (`train_lodo.py` implemented) — Leave-One-Domain-Out CV, `--fold D1..D5`, trains on 4 domains, validates on held-out domain; gives true out-of-domain BA per fold. Not yet run.
- **AST** (`framework/ast_model.py` implemented) — 8×8 patch, d=192, 4 transformer layers, ~2M params, 2D sinusoidal pos embed, attention masking for variable-length. Select with `"model_type": "ast"` in config.

---

## Milestones

| Date | Milestone |
|------|-----------|
| — | Baseline reproduced locally |
| — | First improved run > BA_unseen 0.185 |
| — | Challenge submission |
