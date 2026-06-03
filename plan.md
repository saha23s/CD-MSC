# Research Plan — BioDCASE 2026 CD-MSC

**Challenge:** Cross-Domain Mosquito Species Classification  
**Primary metric:** `species_balanced_accuracy` on unseen domains (LODO BA_unseen)  
**DSG** (`|BA_unseen - BA_seen|`) — lower is better  
**Baseline (10-seed, official partition):** BA_seen=0.881, BA_unseen=0.175, DSG=0.705

---

## Evaluation Protocol

All DG experiments use **LODO (Leave-One-Domain-Out)** as the evaluation:
- Train on 4 domains, validate + test on the held-out domain
- Run all 5 folds (D1–D5), report **mean LODO BA_unseen** as primary metric
- Secondary: mean BA_seen, mean DSG
- Single seed (42) for initial screen; multi-seed only if mean LODO BA_unseen improves >5pp over B1

**B1 LODO baseline (seed 42, corrected BA_seen/BA_unseen):**

| Fold | BA_seen | BA_unseen | DSG   |
|------|---------|-----------|-------|
| D1   | 0.589   | 0.051     | 0.538 |
| D2   | 0.598   | 0.246     | 0.352 |
| D3   | 0.499   | 0.265     | 0.235 |
| D4   | 0.625   | 0.100     | 0.525 |
| D5   | 0.209   | 0.067     | 0.142 |
| **Mean** | **0.504** | **0.146** | **0.358** |

Note: D5 fold collapses because D5 = 99.4% of training data (212,339/213,647 samples).
D1–D4 each have only 80–634 training samples (0.04–0.3%).

---

## Key Insight: D5 Dominance is the Root Cause

Any DG technique applied without addressing training imbalance is fighting D5 overfitting.
Domain-balanced sampling (M2) is the highest-priority structural fix.
MixStyle without balanced sampling mostly mixes D5 with D5 — not useful.

---

## Experiment Priority Queue

### Tier 1 — Structural (highest expected impact)

| ID | Config | Change from B1 | Rationale |
|----|--------|----------------|-----------|
| **T1** | — (no retraining) | Re-eval B1 with `ttbn: true` | TTBN uses batch stats at test time instead of stored D5-biased running stats. Free. |
| **M2** | `lodo_balanced.json` | `domain_balanced_sampling: true` | WeightedRandomSampler equalises domain frequency per epoch. Most direct fix for D5 dominance. |

### Tier 2 — Feature-level DG

| ID | Config | Change from B1 | Rationale |
|----|--------|----------------|-----------|
| **M1** | `lodo_mixstyle.json` | `use_mixstyle: true` | MixStyle mixes instance-norm stats (μ, σ) across samples. Ablation: without balanced sampling. |
| **M2+M1** | `lodo_balanced_mixstyle.json` | `domain_balanced_sampling` + `use_mixstyle` | With balanced sampling, MixStyle mixes across diverse domain statistics. Best expected combo. |

### Tier 3 — Adversarial

| ID | Config | Change from B1 | Rationale |
|----|--------|----------------|-----------|
| **D1** | `lodo_dann.json` | `domain_adversarial: true` | DANN GRL. Ablation: DANN alone (without balanced sampling). |
| **M2+D1** | `lodo_balanced_dann.json` | `domain_balanced_sampling` + `domain_adversarial` | DANN with balanced domains — adversary sees diverse mix. |

### Tier 4 — Augmentation

| ID | Config | Change from B1 | Rationale |
|----|--------|----------------|-----------|
| **B2** | `lodo_aug.json` | SpecAugment + Gaussian noise | Breaks texture patterns that may be domain-specific. Already configured. |
| **H1** | `lodo_hpss.json` | HPSS (p=0.5, α_min=0) | Attenuates percussive recording noise. Already configured. |

### Tier 5 — Architecture (AST, ~1.8M params, LR=5e-4)

| ID | Config | Change from B1 | Matches MTRCNN |
|----|--------|----------------|----------------|
| **E0** | `lodo_ast_base.json` | AST only, no aug | B1 |
| **E1** | `lodo_ast.json` | AST + SpecAugment + noise | B2 |
| **E2** | `lodo_ast_balanced.json` | AST + domain-balanced | M2 |
| **E3** | `lodo_ast_balanced_mixstyle.json` | AST + balanced + MixStyle | M2+M1 |
| **E4** | `lodo_ast_dann.json` | AST + DANN + SpecAugment | D1 |

---

## Experiment Results Table

| ID | Model | Config | Status | LODO BA_unseen | BA_seen | DSG | Job ID |
|----|-------|--------|--------|---------------|---------|-----|--------|
| B1 | MTRCNN | default | done | 0.146 | 0.504 | 0.358 | — |
| T1 | MTRCNN | — (TTBN re-eval) | running | — | — | — | 9728914 |
| M2 | MTRCNN | lodo_balanced | running | — | — | — | 9728915 |
| M1 | MTRCNN | lodo_mixstyle | running | — | — | — | 9728916 |
| M2+M1 | MTRCNN | lodo_balanced_mixstyle | running | — | — | — | 9728917 |
| B2 | MTRCNN | lodo_aug | queued | — | — | — | 9728918 |
| D1 | MTRCNN | lodo_dann | queued | — | — | — | 9728919 |
| M2+D1 | MTRCNN | lodo_balanced_dann | queued | — | — | — | 9728920 |
| H1 | MTRCNN | lodo_hpss | planned | — | — | — | — |
| E0 | AST | lodo_ast_base | queued | — | — | — | 9728962 |
| E1 | AST | lodo_ast | queued | — | — | — | 9728963 |
| E2 | AST | lodo_ast_balanced | queued | — | — | — | 9728964 |
| E3 | AST | lodo_ast_balanced_mixstyle | queued | — | — | — | 9728965 |
| E4 | AST | lodo_ast_dann | queued | — | — | — | 9728966 |

---

## Submission Strategy

1. Gate M2+D1, H1 on M2 results; gate multi-seed on best single-seed winner (>5pp BA_unseen over B1)
2. TTBN (large batch, eval_batch_size=256) is valid for submission — uses test-time BN stats, no labels needed
3. Winner → 10-seed multi-seed run → re-evaluate with official partition (split_summary.json)
4. Submission format: one TXT file, one row per clip, file_id + predicted species ID

---

## Implementation Notes

- All experiments are config-only (no code changes beyond initial fixes)
- `eval_lodo.py --ttbn --eval-batch-size 256` writes to `*_ttbn/` subdirs (B1 checkpoints untouched)
- `outputs/`, `*.pth`, `logs/` are git-ignored (added 2026-06-03)
- BA_seen/BA_unseen now correctly uses held-out domain for LODO (fixed 2026-06-03)
- `experiment_tag` in config → appended to output dir name (prevents collisions across DG variants)
- sbatch scripts in `sbatch/`; AST jobs use 32G mem (attention over long sequences)

---

## Ideas / Future Directions

- Domain-specific normalization: separate feature stats per domain at training time
- Test-time augmentation (TTA): ensemble augmented views at inference
- Contrastive loss: push same-species embeddings together across domains
- Source-free domain adaptation: fine-tune BN layers on unlabelled test domain data
