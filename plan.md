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
- Single seed (42) for initial screen; multi-seed on best winner (>5pp over B1 ✓ met)

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

## Key Findings (MTRCNN, seed 42)

**Domain balance is a prerequisite for all DG methods in this setting.**

| Exp | BA_unseen | Δ vs B1 | BA_seen | DSG | Per-fold (D1/D2/D3/D4/D5) |
|-----|-----------|---------|---------|-----|---------------------------|
| **M2+D1** | **0.316** | **+17pp** | 0.497 | 0.263 | 0.271/0.739/0.328/0.134/0.110 |
| **M2** | **0.267** | **+12pp** | 0.450 | 0.227 | 0.186/0.629/0.343/0.127/0.049 |
| M2+M1 | 0.262 | +11pp | 0.460 | 0.257 | 0.175/0.619/0.318/0.141/0.056 |
| D1 (DANN) | 0.151 | +0.5pp | 0.550 | 0.399 | 0.110/0.211/0.283/0.069/0.080 |
| B1 (base) | 0.146 | — | 0.504 | 0.358 | 0.051/0.246/0.265/0.100/0.067 |
| B2 (aug) | 0.140 | −0.6pp | 0.496 | 0.356 | 0.033/0.247/0.275/0.083/0.062 |
| M1 (mix) | 0.143 | −0.3pp | 0.523 | 0.381 | 0.096/0.158/0.267/0.148/0.046 |
| **T1 (TTBN)** | **0.112** | **−3pp** | 0.262 | 0.151 | 0.162/0.062/0.239/0.059/0.036 |

**Interpretations:**
- **M2+D1 is the current best (+17pp).** DANN+balance are synergistic: balance ensures the adversary sees diverse domains; adversary forces domain-invariant representations. Without balance, DANN is nearly useless (+0.5pp).
- **MixStyle adds nothing on top of M2.** Mixing only 80–634 samples per minority domain produces insufficient style diversity. M2+M1 ≈ M2.
- **SpecAugment (B2) doesn't help DG** — it improves robustness but not domain shift.
- **TTBN is harmful here.** Test set is ~85% D5 → batch norm adapts toward D5, hurting OOD clips. Confirmed negative finding, useful for paper.

---

## Experiment Results Table

| ID | Model | Config | Status | BA_unseen | BA_seen | DSG |
|----|-------|--------|--------|-----------|---------|-----|
| B1 | MTRCNN | default | **done** | 0.146 | 0.504 | 0.358 |
| T1 | MTRCNN | TTBN re-eval | **done** | 0.112 | 0.262 | 0.151 |
| M2 | MTRCNN | lodo_balanced | **done** | 0.267 | 0.450 | 0.227 |
| M1 | MTRCNN | lodo_mixstyle | **done** | 0.143 | 0.523 | 0.381 |
| M2+M1 | MTRCNN | lodo_balanced_mixstyle | **done** | 0.262 | 0.460 | 0.257 |
| B2 | MTRCNN | lodo_aug | **done** | 0.140 | 0.496 | 0.356 |
| D1 | MTRCNN | lodo_dann | **done** | 0.151 | 0.550 | 0.399 |
| M2+D1 | MTRCNN | lodo_balanced_dann | **done** | **0.316** | 0.497 | 0.263 |
| clipnorm | MTRCNN | lodo_clipnorm | running | — | — | — |
| bal+clip | MTRCNN | lodo_balanced_clipnorm | running | — | — | — |
| sp_only | MTRCNN | lodo_species_only | running | — | — | — |
| bal+sp | MTRCNN | lodo_balanced_species_only | running | — | — | — |
| bal+mix+clip | MTRCNN | lodo_balanced_mixstyle_clipnorm | **submitted** | — | — | — |
| H1 | MTRCNN | lodo_hpss | planned | — | — | — |
| E0 | AST | lodo_ast_base | running | — | — | — |
| E1 | AST | lodo_ast_aug | running | — | — | — |
| E2 | AST | lodo_ast_balanced | running | — | — | — |
| E3 | AST | lodo_ast_balanced_mixstyle | running | — | — | — |
| E4 | AST | lodo_ast_dann | running | — | — | — |

---

## Next Steps

- [ ] Read clipnorm, species_only, bal+mix+clip results when done
- [ ] Read AST results — does global attention generalise better than local conv?
- [ ] Multi-seed run on M2+D1 (current best, already >5pp threshold met)
- [ ] TENT on eval set using M2+D1 checkpoint (evaluate_tent.py)
- [ ] Produce submission file from M2+D1 checkpoint
- [ ] Revisit TTBN: try with eval data only (not test set), or larger batches

---

## Submission Strategy

1. **Current best: M2+D1 (0.316 LODO BA_unseen)** → multi-seed → submission
2. TENT on eval clips (`evaluate_tent.py`) applied to M2+D1 checkpoint — may adapt to D1–D4 distribution
3. TTBN confirmed harmful on D5-dominated test set — do NOT use for submission
4. Re-evaluate M2+D1 with official partition (split_summary.json) for challenge metric
5. Submission format: one TXT, one row per clip, file_id + 1-based species ID

---

## Implementation Notes

- `outputs/`, `*.pth`, `logs/`, `sbatch/` are git-ignored
- BA_seen/BA_unseen uses held-out domain for LODO (fixed 2026-06-03)
- `experiment_tag` in config → unique output dir name
- AST jobs: 32G mem (long-sequence attention)
- Per-clip normalisation: `clip_normalize: true` in config (dataset.py)
- Domain loss weight: `domain_loss_weight: 0.0` for species-only training
- TENT: `evaluate_tent.py --checkpoint <path> --eval-dir data/Evaluation_data/`

---

## Ideas / Future Directions

- Multi-seed M2+D1 to get confidence intervals
- Curriculum domain sampling: anneal from natural imbalance → balanced over training
- Joint (species × domain) balanced sampling
- Contrastive loss: push same-species embeddings together across domains
- Source-free domain adaptation on unlabelled eval clips (TENT variant)
- Analyse what the eval set domain distribution looks like after M2+D1 embeddings
