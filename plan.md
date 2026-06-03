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
| bal+mix+clip | MTRCNN | lodo_balanced_mixstyle_clipnorm | running | — | — | — |
| H1 | MTRCNN | lodo_hpss | planned | — | — | — |
| DicL | MTRCNN | lodo_balanced_dicl | running | — | — | — |
| DANN+DicL | MTRCNN | lodo_balanced_dann_dicl | running | — | — | — |
| DicL+SdaL | MTRCNN | lodo_balanced_dicl_sdal | running | — | — | — |
| **FBS-Mix** | MTRCNN | lodo_balanced_fbsmix | **running** | — | — | — |
| **DANN+FBS** | MTRCNN | lodo_balanced_dann_fbsmix | **running** | — | — | — |
| **DANN+DicL+FBS** | MTRCNN | lodo_balanced_dann_dicl_fbsmix | **running** | — | — | — |
| E0 | AST | lodo_ast_base | running | — | — | — |
| E1 | AST | lodo_ast_aug | running | — | — | — |
| E2 | AST | lodo_ast_balanced | running | — | — | — |
| E3 | AST | lodo_ast_balanced_mixstyle | running | — | — | — |
| E4 | AST | lodo_ast_dann | running | — | — | — |

---

## Next Steps

- [ ] Read DicL, FBS-Mix, clipnorm, species_only results when done
- [ ] Read AST results — does global attention generalise better than local conv?
- [ ] Update ensemble_template.json with best members + weights once results land
- [ ] Multi-seed run on winner (gate on FBS-Mix / DicL results vs M2+D1)
- [ ] TENT on eval set using best checkpoint (evaluate_tent.py)
- [ ] Produce submission file (evaluate_ensemble.py or single-model)

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
- AST jobs: 32G mem; `max_eval_frames: 1024` required to avoid OOM on long clips
- Per-clip normalisation: `clip_normalize: true` in config (dataset.py)
- Domain loss weight: `domain_loss_weight: 0.0` for species-only training
- TENT: `evaluate_tent.py --checkpoint <path> --eval-dir data/Evaluation_data/`
- DicL: `dicl_weight: 0.1, dicl_tau: 0.07` — needs `domain_balanced_sampling: true` for cross-domain pairs in batch
- FBS-Mix: `freq_band_mixstyle: true, fbmix_species_lo: 9, fbmix_species_hi: 36` — padding-aware (masks stats on valid frames, restores zeros after mix)
- Ensemble: `evaluate_ensemble.py --members ensemble_template.json --eval-dir data/Evaluation_data/`

---

## Novel Method: FBS-Mix

**Frequency-Band Selective Mix** — data-motivated novel contribution.

Variance ratio analysis (within-domain / cross-domain) on test split:
- Bins 0-8  (~0-500 Hz):    ratio 0.30-0.93 → **domain-dominated** (recording noise floor)
- Bins 9-35 (~500-2.2kHz):  ratio 1.50-4.12 → **species-dominated** (wingbeat core, peak at bin 17 = 1060Hz)
- Bins 36-63 (~2.2-4kHz):   ratio 0.73-1.47 → mixed/unclear

This explains M2+M1 ≈ M2: standard MixStyle mixes ALL bins, corrupting species signal (9-35) while helping domain noise (0-8). Net effect ≈ 0.

FBS-Mix: mix ONLY bins 0-8 (domain), protect bins 9-35 exactly, leave 36-63 unchanged.
Key implementation detail: statistics computed on valid frames only (masked by lengths tensor) — padding zeros would otherwise bias mean/std by 3× for short clips.

---

## Ideas / Future Directions

- Multi-seed on winner (gate on FBS-Mix / DicL results)
- Curriculum domain sampling: anneal from natural imbalance → balanced over training
- Joint (species × domain) balanced sampling
- Source-free domain adaptation on unlabelled eval clips (TENT variant)
- Analyse M2+D1 embedding space — does the t-SNE cluster collapse fix?
- Domain Envelope Subtraction (DES): subtract per-clip smooth spectral envelope (Gaussian smoothed over mel bins) before model — closed-form acoustic domain removal
- FBS-Mix for AST: apply at patch-embed output, split frequency patches into low/high
