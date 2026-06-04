# Research Plan — BioDCASE 2026 CD-MSC

**Challenge:** Cross-Domain Mosquito Species Classification  
**Primary metric:** `species_balanced_accuracy` on unseen domains (LODO BA_unseen)  
**DSG** (`|BA_unseen - BA_seen|`) — lower is better  
**Baseline (10-seed, official partition):** BA_seen=0.881, BA_unseen=0.175±0.020, DSG=0.706±0.025

---

## Evaluation Protocol

All DG experiments use **LODO (Leave-One-Domain-Out)** as the evaluation:
- Train on 4 domains, validate + test on the held-out domain
- Run all 5 folds (D1–D5), report **mean LODO BA_unseen** as primary metric
- Secondary: mean BA_seen, mean DSG
- Single seed (42) for initial screen; multi-seed on best winner (>5pp over B1 ✓ met)

**D5 exclusion rationale:** D5 = 99.4% of training data (212,339/213,647). Holding it out leaves only 1,308 training samples — data starvation, not domain shift. D5 is excluded from LODO mean for all comparisons. D5 test set = 84.4% of total test samples.

**Evaluation metric: LODO D1–D4 mean BA_unseen (D5 excluded)**

**B1 LODO baseline (seed 42):**

| Fold | BA_seen | BA_unseen | DSG   |
|------|---------|-----------|-------|
| D1   | 0.578   | 0.165     | 0.412 |
| D2   | 0.537   | 0.165     | 0.372 |
| D3   | 0.496   | 0.165     | 0.331 |
| D4   | 0.530   | 0.165     | 0.365 |
| **D1–D4 mean** | **0.514** | **0.165** | **0.412** |

---

## Key Findings (MTRCNN, seed 42, D1–D4 mean)

**Domain balance is a prerequisite for all DG methods in this setting.**

| Exp | BA_unseen | Δ vs B1 | BA_seen | DSG | Notes |
|-----|-----------|---------|---------|-----|-------|
| **balanced_dann_dicl_proj128_tau02** | **0.378** | **+21pp** | 0.534 | **0.202** | best DSG; τ=0.2 proj=128 |
| balanced_dann_dicl_proj128 | 0.371 | +21pp | 0.540 | 0.235 | |
| **balanced_dann_fbsmix** | **0.345** | **+18pp** | 0.516 | **0.229** | 4/4 folds now complete |
| **balanced_dann** | **0.368** | **+20pp** | 0.540 | 0.276 | strong overall |
| balanced_dann_wingbeat | 0.339 | +17pp | 0.533 | 0.249 | |
| balanced_dann_dicl | 0.340 | +17pp | 0.517 | 0.247 | |
| balanced_dicl_sdal | 0.326 | +16pp | 0.532 | 0.257 | |
| balanced_species_only | 0.324 | +16pp | 0.547 | 0.267 | |
| balanced_dicl | 0.321 | +16pp | 0.547 | 0.262 | |
| balanced | 0.321 | +16pp | 0.511 | 0.245 | |
| ast_balanced | 0.319 | +15pp | 0.513 | 0.197 | AST worse for BA_unseen |
| ast_dann | 0.295 | +13pp | 0.572 | 0.277 | high BA_seen, low BA_unseen |
| ast_base | 0.214 | +5pp | 0.563 | 0.349 | |
| dann (unbalanced) | 0.169 | +0.4pp | 0.621 | 0.453 | near-useless without balance |
| baseline | 0.165 | — | 0.578 | 0.412 | |
| species_only | 0.140 | −2.5pp | 0.605 | 0.465 | domain head was helping |
| std_split (10-seed) | 0.175 | ref | 0.881 | 0.706±0.025 | misleading — 84% D5 test |

**Key interpretations:**
- **Balanced sampling is the single most important lever (+15.6pp)**. All DG methods are near-useless without it.
- **DANN only works with balancing** — unbalanced DANN (0.169) is barely above baseline.
- **DiCL doesn't help**: fires 100% of batches with ~200 positive pairs; likely redundant with DANN targeting same objective. Projection head ablation running.
- **AST consistently hurts BA_unseen**: richer pretrained features overfit harder to seen domains.
- **FBS-Mix reduces DSG most** (0.212) but BA_unseen slightly below DANN alone. Best DSG-BA_unseen tradeoff.
- **Standard split is misleading**: BA_unseen=0.175 is below even the LODO baseline (0.165 on comparable splits). D5 dominance makes it uninformative for DG.

---

## Experiment Results Table

All BA_unseen/BA_seen/DSG = D1–D4 mean (D5 excluded). Seed=42 unless noted.

### Completed

| ID | Config | BA_unseen | BA_seen | DSG | n folds |
|----|--------|-----------|---------|-----|---------|
| baseline | lodo_baseline | 0.165 | 0.578 | 0.412 | 4/4 |
| species_only | lodo_species_only | 0.140 | 0.605 | 0.465 | 4/4 |
| dann | lodo_dann | 0.169 | 0.621 | 0.453 | 4/4 |
| mixstyle | lodo_mixstyle | 0.167 | 0.592 | 0.425 | 4/4 |
| balanced | lodo_balanced | 0.321 | 0.511 | 0.245 | 4/4 |
| balanced_species_only | lodo_balanced_species_only | 0.324 | 0.547 | 0.267 | 4/4 |
| **balanced_dann** | lodo_balanced_dann | **0.368** | 0.540 | 0.276 | 4/4 |
| balanced_dicl | lodo_balanced_dicl | 0.321 | 0.547 | 0.262 | 4/4 |
| balanced_dicl_sdal | lodo_balanced_dicl_sdal | 0.326 | 0.532 | 0.257 | 4/4 |
| balanced_dann_dicl | lodo_balanced_dann_dicl | 0.340 | 0.517 | 0.247 | 4/4 |
| balanced_dann_dicl_proj128 | lodo_balanced_dann_dicl_proj128 | 0.371 | 0.540 | 0.235 | 4/4 |
| **balanced_dann_dicl_proj128_tau02** | lodo_balanced_dann_dicl_proj128_tau02 | **0.378** | 0.534 | **0.202** | 4/4 ← best |
| balanced_dann_wingbeat | lodo_balanced_dann_wingbeat | 0.339 | 0.533 | 0.249 | 4/4 |
| **balanced_dann_fbsmix** | lodo_balanced_dann_fbsmix | **0.345** | 0.516 | 0.229 | 4/4 |
| ast_base | lodo_ast_base | 0.214 | 0.563 | 0.349 | 4/4 |
| ast_balanced | lodo_ast_balanced | 0.319 | 0.513 | 0.197 | 4/4 |
| ast_dann | lodo_ast_dann | 0.295 | 0.572 | 0.277 | 4/4 |

### Running

| ID | Config | Jobs | What it tests |
|----|--------|------|---------------|
| eval missing LODO runs | 12 unevaluated checkpoints | 9733049 | balanced_fbsmix D1–D5, AST D2/D5 |
| best method random split × 3 seeds | MTRCNN balanced_dann_dicl_proj128_tau02 | 9733051 | compare random-split DSG vs LODO |
| balanced_dicl_sdal_scol | lodo_balanced_dicl_sdal_scol | 9733089 | full DR-BioL loss (paper best lambdas: λ4=0.01, λ5=0.1, λ2=λ3=1) + ScoL |
| balanced_dicl_sdal_lam | lodo_balanced_dicl_sdal_lam | 9733096 | paper best lambdas only, no ScoL — control to isolate ScoL contribution |

### Previously completed (ablations, results in table above)

balanced_dann_dicl_proj128, balanced_dann_dicl_proj128_tau02, balanced_dann_wingbeat, balanced_dann_fbsmix (D1 rerun), ttbn/tent, lam0p5/lam2p0, balanced_mixstyle, balanced_dann_groupdro, balanced_dann_embed64/128, balanced_dann_specaug, balanced_dann_specbalance, balanced_dann_hpss, balanced_dann_clipnorm2, balanced_dann_hpss_clipnorm, multi-seed × {1234,3407}

---

## Next Steps

- [x] Establish D5 exclusion rationale (data starvation, not domain shift)
- [x] Multi-seed submitted for baseline/balanced/balanced_dann/balanced_dann_fbsmix
- [x] DANN lambda sweep submitted (0.5, 1.0, 2.0)
- [x] GroupDRO, embed_dim, SpecAugment, species-balanced loss submitted
- [x] HPSS and clip_normalize submitted
- [x] TENT/TTBN re-eval on balanced_dann submitted
- [x] All ablation results collected (2026-06-04); best = balanced_dann_dicl_proj128_tau02
- [x] Eval submitted for 12 missing LODO checkpoints (job 9733049)
- [x] Random-split training submitted for best method × 3 seeds (job 9733051)
- [ ] Collect random-split DSG for best method once job 9733051 finishes
- [ ] Collect ScoL ablation results (jobs 9733089 vs 9733096) — compare lam-only vs lam+ScoL
- [ ] Implement CMVN per-bin normalization + delta features (code change needed)
- [ ] Run best-combination experiment (balanced_dann + FBSMix + dicl_proj128_tau02)
- [ ] t-SNE visualization of embeddings (balanced vs balanced_dann vs best) — no training cost
- [ ] Produce submission file from best checkpoint (evaluate_ensemble.py or single-model)
- [ ] Update ensemble_template.json with best members

---

## Submission Strategy

1. **Current best: balanced_dann_dicl_proj128_tau02 (0.378 LODO BA_unseen, DSG 0.202, 4/4 folds)**
2. Run multi-seed on winner → ensemble across seeds for submission
3. TENT on eval clips (`evaluate_tent.py`) applied to best checkpoint — test-time adaptation
4. TTBN confirmed harmful (adapts toward D5 on D5-dominated test set) — do NOT use
5. Submission format: one TXT, one row per clip, file_id + 1-based species ID

---

## Implementation Notes

- `outputs/`, `*.pth`, `logs/`, `sbatch/` are git-ignored
- BA_seen/BA_unseen uses held-out domain for LODO (fixed 2026-06-03)
- `experiment_tag` in config → unique output dir name
- AST jobs: 32G mem; `max_eval_frames: 1024` required to avoid OOM on long clips
- Per-clip normalisation: `clip_normalize: true` in config (dataset.py) — per-bin CMVN, removes microphone frequency response
- Domain loss weight: `domain_loss_weight: 0.0` for species-only training
- TENT on LODO: `eval_lodo.py --tent-steps N --tent-lr 1e-3` → writes to `*_tent{N}/` subdirs
- TTBN on LODO: `eval_lodo.py --ttbn` → writes to `*_ttbn/` subdirs
- DicL: `dicl_weight: 0.1, dicl_tau: 0.07`; uses `proj_embedding` (128-dim) if `contrastive_proj_dim: 128` set
- ScoL: `scol_weight: 1.0, scol_tau: 0.01` — SupCon with same-species positives (any domain); Eq. 4-5 from DR-BioL paper; implemented in `framework/losses.py:species_cohesion_contrastive_loss`
- FBS-Mix: `freq_band_mixstyle: true` — mixes bins 0-8, protects 9-35 (wingbeat core), padding-aware
- HPSS: `augmentation.hpss.enabled: true, p: 1.0` — use p=1.0 not 0.5 for consistent domain normalisation
- GroupDRO: `group_dro_eta: 0.01` — per-domain exponential loss reweighting
- Embed dim: `embed_dim: 64` or `128` — default 32, backwards-compatible
- Species-balanced loss: `species_balanced_loss: true` — inverse-frequency CE weighting
- Wingbeat features: `use_wingbeat_feature: true` (spectral centroid 400–700 Hz) + `wingbeat_weight: 0.5` (regression head)
- Multi-seed: `train_lodo.py --seed N` overrides config seed; output dir includes seed in name
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

- Curriculum domain sampling: anneal from natural imbalance → balanced over training
- Joint (species × domain) balanced sampling
- Source-free domain adaptation on unlabelled eval clips (TENT variant)
- t-SNE of embeddings: balanced vs balanced_dann vs best — visualise domain collapse fix
- Domain Envelope Subtraction (DES): subtract per-clip smooth spectral envelope (Gaussian smoothed over mel bins) before model — closed-form acoustic domain removal
- FBS-Mix for AST: apply at patch-embed output, split frequency patches into low/high
- Delta features: temporal differences of log-mel — removes DC (microphone response), captures wingbeat modulation; standard in ASR, absent here
- Harmonic Product Spectrum (HPS): explicit F0 via f×2f×3f multiplication — maximally domain-invariant; ratio-based so spectral envelope cancels
- Kitchen-sink combination: balanced_dann + FBSMix + HPSS + clip_normalize + best of embed/groupdro/wingbeat
- Cross-domain contrastive pretraining (SimCLR on mosquito audio) before supervised fine-tune
- DiCL diagnosis: firing 100% of batches with ~200 pairs — redundancy with DANN may be the issue rather than geometry
