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
- Experiment matrix: B1 (baseline LODO) → B2 (aug), D1 (DANN), E1 (AST), H1 (HPSS) as independent ablations → D2 (AST+DANN) only if D1+E1 both win.
- HPSS implemented (`framework/augmentation.py`): Fitzgerald 2010 Wiener masking on log-mel. Harmonic (horizontal) = wingbeat signal; percussive (vertical) = recording-condition noise. `x_aug = x × (M_h + α×M_p)`, α ~ Uniform(alpha_min, 1.0). kernel_harm=17 (freq), kernel_perc=9 (time — smaller due to ~63 frame clips). scipy.ndimage.median_filter used internally; scipy already present via librosa dep.
- Key insight from baseline analysis: Cx. pipiens is the only species that generalises (BA_unseen=0.804). An. gambiae and Cx. quinquefasciatus collapse to ~0 BA_unseen. D4 is hardest fold (BA=0.118). Avg training runs only 23.6 epochs — fast convergence on D5.
- MixStyle implemented (`framework/mixstyle.py`): mixes (μ, σ) of feature maps between random batch pairs; lam~Beta(alpha,alpha). MTRCNN: one shared instance applied after ConvStage 0 in all 3 branches (consistent perm/lam). AST: after patch_embed [B,D,T_p,F_p] before flatten. Config: `use_mixstyle, mixstyle_alpha (0.1), mixstyle_p (0.5)`.
- Domain-balanced sampling (`framework/utilization.py`): WeightedRandomSampler, weight[i]=1/count(domain[i]). D1-D4 upweighted ~200×. Config: `domain_balanced_sampling: true`.
- TTBN (`framework/engine.py`): after model.eval(), selectively sets BN layers back to training=True so they use batch stats at test time. Applied only to test split (not val loop). Config: `ttbn: true`. T1 = run evaluate.py with ttbn=true on B1 checkpoints — costs nothing.

## 2026-06-03

- LODO BA_seen/BA_unseen bug fixed: `evaluate.py` was using static `split_summary.json` unseen_domain_by_species for all folds — inverted D5 fold metrics entirely. Fixed by passing `lodo_held_out_domain` to `append_official_metrics`, partitioning by domain label directly.
- [B1] Corrected LODO baseline (seed 42): mean BA_unseen=0.146, BA_seen=0.504, DSG=0.358. Per-fold BA_unseen: D1=0.051, D2=0.246, D3=0.265, D4=0.100, D5=0.067.
- D5 = 99.4% of training data (212,339/213,647). D1–D4 have only 80–634 training samples each. Root cause of all DG failures.
- Eval set: 15,573 clips, ~85% D5-like by nearest-centroid in log-mel space. Challenge metric dominated by D5; LODO is the honest DG measure.
- AST bug fixed: `MAX_TIME_PATCHES=128` (1024 frames max) insufficient for LODO val/test clips up to 15k frames. Fixed in `ast_model.py` forward(): generate 2D sinusoidal pos embeddings on-the-fly when clip exceeds precomputed table. Short clips use cached buffer.
- [T1] TTBN harmful: BA_unseen 0.146→0.112 (−3pp). Test set is ~85% D5 → batch norm adapts toward D5 at inference, hurting OOD clips. Confirmed negative result.
- [M2] Domain-balanced sampling: BA_unseen=0.267 (+12pp). Structural fix. WeightedRandomSampler upsamples D1–D4 ~200–2600× to equalise domain frequency per epoch.
- [M1] MixStyle alone: BA_unseen=0.143 (−0.3pp, null). Without balance, mixes D5 with D5 ~99% — useless for DG.
- [M2+M1] Balanced+MixStyle: BA_unseen=0.262 (+11pp). MixStyle adds nothing on top of M2. 80–634 samples per minority domain too few for meaningful style synthesis.
- [D1] DANN alone: BA_unseen=0.151 (+0.5pp). Adversary learns D5 vs not-D5 only; nearly useless without domain diversity.
- [M2+D1] Balanced+DANN: BA_unseen=0.316 (+17pp) — current best. Synergistic: balance provides diverse domain signal for adversary; adversary enforces domain-invariant embeddings. D2: 0.246→0.739, D1: 0.051→0.271.
- [B2] SpecAugment+noise: BA_unseen=0.140 (−0.6pp). Augmentation improves robustness but not domain shift.
- Key research finding: domain balance is prerequisite for DG methods in severely imbalanced settings. DANN/MixStyle effectively disabled without it.
- Per-clip instance normalisation (`framework/dataset.py:clip_instance_normalize`): subtracts per-bin temporal mean and std. Removes recording-level spectral coloring. Config: `clip_normalize: true`.
- TENT implemented (`framework/tent.py`): freeze all, unfreeze BN/LN affine (γ, β), minimise prediction entropy on unlabelled eval clips. MTRCNN: 10 BN layers, 800 params. AST: 9 LN layers, 3456 params. Script: `evaluate_tent.py`.
- `domain_loss_weight` added to `train_one_epoch()` (default 1.0). Set 0.0 for species-only ablation — tests whether domain supervision hurts under severe imbalance.
- t-SNE (logit-space, 9-dim) of B1 embeddings: D1–D4 collapse to isolated corner, completely separate from D5 cloud. Model routes all OOD samples to same region — structured wrong predictions, not uncertainty.
- `experiment_tag` in config appended to output dir name to prevent collisions across DG variants sharing same hyperparams.

## 2026-06-03 (session 2)

**D5 exclusion finding:** D5 = 99.4% of training data (212,339/213,647 samples). Holding it out leaves only 1,308 training samples — data starvation, not domain shift. All LODO means must exclude D5. D5 test set = 84.4% of total test set, so standard-split BA_unseen=0.175 is also D5-dominated and uninformative. LODO D1–D4 mean is the honest metric.

**Current best (LODO D1–D4 mean, seed 42):** balanced_dann_fbsmix: BA_unseen=0.385, DSG=0.212 (3/4 folds — D1 rerunning job 9730635). balanced_dann: BA_unseen=0.368, DSG=0.276 (4/4, confirmed best complete result).

**DiCL diagnosis:** DiCL fires on 100% of batches with ~200 cross-domain positive pairs — not a hit-rate problem. Likely redundant with DANN (both push domain invariance from opposite directions on a 32-dim embedding). Projection head ablation (32→128 dim, τ=0.07 and τ=0.2) submitted as jobs 9730937–38 to test geometry hypothesis.

**Physics-grounded wingbeat features implemented:**
- Spectral centroid of mel bins 14–23 (400–700 Hz) computed via softmax-weighted bin expectation over valid frames — amplitude-invariant, domain-invariant by construction
- Concatenated to CNN output (192→193 dim) before embedding layer
- Auxiliary regression head predicts normalised wingbeat frequency (0–1 over 400–700 Hz) from per-species literature values (Mukundarajan 2017, Brogdon 1994, Kiskin 2020 HumBugDB)
- `use_wingbeat_feature: true` + `wingbeat_weight: 0.5` in config

**New model features added:**
- `embed_dim` configurable (default 32, backwards-compatible) — affects all downstream layers
- `GroupDROState` (framework/group_dro.py): per-domain exponential loss reweighting, `group_dro_eta` config key
- `species_balanced_loss: true` — inverse-frequency CE weighting computed from training split
- `contrastive_proj_dim: 128` — 2-layer MLP projection head (32→128→128) for DiCL/SdaL
- TENT on LODO: `eval_lodo.py --tent-steps N` — per-batch entropy minimisation of BN/LN affine params

**Domain-specific gap identified:** Current model treats this as generic audio classification. Mosquito wingbeats = narrowband harmonic signal; microphone response changes spectral envelope but NOT harmonic spacing. Key interventions submitted:
- HPSS p=1.0 (job 9731297): separates harmonic wingbeat from percussive background
- clip_normalize (job 9731298): per-bin CMVN removes microphone frequency response
- HPSS + clip_normalize (job 9731299): both combined

**Paper ablation experiments submitted (2026-06-03):**
- Multi-seed (1234, 3407) for baseline/balanced/balanced_dann/balanced_dann_fbsmix: jobs 9731135–42
- DANN lambda sweep (0.5, 2.0; 1.0 already done): jobs 9731143–44
- balanced_mixstyle: job 9731145
- GroupDRO (η=0.01): job 9731174
- embed_dim=64,128: jobs 9731175–76
- SpecAugment (time+freq masking): job 9731177
- Species-balanced loss: job 9731178
- HPSS/clip_normalize: jobs 9731297–99

## 2026-06-03 (continued — session 1)

- DR-BioL (Hou et al., arXiv:2510.00346) reviewed: same mosquito problem, introduces DicL (domain-invariant contrastive loss — positive pairs = same species, different domain) and SdaL (species-conditional MMD). Their DANN ablation hurts by -0.45% — consistent with our D1 result (+0.5pp). DicL is conceptually stronger than DANN: explicit pull vs adversarial push.
- DicL implemented (`framework/losses.py`): τ=0.07 (vs paper's 0.01 — more stable with small minority batches). Returns 0 gracefully if no cross-domain pairs in batch. Must combine with domain_balanced_sampling.
- SdaL implemented: per-species RBF-kernel MMD², skips species-domain subsets with <2 samples.
- Both models now expose `"embedding"` in forward() output dict (MTRCNN: 32-dim; AST: embed_dim CLS token). Fixes the t-SNE hook issue.
- AST OOM fix: `max_eval_frames: 1024` centre-crop at eval time. 15k-frame clips → 15k tokens → 225M attention elements per layer → OOM. Fix: centre-crop to 1024 frames (128 patches × 8). Set in all AST configs.
- Eval set analysis: 15,573 clips, ~85% D5-like by nearest-centroid in log-mel space. Challenge metric dominated by D5; LODO is the honest DG measure. Eval set is a valid target for test-time adaptation (TENT, domain envelope removal).
- Frequency-band variance ratio analysis on test split: within-domain/cross-domain ratio per mel bin reveals:
  - Bins 0-8 (0-500 Hz): ratio 0.30-0.93 → domain-dominated (recording noise floor)
  - Bins 9-35 (500-2.2 kHz): ratio 1.50-4.12 → species-dominated (wingbeat core; peak at bin 17 = 1060 Hz)
  - Bins 36-63 (2.2-4 kHz): ratio 0.73-1.47 → mixed
  - This U-shape explains M2+M1 ≈ M2: MixStyle mixes ALL bins, corrupting the wingbeat core while helping domain noise — effects cancel.
- FBS-Mix implemented (`framework/augmentation.py:fbs_mix_batch`): input-level batch transform, mixes bins 0-8 only via instance-stat interpolation (Beta λ~B(0.1,0.1)), protects bins 9-35 exactly, leaves 36-63 unchanged.
- FBS-Mix padding fix: statistics computed on valid frames only using lengths mask. Without this, a 63-frame clip padded to 200 has mean biased 3× toward zero (68% zero padding). Padded positions restored to zero after mixing.
- 3 FBS-Mix experiments submitted (jobs 9729812-9729814): balanced_fbsmix (vs M2+M1), balanced_dann_fbsmix (vs M2+D1 current best 0.316), balanced_dann_dicl_fbsmix (full combo).
- Ensemble inference script: `evaluate_ensemble.py` — weighted softmax averaging from JSON spec, supports both submission (WAV dir) and eval (feature pickle) modes.
- Experiment timing (single LODO fold, 1 GPU, early stop min10/pati5): MTRCNN baseline ~27 min (14 epochs, ~1:50/epoch); MTRCNN balanced+DANN+DicL ~23 min (15 epochs, ~1:30/epoch); AST base ~1h 43 min (14 epochs, ~7.3 min/epoch). AST is ~4× slower per epoch than MTRCNN.
- Scale-up: MTRCNN — all 5 folds × 1 seed sequential ~2.5h; 5 seeds × 5 folds sequential ~12h (or ~2.5h with 5 parallel SLURM jobs). AST — all 5 folds × 1 seed ~8-10h (overnight); 5 seeds × 5 folds ~10h with 5 parallel jobs. AST multi-seed sweeps only feasible for selected configs, not full ablation matrix.

