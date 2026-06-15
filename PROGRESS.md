# CD-MSC Project Progress & Technical Notes

Branch: `aaron/preprocessing` | Collaborator: saha23s | Deadline: passed 2026-06-15

---

## Environment & Setup

- Repo: shared fork `saha23s/CD-MSC`, working branch `aaron/preprocessing`
- Collaborator saha23s has branch `feature/lodo-ast-augmentation` (AST — did not improve BAunseen, see below)
- Local venv set up; `CLAUDE.md` committed with full architecture docs
- Evaluation set released 2026-06-01 on Zenodo (link in README.md line 9) — not yet downloaded

---

## Data

- 271K clips, 9 species, 5 domains (D1–D5)
- Raw audio: `Development_data/raw_audio/` (local + Drive, gitignored)
- Metadata split files: `Development_data/metadata/` (committed to repo)
- Features: `Development_data/feature/*.pkl` (~5 GB, backed up to `MyDrive/CD-MSC-feature`)

### Domain Distribution in Training

| Domain | Training samples | % of training |
|--------|-----------------|---------------|
| D5 (lab) | 212,339 | **99.4%** |
| D1 | 634 | 0.30% |
| D3 | 364 | 0.17% |
| D2 | 230 | 0.11% |
| D4 | 80 | 0.04% |

D5 dominance is not a design choice — it reflects the nature of the data. D5 is a large
controlled lab study; D1–D4 are scarce real-world field recordings.

### Per-Species Unseen Domain Assignments (from split_summary.json)

| Species | Unseen domain | Test clips | Notes |
|---------|--------------|-----------|-------|
| Ae. aegypti | D3 | 8,159 | Has seen clips too |
| Ae. albopictus | D2 | 1,852 | Has seen clips too |
| Cx. quinquefasciatus | D1 | 7,206 | Has seen clips too |
| An. gambiae | D1 | 4,700 | Has seen clips too |
| An. arabiensis | D1 | 2,112 | Has seen clips too |
| **An. dirus** | **D4** | **40** | ALL 40 test clips are D4 — ALL unseen; only 127 clips total |
| Cx. pipiens | D3 | 2,975 | Has seen clips too |
| **An. minimus** | **D2** | **99** | ALL 99 test clips are D2 — ALL unseen; only 550 clips total |
| **An. stephensi** | **D4** | **74** | ALL 74 test clips are D4 — ALL unseen; only 674 clips total |

**The three bold species determine BAunseen.** They have zero training data in their
unseen test domain. Any improvement to BAunseen must come from these three species
generalising from D5 lab recordings to field conditions they have never seen.

### How BAunseen and DSG Are Calculated

Each species has one designated unseen domain. Test clips are partitioned:
- **unseen**: clip's domain matches that species' unseen domain
- **seen**: all other test clips

```
BAunseen = mean per-class recall across all 9 species, evaluated only on unseen clips
BAseen   = mean per-class recall, evaluated only on seen clips
           (An. dirus, An. minimus, An. stephensi excluded — all their test clips are unseen)
DSG      = |BAunseen − BAseen|   (lower = better generalisation)
```

Ceiling analysis: if An. dirus, An. minimus, and An. stephensi all score 0% recall
(no unseen-domain training data), they drag BAunseen down by 3/9 of the average.
The other 6 species must average ~39% recall in their unseen domains to produce
BAunseen = 0.2626 (our current best).

---

## Baseline Results (released, 10-seed mean ± std)

| Metric | Test |
|--------|------|
| BAseen | 0.8806 ± 0.01 |
| BAunseen | 0.1751 ± 0.02 |
| DSG | 0.7055 ± 0.02 |

Seed 42 single run: BAseen=0.883, BAunseen=0.168, DSG=0.716

---

## All Experiments (chronological)

| ID | Config | BAseen | BAunseen | DSG | Verdict |
|----|--------|--------|----------|-----|---------|
| Baseline (10-seed) | Released baseline | 0.8806 | 0.1751 | 0.7055 | Reference |
| Exp 1 | DANN α=1.0 | — | — | — | ❌ Collapsed ep13; α too high |
| Exp 2 | DANN α=0.3 | 0.8769 | 0.2225 | 0.6544 | ✅ +27% BAunseen |
| Exp 3 | C-DANN α=0.3 | 0.8879 | 0.1865 | 0.7014 | ⚠️ Worse than Exp 2; no balancing |
| **Exp 4** | **C-DANN α=0.3 + balanced batches** | **0.7950** | **0.2626** | **0.5324** | **✅ BEST — benchmark to beat** |
| Exp 5 | Exp 4 + B128 + SpecAugment | 0.7227 | 0.2308 | 0.4919 | ⚠️ Worse; SpecAug hurt rare clips |
| Exp 6 | Exp 4 + CMN + D5noise(0.1) | ~0.79 | 0.2452 | ~0.54 | ⚠️ Marginal; preprocessing not helpful |
| SupCon-A | SupCon 1.0, no DANN, sxd-balanced | — | 0.2008 | — | ❌ Worse; D5-only positives useless for bottleneck species |
| SupCon-B | Exp 4 + SupCon 1.0 | ~0.80 | 0.2436 | 0.5538 | ⚠️ Below Exp 4 |
| SupCon-C | Exp 4 + SupCon 0.5 + freqshift±3 | ~0.72 | 0.2303 | 0.4869 | ❌ Confounded; freqshift collapsed BAseen |
| HM-1 | Exp 4 + approx HPSS + hist_match | — | — | — | ❌ Collapsed; domain stats bug (wrong source distribution) |
| HM-2 | Exp 4 + hist_match only | 0.3030 | 0.2495 | 0.0534 | ❌ BAseen collapsed; D5 train/test distribution mismatch |
| AttnPool | Exp 4 + learned frame attention pool | 0.7085 | 0.2148 | 0.4936 | ❌ Worse than Exp 4; max pooling was useful; spurious early stop at ep20 |
| FDA | Exp 4 + Fourier Domain Adaptation (β=0.05, p=0.5) | 0.7983 | 0.2490 | 0.5493 | ❌ Worse than Exp 4; val_BA=0.838 (new high) but BAunseen fell; val_BA ≠ BAunseen |

BAseen for SupCon-B/C derived from DSG (BAseen = BAunseen + DSG); BAseen for SupCon-A not recorded.

### Why Each Approach Failed

**Exp 1 (DANN α=1.0):** Ganin schedule reaches λ≈0.57 by epoch 13. With D5 at 99.4%
of batches, the domain gradient is large and mostly "D5 vs rest." Negated at α=1.0,
it overwhelmed species learning. Both metrics collapsed to random chance simultaneously.

**Exp 3 (C-DANN without balancing):** Without balanced batches, nearly every batch is
all-D5. The domain discriminator only sees D5 vs. tiny D1–D4 representation, producing
a weak and noisy adversarial signal. Balancing (Exp 4) fixed this.

**Exp 5 (SpecAugment):** Time masks up to 40 frames and freq masks up to 10/64 mel bins
are too aggressive for heavily oversampled rare clips (An. dirus has only ~80 D4 training
clips; masking large portions of each destroys the species signal). SpecAugment also
caused BAseen to fall more than BAunseen, suggesting it hurt D5 performance without
helping field generalisation.

**SupCon-A/B/C (Supervised Contrastive):** SupCon pulls same-species embeddings together.
But for An. dirus, An. minimus, An. stephensi — the species that determine BAunseen —
all in-batch positives are D5 vs D5. SupCon tightens D5 clusters for these species but
has no mechanism for cross-domain generalisation (no cross-domain positives exist for them).
At weight 1.0 it actively hurt (0.2436 vs 0.2626) by competing with C-DANN's domain-invariance
gradient in the 32-dim embedding. SupCon-C was additionally confounded by freqshift±3 bins,
which randomly displaced species-discriminative mel bins (~30 Hz at wingbeat frequencies,
enough to shift An. arabiensis toward An. gambiae).

**HM-1 (HPSS + hist_match):** domain_feature_stats.json was computed from raw features,
but HPSS was applied before hist_match in __getitem__, so hist_match received HPSS-filtered
features as source and raw stats as reference — wrong distribution. Val loss rose from
5.5 → 7.5; training abandoned.

**HM-2 (hist_match only):** Transforms D5 training clips to match D1–D4 statistics. But
at test time D5 clips arrive untransformed — the model trains on fake-field D5 but is tested
on real-lab D5. BAseen collapsed to 0.303. DSG was low (0.053) only because both seen and
unseen performance were equally poor.

**AttnPool (frame attention pooling):** Replaces `masked_mean_max` (mean+max concatenation,
128-dim per branch) with learned soft-weighted average (64-dim per branch). Two failures:
(a) the max-pooled component captures the single highest-energy wingbeat frame and is
discriminative; soft averaging over all frames dilutes this. (b) The D1–D4 field BA early
stopping criterion spuriously saved a checkpoint at epoch 20 (D4's 3 val clips all got lucky),
approximately 9 epochs before the val species BA peak. These issues are separable — attention
pooling itself might not be dead, but it needs a fair comparison without the spurious stopping.

**AST (collaborator saha23s, LODO branch):** Pretrained on AudioSet which contains no
mosquito wingbeats — the pretrained representations have no spectral resolution for 20-80 Hz
frequency differences between species. AST representations may also encode field background
textures (AudioSet contains wind, traffic, crowds) and use them as domain signal rather than
suppressing them. Additionally, fine-tuning ~87M parameters on datasets with only 127 An.
dirus clips is catastrophic forgetting territory.

**FDA (Fourier Domain Adaptation, β=0.05):** val_BA reached 0.838 — highest observed — but
BAunseen fell to 0.2490 (below Exp 4's 0.2626). Two mechanisms likely interfered: (a) the
val set is ~99% D5, so any improvement to D5 performance boosts val_BA without helping field
generalisation — val_BA is not a reliable proxy for BAunseen; (b) FDA modifies the D5 training
distribution, but C-DANN's adversarial gradient is calibrated for a natural D5-heavy distribution;
adding synthetic "hybrid" clips makes that gradient noisier without creating clips that truly
resemble test-time field recordings. Result: model adapts to FDA artifacts rather than real
domain shift. A lower β (e.g. 0.01) or applying FDA without DANN might be fairer comparisons.

---

## What's Implemented (committed to aaron/preprocessing)

| Feature | Config flag | Notes |
|---------|------------|-------|
| DANN / C-DANN | `dann_alpha_max`, `cdann` | GRL + Ganin schedule; C-DANN conditions domain head on species one-hot |
| Batch balancing | `batch_balance_domain`, `balance_mode` | WeightedRandomSampler; "domain" or "species_domain" |
| SpecAugment | `spec_augment` | Off at eval; configurable mask sizes |
| CMN | `cmn` | Applied at both train and eval |
| D5 Gaussian noise | `d5_noise_std` | Train only, D5 clips only |
| Delta features | `use_delta` | Appends time-delta to mel; model_n_mels doubles to 128 |
| Freq shift | `freq_shift_bins` | Train only, D5 clips only; ±N mel bin roll |
| Approx HPSS | `use_approx_hpss` | Wiener masking via median filter; applied at both train and eval |
| Hist match | `hist_match` | D5 train clips only; requires domain_feature_stats.json on Drive |
| SupCon | `supcon_weight`, `supcon_temperature` | Projection head 32→64→128, L2-normalised |
| **Frame attention pool** | **`use_attention_pool`** | **Replaces masked_mean_max in each branch; 195 params total** |

Training notebook: `colab_dann.ipynb` — all flags exposed as Python variables in cell 6.

### Bugs Fixed (2026-06-15)

1. **evaluate.py**: `use_approx_hpss` was not passed to dataset in `evaluate_checkpoint` —
   HPSS models would have been evaluated on raw features (train/test feature mismatch).
   No recorded results were affected (only the collapsed HM-1 used HPSS).

2. **train.py**: Early stopping log message said "best validation species_balanced_accuracy"
   but the actual criterion is mean D1–D4 field BA in the validation set. Checkpoint metadata
   now correctly says `"selection_metric": "mean_field_domain_BA"`.

3. **colab_dann.ipynb**: `balance_mode` was never written to cfg (silently defaulted to "domain"
   — correct for all experiments so far, but wrong if "species_domain" was intended). Fixed.
   Output dir print was hardcoded with stale `earlystop_min10_pati5`; now reads from cfg.

### Known Limitation: Early Stopping Is Noisy

The val set is ~12.5% of trainval, which is itself ~99.4% D5. So D1–D4 validation clips
number only ~180 total across 4 domains × 9 species — roughly 5 clips per bucket. The
mean D1–D4 field BA used for early stopping is extremely noisy at this scale; a single
lucky epoch can spike to ~0.99 (observed in HM-2 ep30 and AttnPool ep20 where checkpoint
was saved ~9 epochs before val species BA peak). **Reverted to `val_species_balanced_accuracy`
for all future experiments.** The field-domain BA criterion is correct in principle but
requires far more D4 val clips to be reliable.

---

## AttnPool Result and Analysis (2026-06-15)

**Result:** BAseen=0.7085, BAunseen=0.2148, DSG=0.4936 (best model, seed42)
Final model (ep30): BAseen=0.7389, BAunseen=0.2061, DSG=0.5328

**Verdict:** Worse than Exp 4 (BAunseen 0.2148 vs 0.2626). Two issues:

1. **Max pooling was doing real work.** `masked_mean_max` concatenates mean+max into 128-dim
   per branch; the max component captures the single most discriminative frame (peak wingbeat
   energy). Replacing it with a soft-weighted average loses the max signal. The learned
   attention gate doesn't automatically recover max behaviour — it tends toward a smoother
   weighted mean.

2. **Spurious early stopping at epoch 20.** The mean D1–D4 field BA spiked to 0.979861
   at epoch 20 (caused by 3 D4 val clips → one good batch = D4 BA = 1.0). Checkpoint saved
   at epoch 20. But val species BA was still rising: ep20=0.783, ep23=0.799, ep26=0.804,
   ep29=0.809. We saved a checkpoint ~9 epochs before the actual peak.

**Action: revert early stopping criterion to `val_species_balanced_accuracy`** (the original
baseline criterion). The field-domain BA criterion is correct in principle but unusable with
3 D4 val clips. Reverting to val species BA also aligns the checkpoint with what the final
submission optimises (BAunseen averaged across species). See also: Known Limitation section.

---

## FDA Result and Analysis (2026-06-15)

**Result:** best checkpoint (ep29): BAseen=0.7983, BAunseen=**0.2490**, DSG=0.5493.
Final checkpoint (ep39): BAseen=0.7791, BAunseen=0.2372, DSG=0.5419.
Early stopping fired at ep39 (best ep=29, patience=10, min_epoch=20).

**Verdict: ❌ Worse than Exp 4** (BAunseen 0.2490 < 0.2626). FDA hurt.

**Key diagnostic — val_BA ≠ BAunseen:** val_BA peaked at **0.838** (ep29) — the highest
val_BA observed in any experiment. But BAunseen fell relative to Exp 4. This decoupling is
fundamental: the validation set is ~99% D5. A model that improves D5 performance always
increases val_BA, regardless of whether field generalisation improved or worsened. Val_BA is
not a reliable proxy for BAunseen. We are essentially blind to domain-shift progress during
training unless we use the test set (which we must not optimise against).

**Why FDA hurt on top of C-DANN:**
- C-DANN's adversarial gradient is calibrated for the natural D5-heavy training distribution.
  FDA adds synthetic "hybrid" clips with swapped Fourier amplitude; these clips are neither
  clean D5 nor real field recordings, so DANN's gradient is optimising against a distribution
  that doesn't exist at test time.
- The model learns to handle FDA artifacts rather than generalising across real domain shift.
- FDA without DANN (plain cross-entropy + FDA) or with smaller β (0.01) might be a fairer
  test of the augmentation idea in isolation.

---

## Next Steps (priority order, updated 2026-06-15)

1. **Exp B — D1-heavy batch weighting (immediate, ~50 min):** ✅ Implemented.
   `d1_oversample=3.0` in `train.py` sampler; exposed as `D1_OVERSAMPLE` in colab_dann.ipynb cell 6.
   Output dir: `MTRCNN_seed42_B64_E100_earlystop_min20_pati10_dann0.3_cdann_balanced_d1x3.0/`

2. **Exp C — Per-clip CMVN (~50 min):** Set `CMN=True` in notebook.
   Subtracts per-clip time-axis mean per mel bin — already implemented, just never tried on Exp 4 base alone.

3. **Ablation: balanced batches only, no domain head (~50 min):** Set `DANN_ALPHA_MAX=0.0`, `CDANN=False`, `BALANCE_BATCHES=True`.
   We've never isolated how much of Exp 4's gain is the sampler vs. the adversarial head. If this scores ~0.24+, C-DANN is adding little.

4. **Field-val split (research, not a one-line change):** The val set is 99% D5 so val_BA is blind to BAunseen.
   To fix model selection, we would need to move some D1–D4 training clips into the val set. Caveat: we tried
   field-domain BA early stopping before (AttnPool experiment) and it spuriously fired at ep20 because only 3 D4
   val clips exist — any fix requires meaningfully more field-domain val samples, which means taking them from training.
   Worth planning but not trivial.

5. **Exp E — 10-seed sweep of Exp 4 (~8 hrs):** Ground truth for Exp 4 mean BAunseen.
   Single-seed baseline ranged 0.147–0.209 (~30%); Exp 4 at seed 42 could be similarly variable.

---

## Technical Reference: Model Architecture, DANN, and C-DANN

### 1. The Backbone: What Flows Into the Heads

The three CNN branches (kernel 3×3, 5×5, 7×7) each independently process the
same input spectrogram `[B, T, 64]` through three `ConvStage` layers. Each branch's
output is pooled and projected to a **64-dim vector**. The three are concatenated:

```
branch_3  → [B, 64]
branch_5  → [B, 64]  →  cat  →  [B, 192]
branch_7  → [B, 64]
```

Then:
```python
embedding = F.gelu(self.embedding(features))   # Linear(192 → 32) + GELU
```

This gives **z ∈ ℝ³²** — a single vector per sample summarising the entire recording.
This is the "neck" between backbone and heads. Everything downstream reads only from z.

### 2. Frame Attention Pooling (new in AttnPool experiment)

Replaces `masked_mean_max` inside each branch:

```python
frame_feat = branch_output.mean(dim=-1).permute(0, 2, 1)  # [B, T, C=64]
scores = Linear(64, 1)(frame_feat)                          # [B, T, 1]
scores[invalid_frames] = -inf                               # mask padding
gate = softmax(scores, dim=1)                               # [B, T, 1]
pooled = (branch_output * gate).sum(dim=2)                  # [B, C, F]
```

65 parameters per branch (64 weights + 1 bias), 195 total. The gate is trained
end-to-end — no separate objective. Output shape identical to `masked_mean_max`
so no downstream changes needed.

### 3. DANN and C-DANN

**DANN**: GRL inserted between embedding z and domain head. In the backward pass,
domain gradient is negated and scaled by λ, pushing the backbone toward
domain-invariant features (minimax game). Ganin lambda schedule ramps λ from 0
to alpha_max over training to avoid early collapse.

**C-DANN**: Domain classifier conditioned on species one-hot:
`domain_input = [z || onehot(species)]  →  W_d ∈ ℝ^{5×41}`
Targets conditional domain invariance `p(domain | z, species) ≈ uniform` rather
than marginal — preserves species-correlated features while removing recording
environment artefacts. Critical for An. dirus/minimus/stephensi whose species
identity is correlated with their (limited) domain.

**Key finding**: Batch balancing (WeightedRandomSampler by domain) was the dominant
driver of Exp 4's improvement. Without balanced batches (Exp 3), C-DANN was worse
than regular DANN. With balanced batches (Exp 4), the combination was best so far.

### 4. How to Read the Training Metrics

| Metric | What to watch for |
|--------|-------------------|
| `train_species_loss` | Should decrease; healthy range ~0.5–1.5 |
| `train_domain_accuracy` | **Want this to FALL** toward ~0.2 (chance) — means DANN is working |
| `val_species_balanced_accuracy` | Logged each epoch; should hold steady or rise |
| `val_domain_balanced_accuracy` | Lower is better for DANN |
| Early stopping criterion | **Reverted to `val_species_balanced_accuracy`** (field-domain BA criterion retired — too noisy with 3 D4 val clips) |

**Warning signs:**
- Both species and domain accuracy fall to random chance (1/9=0.11, 1/5=0.20) simultaneously → full collapse, cancel the run
- Val species BA collapses suddenly → alpha overwhelming species learning

---

## Where Things Live

---

## Research Findings & Brainstorming — June 15, 2026

### Strategic reframe: D1 species are the highest-leverage target

BAunseen is a mean across 9 species. An. dirus is likely stuck at 0 (76 clips, D4 unseen, zero across 10 seeds). Accepting that, the ceiling is 8/9 = 0.889. The real gains are in the three D1 species, which together have 3310 unseen test clips and large training datasets but currently contribute essentially zero:

| Species | Training clips | D1 unseen test clips | Baseline BAunseen | If improved to 0.3 |
|---------|---------------|---------------------|-------------------|--------------------|
| An. arabiensis | 16,630 | 1820 | 0.002 | +0.033 to BAunseen |
| An. gambiae | 37,011 | 818 | 0.0001 | +0.033 |
| Cx. quinquefasciatus | 56,745 | 672 | 0.001 | +0.033 |

Getting all three D1 species to 0.3 recall = **+0.10 to BAunseen**. This is the largest single lever available. FDA directly targets this by creating synthetic D1-like training clips. D1-heavy batch weighting (upweight D1 above equal-domain balance) is a one-line change that could help further.

### Bee and insect bioacoustic research — key findings

**Bricout et al. 2024 — "Bee Together" (*Sensors* 24(18))** — most directly relevant:
- Standard CNN: 99.2% on seen hives, collapses to 34–84% on unseen hives — structurally identical to CD-MSC.
- Fix: **pairwise xnorC classifier** — predicts whether two clips share the same label (binary) rather than predicting absolute class. Domain identity cancels out in relative comparisons.
- Result: 99.98% extrapolation to unseen hives. Key lesson: cross-entropy training learns domain fingerprints; relative comparison does not.

**Faiß & Stowell 2023 — "Adaptive representations" (*PMC*)** — LEAF learnable filterbank:
- Replaces fixed 64-bin mel with trainable filterbank; beats mel across all dataset sizes.
- Uses **room impulse response (RIR) augmentation**: convolve D5 clips with recorded room impulse responses → simulates different reverb environments without touching frequency content.
- Critical insect frequency band: 100–600 Hz.

**Ferreira et al. 2023/2025 (*Ecological Informatics*)** — 16 bee species, ~500 clips (similar scale to bottleneck species):
- Pre-training + Mixup + random crop augmentation is critical at small dataset sizes.
- 2025: AST (Audio Spectrogram Transformer) beats CNNs when pre-trained.

**Hearon et al. 2025 — buzzdetect** — YAMNet transfer learning across 5 different crop field environments:
- Works for detection but sensitivity is poor for quiet/short sounds.
- Confirms YAMNet transfers across field environments for general detection; less clear for fine-grained species ID.

**Faiß, Ghani & Stowell 2025 — InsectSet459** — 459 insect species mixed lab + field:
- Established as a cross-domain benchmark; "good but significant room for improvement."

### Planned Experiments — Priority Order

**Exp A — Prototype inference on FDA checkpoint (COMPLETE, 2026-06-15):**

| Classifier | BAseen | BAunseen | DSG |
|------------|--------|----------|-----|
| Softmax (standard) | 0.7982 | 0.2490 | 0.5492 |
| Prototype (cosine sim) | 0.7764 | 0.2419 | 0.5346 |
| Delta | −0.022 | **−0.007** | −0.015 |

Per-species (unseen domain recall):

| Species | Softmax | Prototype | Delta | Unseen |
|---------|---------|-----------|-------|--------|
| Ae. aegypti | 0.737 | 0.671 | −0.067 | D3 |
| Ae. albopictus | 0.585 | 0.649 | **+0.064** | D2 |
| Cx. quinquefasciatus | 0.830 | 0.882 | **+0.052** | D1 |
| An. gambiae | 0.677 | 0.630 | −0.047 | D1 |
| An. arabiensis | 0.110 | 0.095 | −0.016 | D1 |
| An. dirus | 0.000 | 0.000 | 0.000 | D4 |
| Cx. pipiens | 0.875 | 0.886 | +0.012 | D3 |
| An. minimus | 0.788 | 0.687 | **−0.101** | D2 |
| An. stephensi | 0.243 | 0.257 | +0.014 | D4 |

**Verdict: ❌ Negative overall.** Gains and losses cancel; net BAunseen −0.007.

**Key insight — embedding domain-invariance is heterogeneous across species:**
- Prototype helps species with large training sets where DANN embedding is well-aligned (Cx. qui 57K clips: +0.052, Ae. albo: +0.064)
- Prototype hurts small-dataset species (An. minimus 395 clips: −0.101) and species where D5→field shift is large (Ae. aegypti: −0.067, An. gambiae: −0.047)
- The softmax head partially compensates for residual domain bias that cosine distance to a D5 prototype cannot
- Prototype inference dropped from the priority queue; may revisit after metric learning or contrastive fine-tuning

**Exp B — D1-heavy batch weighting on Exp 4 (one training run, ~50 min):**
Modify the `WeightedRandomSampler` in `train.py`: give D1 clips 3× the weight of D2–D4 clips (instead of equal 1× per domain). D5 weighting unchanged. Everything else = Exp 4 config.

*Why:* An. arabiensis (1820 unseen test clips), An. gambiae (818), Cx. quinquefasciatus (672) all have D1 as their unseen domain and currently score near zero. They all have large D5 training datasets (16K–57K clips) — the bottleneck is D1 exposure, not data quantity. There are 634 D1 training clips total. Upweighting them gives the model more D1 passes per epoch without changing the training set.

**Exp C — Per-clip CMVN on Exp 4 (one training run, ~50 min):**
Normalize each clip to zero mean and unit variance **per mel bin over time**, applied at both train and test. This removes both gain (amplitude offset from microphone sensitivity) and variance (dynamic range differences) between recording environments. Unlike CMN (mean only) and unlike histogram matching (train-only → mismatch), CMVN is symmetric and stateless per clip.

*Why:* D4's spectral peak at 223 Hz vs D5's 595 Hz could partly be a gain/response artifact rather than true spectral content difference. CMVN can't fix that, but it removes one dimension of inter-domain variation cheaply.

**Exp D — FDA without C-DANN, β=0.01 (one training run, ~50 min):**
Plain cross-entropy + balanced batches + FDA at β=0.01 (vs β=0.05 previously). No adversarial domain head. This isolates whether the Fourier augmentation mechanism has value independent of the interaction with DANN's gradient.

*Why:* We only tested FDA entangled with C-DANN. If FDA alone (or with tiny β) beats the no-FDA baseline, it has merit and we can think about how to combine it more carefully.

**Exp E — 10-seed sweep of Exp 4 (~8 hrs):**
Run `run_multi_seed_experiments.py` with the Exp 4 config across all 10 standard seeds. Exp 4's single-seed result (BAunseen=0.2626) may be high or low relative to the true mean. The baseline 10-seed spread was 0.147–0.209 (~30% range) — Exp 4 could be similar.

*Why:* We can't confidently claim any approach beats another without multi-seed results. This is the ground truth for Exp 4.

### FDA result (2026-06-15, complete)

**Negative result.** Best checkpoint ep29: BAunseen=0.2490 < Exp 4's 0.2626. Early stopped ep39.
val_BA peaked at 0.838 — the highest val_BA observed — but this revealed a critical insight:
**val_BA and BAunseen are decoupled.** Val set is ~99% D5; improving D5 performance raises
val_BA without improving field generalisation. We are essentially blind during training to
whether BAunseen is improving. This complicates early stopping and model selection fundamentally.

Despite failing, FDA+C-DANN still strongly beats the vanilla baseline (0.2490 vs 0.1751 mean)
— the C-DANN balancing is carrying the load. FDA alone may still be worth trying without DANN,
or at smaller β, as a standalone augmentation on the Exp 4 base.

---

## Where Things Live

| Thing | Location |
|---|---|
| Raw audio | `Development_data/raw_audio/` (local + Drive, gitignored) |
| Features | Drive: `MyDrive/CD-MSC-feature` (restored by colab_dann.ipynb) |
| Domain feature stats | Drive: `MyDrive/CD-MSC-feature/domain_feature_stats.json` |
| Released baseline checkpoints | `outputs/MTRCNN_seed42_B64_E100_earlystop_min10_pati5/` (in repo) |
| Experiment outputs | Drive: `MyDrive/CD-MSC-outputs/` (saved at end of each Colab run) |
| Training notebook | `colab_dann.ipynb` — all experiment flags in cell 6 |
| Evaluation set | Not yet downloaded — Zenodo link in README.md line 9 |
| Brainstorm doc | `docs/brainstorm_jun2026.md` |
