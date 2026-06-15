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
| **AttnPool** | **Exp 4 + learned frame attention** | — | — | — | **🔄 RUNNING NOW** |

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

**AST (collaborator saha23s, LODO branch):** Pretrained on AudioSet which contains no
mosquito wingbeats — the pretrained representations have no spectral resolution for 20-80 Hz
frequency differences between species. AST representations may also encode field background
textures (AudioSet contains wind, traffic, crowds) and use them as domain signal rather than
suppressing them. Additionally, fine-tuning ~87M parameters on datasets with only 127 An.
dirus clips is catastrophic forgetting territory.

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
lucky epoch can spike to ~0.99 (observed in HM-2, caused premature stopping at ep30).
**When watching a run**, be suspicious if early stopping fires before epoch 40.

---

## Current Experiment: Frame Attention Pooling (AttnPool)

**What it changes:** Replaces `masked_mean_max` in each of the three MTRCNN branches
with a learned `FrameAttentionPool` module. Scores each valid time frame by passing its
channel-mean vector through Linear(64→1), then softmax-weights the frames before summing.

**Rationale:** Mean+max pooling treats all valid frames equally. A field recording of
An. dirus in D4 has the wingbeat present for only a fraction of the clip; the rest is
D4-specific background noise. If the model learns to upweight clean wingbeat frames —
which it has seen in D5 lab recordings — the D5 representation may transfer to field
conditions more cleanly, directly helping the three bottleneck species.

**Config:** Identical to Exp 4 except `use_attention_pool=True`. Only 195 new parameters
(3 branches × Linear(64→1)+bias). Output dir: `MTRCNN_seed42_B64_E100_earlystop_min20_pati10_dann0.3_cdann_balanced_attnpool/`

**Interpreting the result:**
- BAunseen > 0.2626: noise dilution in pooling was a real problem; attention helps
- BAunseen ≈ Exp 4: pooling is not the bottleneck; move to feature-level augmentation
- BAunseen < 0.2626: mean+max was doing something useful; investigate why

---

## Next Steps (priority order after AttnPool result)

1. **If AttnPool wins:** Run 2–3 more seeds to confirm, then combine with FDA
2. **If AttnPool ties/loses:** Try FDA (Fourier Domain Adaptation) — directly synthesises
   field-domain versions of D5 clips by swapping low-frequency STFT amplitude with D1–D4
   clips at training time. Requires raw audio on Drive (confirmed available as zip).
   Most principled augmentation for the bottleneck species problem.
3. **Per-clip CMVN:** Extend CMN (mean-only) to full mean+variance normalisation per clip.
   Removes both DC offset and gain differences from features without domain labels.
   One-line change; worth a quick ablation against Exp 4.
4. **PCEN:** Replace log-mel with Per-Channel Energy Normalisation. Adapts to local noise
   floor, making features more robust to variable field SNR. Requires re-extraction.

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
| Early stopping criterion | Mean D1–D4 field BA in val set (very noisy — see Known Limitation above) |

**Warning signs:**
- Both species and domain accuracy fall to random chance (1/9=0.11, 1/5=0.20) simultaneously → full collapse, cancel the run
- Val species BA collapses suddenly → alpha overwhelming species learning

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
