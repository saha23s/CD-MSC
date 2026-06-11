# CD-MSC Project Progress & Technical Notes

Branch: `aaron/preprocessing` | Collaborator: saha23s | **Deadline: 2026-06-15**

---

## Environment & Setup

- Repo: shared fork `saha23s/CD-MSC`, working branch `aaron/preprocessing`
- Collaborator saha23s has branch `feature/lodo-ast-augmentation` (Leave-One-Domain-Out + AST + augmentation)
- Local venv set up; `CLAUDE.md` committed with full architecture docs
- Evaluation set released 2026-06-01 on Zenodo (link in README.md line 9) — not yet downloaded

---

## Data

- 271K clips, 9 species, 5 domains (D1–D5)
- Raw audio: `Development_data/raw_audio/` (local + Drive, gitignored)
- Metadata split files: `Development_data/metadata/` (committed to repo 2026-06-05 — were missing, caused Colab FileNotFoundError on fresh clone)
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

| Species | Unseen test domain | Notes |
|---------|--------------------|-------|
| Ae. aegypti | D3 | |
| Ae. albopictus | D2 | |
| Cx. quinquefasciatus | D1 | |
| An. gambiae | D1 | |
| An. arabiensis | D1 | |
| **An. dirus** | **D4** | 76 training samples total; D4 has only 80 samples across all species |
| Cx. pipiens | D3 | |
| **An. minimus** | **D2** | ALL test samples are in unseen domain |
| **An. stephensi** | **D4** | ALL test samples are in unseen domain |

An. dirus, An. minimus, and An. stephensi have zero seen-domain test samples —
their entire test evaluation is in the unseen domain. Any domain adaptation method
must not destroy their features.

### Why the Authors Chose This Split

The per-species unseen domain assignment is deliberate. D5 is always in training
(it's the only domain with enough data to train on), and one scarce field domain is
withheld per species as the unseen test. The assignment is constrained by which field
domains each species actually has data in — An. dirus only exists in D4 and D5, so
D4 was the only viable unseen domain for it.

The split is challenging by design: you train almost entirely on clean lab audio (D5)
and are tested on field conditions. The train/val/test assignment of sample IDs is
**fixed by the challenge** and must not be changed (results would be incomparable to
the baseline and other teams). What we *can* change is how we sample from training
data (batch balancing, oversampling).

---

## Baseline Results (released, 10-seed mean ± std)

| Metric | Test |
|--------|------|
| BAseen | 0.88 ± 0.01 |
| BAunseen | 0.18 ± 0.02 |
| DSG | 0.71 ± 0.02 |

Seed 42 single run: BAseen=0.883, BAunseen=0.168, DSG=0.716

Per-domain species balanced accuracy (seed 42): D5=0.883, D1=0.415, D3=0.274, D4=0.145, D2=0.146
The model essentially only works on D5.

---

## What's Implemented in the Repo

### Regular DANN with GRL (committed 2026-06-05)

- `framework/model.py` — `GRL` class (torch.autograd.Function); `MTRCNNClassifier.forward()`
  accepts optional `alpha` parameter; routes domain head through GRL when set,
  original behaviour when `alpha=None`
- `framework/engine.py` — `dann_alpha()` Ganin schedule function; `train_one_epoch()`
  accepts `epoch`, `total_epochs`, `dann_alpha_max`
- `train.py` — passes epoch info and `dann_alpha_max` from config; experiment name
  includes `_dann{alpha}` suffix to avoid colliding with baseline output directories
- `colab_dann.ipynb` — Colab notebook: clone/pull, restore features from Drive,
  parameterised DANN_ALPHA_MAX/SEED cell, train, save to Drive, results comparison table

### C-DANN (committed 2026-06-06)

- `framework/model.py` — `cdann` flag; `domain_classifier` input 32→41 when cdann=True;
  `forward()` accepts `species_labels`, concatenates one-hot after GRL
- `framework/engine.py` — `species_labels` passed to model in both train and eval
- `train.py` — `_cdann` suffix added to experiment name when cdann=True
- `colab_dann.ipynb` — `CDANN=True/False` parameter, wired into config cell
- `CDANN.md` — standalone technical reference: An. dirus example, arch diagram, math

### Batch Balancing via WeightedRandomSampler (committed 2026-06-06)

- `train.py` — when `batch_balance_domain=True`, builds a `WeightedRandomSampler` with
  **joint (species, domain) pair** inverse-frequency weights — rare (species, domain)
  combinations get highest weight. D4 clips (~80 samples) are oversampled ~500× per epoch.
  Addresses both domain imbalance AND per-species coverage in scarce field domains.
  Experiment name gets `_balanced` suffix.
- `colab_dann.ipynb` — `BALANCE_BATCHES=True/False` parameter

### SpecAugment (committed 2026-06-08)

- `framework/dataset.py` — `_spec_augment()` applies random time masking (0–40 frames) and
  frequency masking (0–10 mel bins) after crop+normalize, training only.
  `self.spec_augment = spec_augment and training` — automatically off at eval.
- `train.py` — `spec_augment` params passed from config to dataset; `_specaug` suffix in name
- `colab_dann.ipynb` — `SPEC_AUGMENT=True/False` parameter (default True)

---

## Experiments Run

### Experiment 5 — C-DANN alpha=0.3 + Balanced + Batch=128 + SpecAugment (2026-06-08) ⚠️ WORSE THAN EXP 4

| Metric | Baseline mean | Exp 4 (prev best) | Exp 5 | Change vs Exp 4 |
|--------|--------------|-------------------|-------|-----------------|
| BAseen | 0.8806 | 0.7950 | 0.7227 | −0.072 |
| **BAunseen** | **0.1751** | **0.2626** | **0.2308** | **−0.032** |
| **DSG** | **0.7055** | **0.5324** | **0.4919** | **−0.041** |

Adding SpecAugment (time_mask=40, freq_mask=10) and doubling batch size 64→128 degraded BAunseen vs Exp 4.
Two likely causes:
1. **SpecAugment too aggressive for rare species**: ~80 D4 clips are oversampled heavily. Randomly masking
   up to 10 mel bins of 64 frequently destroys wingbeat harmonics in the (species, domain) pairs that
   matter most for BAunseen. The signal band is narrow and masks are not domain-aware.
2. **Fewer gradient updates per epoch**: `WeightedRandomSampler` keeps `num_samples=len(train_dataset)`;
   batch=128 gives half as many `.backward()` calls per epoch as batch=64.

DSG improved slightly (0.5324→0.4919) only because BAseen fell faster than BAunseen — not a genuine gain.
**Lesson**: SpecAugment needs careful tuning (smaller masks, or majority-domain-only) when rare field clips
are already scarce and heavily oversampled.

---

### Experiment 4 — C-DANN alpha=0.3 + Balanced Batches (2026-06-08) ✅ BEST SO FAR

| Metric | Baseline 10-seed mean | Baseline seed 42 | C-DANN+Balanced | Change vs baseline mean |
|--------|----------------------|------------------|-----------------|------------------------|
| BAseen | 0.8806 | 0.8813 | 0.7950 | -0.086 |
| **BAunseen** | **0.1751** | **0.1557** | **0.2626** | **+0.087 (+50%)** |
| **DSG** | **0.7055** | **0.7255** | **0.5324** | **-0.173** |

Single seed (seed=42). Best result so far. BAunseen improved 50% relative over the
official baseline. DSG dropped by 0.173 — the largest domain gap reduction yet.
BAseen fell to 0.795 (−0.086 vs baseline) — expected tradeoff from forcing the model
away from D5 specialisation. Since BAunseen is the primary challenge metric this is
a strong result. Batch balancing appears to be the dominant driver — the adversarial
signal became meaningful for the first time with balanced domain representation per batch.

### Experiment 3 — C-DANN, alpha_max=0.3 (2026-06-06) ⚠️ BELOW REGULAR DANN

| Metric | Baseline 10-seed mean | Baseline seed 42 | C-DANN alpha=0.3 | Change vs baseline mean |
|--------|----------------------|------------------|-----------------|------------------------|
| BAseen | 0.8806 | 0.8813 | 0.8879 | +0.007 |
| **BAunseen** | **0.1751** | **0.1557** | **0.1865** | **+0.011 (+6.5%)** |
| **DSG** | **0.7055** | **0.7255** | **0.7014** | **-0.004** |

Single seed (seed=42). BAunseen improved over baseline but fell short of regular DANN
alpha=0.3 (0.2225). Hypothesis: C-DANN's stronger discriminator (has species one-hot,
so it can predict domain more easily) effectively increases adversarial pressure at the
same alpha, partially collapsing species features. Next step: try C-DANN with alpha=0.1.

### Experiment 2 — Regular DANN, alpha_max=0.3 (2026-06-05) ✅ BEST SO FAR

| Metric | Baseline 10-seed mean | Baseline seed 42 | DANN alpha=0.3 | Change vs baseline mean |
|--------|----------------------|------------------|----------------|------------------------|
| BAseen | 0.8806 | 0.8813 | 0.8769 | -0.004 (negligible) |
| **BAunseen** | **0.1751** | **0.1557** | **0.2225** | **+0.047 (+27%)** |
| **DSG** | **0.7055** | **0.7255** | **0.6544** | **-0.051** |

Single seed (seed=42). BAunseen improved 27% relative over the official baseline while
BAseen barely changed — exactly the desired pattern. DSG dropped by 0.051. Strong
result for a first stable DANN run. Checkpoint and metrics saved to Drive:
`MyDrive/CD-MSC-outputs/MTRCNN_seed42_B64_E100_earlystop_min10_pati5_dann0.3/`

### Experiment 1 — Regular DANN, alpha_max=1.0 (2026-06-05) ❌ COLLAPSED

**Result:** Collapsed at epoch 13.
- Species balanced accuracy → 0.111 (= 1/9, random chance)
- Domain balanced accuracy → 0.200 (= 1/5, random chance)
- Run cancelled.

**Why it failed:** At epoch 13 (p=0.13), the Ganin schedule gives λ≈0.57.
D5 dominates 99.4% of training batches, so the domain gradient is large and
mostly "D5-vs-rest" signal. Negated and scaled to 0.57, it overwhelmed species
learning. The embedding collapsed to noise — domain-invariant but also
species-invariant (both metrics at random chance simultaneously).

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

### 2. What a Classifier Head Actually Is

Each head is just a single linear layer — no hidden layers, no activation:

```
species head:  ŷ_s = W_s · z + b_s    W_s ∈ ℝ^{9×32},  ŷ_s ∈ ℝ⁹
domain head:   ŷ_d = W_d · z + b_d    W_d ∈ ℝ^{5×32},  ŷ_d ∈ ℝ⁵
```

These raw outputs are called **logits** — unnormalised scores. To get probabilities
you'd apply softmax, but cross-entropy loss does this internally:

```
L_species = -log( exp(ŷ_s[y]) / Σ_k exp(ŷ_s[k]) )   for true class y
```

**Why keep the heads simple?** The backbone has to do all the hard representational
work. If the heads had hidden layers, they could compensate for a weak backbone.
A single linear layer forces z to already be the right representation.

### 3. Gradient Flow With Two Heads (Baseline, No GRL)

When `.backward()` is called, PyTorch walks the computation graph from the loss
back through every parameter. With two heads sharing the same z:

```
L = L_species + L_domain

∂L/∂W_s        = ∂L_species/∂W_s              (species head only)
∂L/∂W_d        = ∂L_domain/∂W_d               (domain head only)
∂L/∂θ_backbone = ∂L_species/∂θ_backbone + ∂L_domain/∂θ_backbone
```

Both gradients flow back through z into the shared backbone. The domain gradient
says: *adjust the backbone so z predicts domain better.* This is **multi-task
learning** — the domain head is auxiliary supervision. It shapes the representation
but does NOT enforce domain invariance.

### 4. The GRL: Flipping One Gradient Stream

The GRL is inserted between z and the domain head. Mathematically it defines `g_λ`:

```
Forward:   g_λ(z) = z                     (identity — no effect on predictions)
Backward:  ∂g_λ/∂z = −λI                 (negated and scaled)
```

The forward pass is untouched — the domain head still computes normal cross-entropy.
But when `.backward()` propagates the domain gradient through the GRL, it gets negated:

```
Without GRL:  ∂L/∂θ_backbone = ∂L_species/∂θ_backbone  +  λ·∂L_domain/∂θ_backbone
With GRL:     ∂L/∂θ_backbone = ∂L_species/∂θ_backbone  −  λ·∂L_domain/∂θ_backbone
```

The sign flip changes "improve domain prediction" → "worsen domain prediction."
The domain head still tries to get better at predicting domain; the backbone tries
to prevent it. This is a minimax game:

```
min_{θ_f, θ_s}  max_{θ_d}  [ L_species(θ_f, θ_s) − λ · L_domain(θ_f, θ_d) ]
```

At the **saddle point**, the domain classifier performs at chance (~20% for 5 classes),
meaning z contains zero domain information. The GRL solves this with ordinary SGD
in a single `.backward()` call — no alternating training phases needed.

### 5. Lambda Scheduling: Why You Can't Start at λ=1

Early in training, z is essentially random. If λ=1 immediately, the domain gradient
(even negated) is large and noisy — it competes with species learning before the
backbone has learned anything useful. The Ganin schedule:

```
λ(p) = alpha_max · [ 2 / (1 + exp(−10p)) − 1 ]    p = epoch / total_epochs ∈ [0,1]
```

| Progress p | epoch (of 100) | λ (alpha_max=1.0) | λ (alpha_max=0.3) |
|---|---|---|---|
| 0.00 | 1 | 0.00 | 0.00 |
| 0.13 | 13 | **0.57** | **0.17** |
| 0.30 | 30 | 0.82 | 0.25 |
| 0.50 | 50 | 0.92 | 0.28 |
| 1.00 | 100 | 1.00 | 0.30 |

With alpha_max=1.0, λ reaches 0.57 by epoch 13 — this is what caused the collapse.
With alpha_max=0.3, λ is only 0.17 at epoch 13, giving species learning time to
stabilise before adversarial pressure builds. The maximum pressure ever applied is
0.30, which is much more conservative.

### 6. How to Read the Training Metrics

Each epoch logs:

| Metric | What it means | What to watch for |
|--------|--------------|-------------------|
| `train_species_loss` | Cross-entropy on species classification | Should decrease and stay low (~0.5–1.5) |
| `train_domain_loss` | Cross-entropy on domain prediction (through GRL) | Will fluctuate — not a reliable signal |
| `train_species_accuracy` | Fraction correctly classified (per-sample) | Should be high (>0.80) |
| `train_domain_accuracy` | Fraction correctly classified by domain | **Want this to FALL** toward ~0.2 (chance) |
| `val_species_balanced_accuracy` | **Primary metric** — mean recall per species on val | Early stopping watches this; goal: match or beat baseline ~0.54 |
| `val_domain_balanced_accuracy` | Mean recall per domain on val | Lower is better for DANN; chance = 0.2 |

**Healthy DANN run:**
- `train_domain_accuracy` gradually falls from ~0.9 toward 0.2–0.4
- `val_species_balanced_accuracy` holds steady or improves vs baseline
- Both losses decrease smoothly

**Warning signs:**
- `val_species_balanced_accuracy` collapses suddenly (alpha overwhelming species learning)
- Both metrics fall to random chance simultaneously: species=0.111, domain=0.200 — full collapse, cancel the run
- `train_domain_accuracy` stays above 0.8 throughout — adversary not working, features still domain-specific

**Why domain_accuracy falling is a good sign:** it means the saddle point is being
approached and the embedding is becoming domain-invariant.

### 7. C-DANN: What Changes Mathematically

Regular DANN makes z domain-invariant in the **marginal** sense:

```
p(domain | z) ≈ uniform
```

Problem: if species and domain are correlated (e.g. An. dirus appears almost
exclusively in D4), then removing domain information also removes the only
signal distinguishing An. dirus. Regular DANN may hurt rare species.

C-DANN conditions the discriminator on the species label y:

```
ŷ_d = W_d · [z ; onehot(y)] + b_d    W_d ∈ ℝ^{5×41}  (32 + 9 = 41)
```

This targets the **conditional** distribution:

```
p(domain | z, y) ≈ uniform    for all y
```

"Given we already know the species, is there still domain information in z?"
Only recording-environment artefacts are removed — species-correlated features
survive. Critical for An. dirus, An. minimus, An. stephensi whose unseen test
domain is their only test signal.

**At inference:** the domain head is not used for predictions (only species_logits
matter), so unavailability of ground-truth species labels at test time is not a
problem.

**Code changes needed for C-DANN:**
- `framework/model.py`: `domain_classifier` input 32 → 41; `forward()` accepts
  optional `species_labels` tensor and concatenates `onehot(species_labels)` to
  the reversed embedding before the domain head
- `framework/engine.py`: pass `species_labels` to model during training

---

## Why alpha_max=0.3 and Not 0.5 or 0.1

The collapse at alpha_max=1.0 was caused by the combination of:
1. The Ganin schedule reaching λ≈0.57 by epoch 13 (13% of training)
2. D5 dominating 99.4% of batches, making the domain gradient disproportionately large

alpha_max=0.3 caps the maximum gradient reversal at 30% of the domain gradient
at any point in training. Even at full strength (epoch 100), the adversarial signal
is modest. This is conservative but the right call given the extreme domain imbalance.
If results with 0.3 are stable, 0.5 is a reasonable next experiment.

---

## Current Plan (priority order)

### 1. Frame Gating — Exp 6 ← NEXT IMPLEMENTATION
Learned per-frame attention weights gate each frame's contribution to the final embedding.
Motivation: not all frames carry species signal. Background frames and domain-artefact frames
(equipment hum, wind noise in field recordings) dilute the pooled embedding. A lightweight
attention module lets the model learn to focus on frames containing wingbeat signal.

**Implementation sketch** (inside `MTRCNNBranch` or before `masked_mean_max`):
```
frame_scores = Linear(64 → 1)(branch_output)   # [B, T, 1] gate logits
gate = softmax(frame_scores, dim=1)             # normalised attention weights
embedding = sum(gate * branch_output, dim=1)    # weighted mean instead of mean+max
```
The gate is trained end-to-end — no separate objective needed. Expected benefit for unseen
field domains: model learns to ignore variable background noise and focus on wingbeat bursts.

### 2. SpecAugment re-tuning
Exp 5 showed default masks (time=40, freq=10) hurt BAunseen. Options:
- Smaller masks: `time_mask=20, freq_mask=5`, keep batch=64
- Domain-aware masking: apply SpecAugment only to D5 clips; field clips are too scarce to mask

### 3. C-DANN alpha=0.1 + balanced batches
Gentler adversarial pressure. C-DANN at 0.3 was below regular DANN without balancing; with
balancing the combination improved (Exp 4). alpha=0.1 may recover BAseen without sacrificing BAunseen.

### 4. Regular DANN + balanced batches (ablation)
Set `CDANN=False`, `DANN_ALPHA_MAX=0.3`, `BALANCE_BATCHES=True`. Isolates whether C-DANN
is contributing or if balancing alone accounts for the Exp 4 jump.

### 5. Download evaluation set and generate submission predictions
Evaluation set on Zenodo (link in README.md line 9). **Must be done before 2026-06-15.**

---

## Preprocessing Ideas (Brainstormed, Not Yet Implemented)

All of these target the core problem: D5 lab conditions look different from D1–D4 field conditions.
The goal is either to make D5 training clips look more like field recordings, or to strip
recording-environment information from both.

### Signal Processing (require raw audio or re-extraction)

**HPSS (Harmonic-Percussive Source Separation)**
Decomposes audio into tonal (harmonic) and transient/noise (percussive) components. Keeping only the
harmonic component suppresses broadband background noise (wind, rain, traffic) which is the main
acoustic difference between D5 and field domains. Mosquito wingbeats are tonal — dominant harmonic
at ~300–800 Hz. Requires raw audio; adds ~100ms per clip; needs re-running `extract_features.py`.

**PCEN (Per-Channel Energy Normalisation)**
Replaces log compression in feature extraction. Uses an exponential moving average to normalise
against the local noise floor, making features more robust to variable background noise levels.
Implemented in `librosa.pcen`. Requires re-running `extract_features.py` (new `config_signature`).

**Gaussian Noise Injection**
Add random white/pink noise to D5 clips during training to simulate field recording conditions.
Simple, cheap, no re-extraction needed. Can approximate field domain SNR statistics.

**Time Stretching / Speed Perturbation**
Vary playback speed ±5–10%. Changes wingbeat frequency slightly — broadens training distribution.
**Caution**: species wingbeat frequency IS the discriminative feature; aggressive stretching
(>10%) may shift a sample across a species class boundary.

### Feature-Space Methods (applicable to existing `.pkl` features)

**CMN (Cepstral Mean Normalisation)**
Subtract per-clip mean across time in mel/cepstral domain. Removes microphone and channel
effects (DC offset in mel space). Fast — applicable directly to loaded features without
re-extraction. Risk: removes low-frequency information that may carry species signal.

**Histogram Matching / Feature Distribution Alignment**
Match the mel-spectrogram statistics of D5 clips to D1–D4 domain statistics.
Variants:
- *Mean/variance alignment*: shift and scale each frequency bin of D5 clips to match D1–D4
  mean and std. Lightweight; can be computed from existing pkl statistics.
- *Full histogram matching*: match the per-bin cumulative distribution of D5 to D1–D4.
  More aggressive; preserves rank ordering.
This is essentially feature-space domain randomisation — the model sees D5 content with
D1–D4 spectral "clothing" applied as augmentation during training.

### Audio-Space Domain Transfer

**Fourier Domain Adaptation (FDA — Yang & Soatto 2020)**
Swap the amplitude spectrum of a D5 clip's STFT with one sampled from a D1–D4 clip,
keeping the D5 phase intact. The result has D5's temporal structure but D1–D4's
frequency coloring (noise floor, spectral tilt).
- Implemented at training time by sampling a random D1–D4 clip per D5 clip in each batch
- Swap only low-frequency components (below threshold β in the 2D STFT) to preserve
  high-frequency wingbeat detail
- Requires raw audio in the training loop; eval can still use precomputed `.pkl` features
- Most direct method for domain appearance transfer; well-studied in vision (less so in audio)

---

## Where Things Live

| Thing | Location |
|---|---|
| Raw audio | `Development_data/raw_audio/` (local + Drive, gitignored) |
| Metadata | `Development_data/metadata/` (in repo) |
| Features | Drive: `MyDrive/CD-MSC-feature` (restored by colab_dann.ipynb) |
| Released baseline checkpoints | `outputs/MTRCNN_seed*/model/` (in repo) |
| Colab quickstart | `colab_quickstart.ipynb` |
| DANN training notebook | `colab_dann.ipynb` |
| Evaluation set | Not yet downloaded — Zenodo link in README.md line 9 |
