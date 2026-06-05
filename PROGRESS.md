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

**NOT yet implemented:** C-DANN (conditional domain discriminator)

---

## Experiments Run

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

### 1. Regular DANN, alpha_max=0.3 (immediate — just change one line in notebook)
Run `colab_dann.ipynb` with `DANN_ALPHA_MAX = 0.3`. Quick reference point.
Expected: stable training, some improvement in BAunseen over baseline.

### 2. C-DANN, alpha_max=0.3 (needs code change in model.py + engine.py)
Implement conditional domain discriminator. Run alongside or immediately after step 1.
Expected: better protection of rare species, potentially higher BAunseen than regular DANN.

### 3. Batch Balancing (alongside C-DANN, not after)
The D5 dominance problem affects all DANN variants — a batch of 64 currently has
~63 D5 samples. The domain discriminator essentially learns "is this D5 or not?"
Weighted sampling to enforce equal domain representation per batch (e.g. ~13 samples
per domain) makes the adversarial signal much more meaningful.

**Important:** D4 has only 80 training samples total — equal-domain sampling will
oversample D4 heavily. Use oversampling (sample with replacement from D4) or
set per-domain weights rather than strict equality.

Batch balancing + C-DANN together address the D5 problem from two angles:
- Balancing fixes what the adversary *sees* (more D1–D4 in every batch)
- C-DANN fixes what the adversary is *conditioned on* (per-species domain invariance)

### 4. Download evaluation set and generate submission predictions
Evaluation set on Zenodo (link in README.md line 9). Must be done before 2026-06-15.
Use `predict.py` or `evaluate.py` once we have a best-performing checkpoint.

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
