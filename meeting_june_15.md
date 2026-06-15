# CD-MSC Meeting Notes — June 15, 2026

**Task:** Classify mosquito species from audio clips recorded in different environments (lab vs. field).
**Primary metric:** `BAunseen` — balanced accuracy on clips from recording environments the model has never seen that species in.
**Baseline BAunseen: 0.175. Our best so far: 0.263.**

---

## The Core Problem

Training data is 99.4% from a quiet lab (D5). Test includes 4 field domains (D1–D4) with noisy, equipment-specific acoustic conditions. The model learns to recognise species in lab conditions but fails when those species appear in field recordings.

**Data gap figures:** `spectrogram_grid_trainval.png`, `spectrogram_domain_comparison_Aedes_aegypti.png`

### Why three species are nearly unsolvable

| Species | Training clips | Test clips | Test domain | Baseline BAunseen |
|---------|---------------|------------|-------------|-------------------|
| An. dirus | 76 | 40 | D4 (all) | **0.000** |
| An. minimus | 395 | 99 | D2 (all) | 0.238 |
| An. stephensi | 525 | 74 | D4 (all) | 0.420 |

These three species have **zero test clips in any domain they've been trained in** — every test prediction is zero-shot cross-domain transfer. An. dirus has 76 training clips (all D5) and scores zero across 10 seeds.

Five other species also score near-zero in their unseen domains despite having 14K–57K training clips — the lab-to-field transfer is the bottleneck, not data quantity.

Only *Cx. pipiens* generalises well (BAunseen=0.804) because its unseen domain (D3) acoustic conditions appear in training data from other species.

---

## What We've Tried

| Approach | BAunseen | vs. Baseline | Verdict |
|----------|----------|--------------|---------|
| Baseline (10-seed, released) | 0.175 | — | Reference |
| DANN α=0.3 (domain adversarial) | 0.223 | +27% | Works |
| C-DANN α=0.3 (conditional adversarial) | 0.187 | +7% | Worse without balancing |
| **C-DANN + domain-balanced batches** | **0.263** | **+50%** | **Best — current benchmark** |
| + SpecAugment / larger batch | 0.231 | +32% | Worse; masks hurt rare clips |
| + CMN + Gaussian noise on D5 | 0.245 | +40% | Marginal |
| Supervised Contrastive (SupCon) variants | 0.201–0.244 | +15-39% | All below best; no cross-domain pairs |
| Histogram matching (D5→field stats) | 0.250 | — | BAseen collapsed; train/test mismatch |
| Frame attention pooling | 0.215 | +23% | Worse than Exp 4; max pooling was useful |
| FDA (Fourier domain adaptation, β=0.05) | 0.2490 | −5% | ❌ Hurt; conflicted with C-DANN; val_BA decoupled from BAunseen |

**Key finding:** Domain-balanced batch sampling was the single largest driver. Without it, C-DANN performed worse than plain DANN (Exp 3 vs Exp 4). The adversarial signal needs balanced exposure to all domains or it only sees D5 vs. D5.

---

## What Has NOT Worked and Why

- **Supervised contrastive learning:** For An. dirus, An. minimus, An. stephensi — all in-batch positives are D5 vs D5. No cross-domain pairs exist, so SupCon just tightens D5 clusters without helping field transfer. It competes with the DANN gradient in the embedding and hurts overall.

- **Feature preprocessing (CMN, noise, HPSS, histogram matching):** Lab→field transforms applied only to training data cause a train/test mismatch at test time: the model sees transformed D5 in training, but D5 test clips arrive untransformed. Histogram matching collapsed BAseen to 0.30.

- **F0 (fundamental frequency) as features:** pyin-extracted F0 for Ae. aegypti is 105 Hz in D1 vs 548 Hz in D5 — not domain-invariant. Failed in training; abandoned. See `f0_heatmap.png`.

- **Attention pooling (replacing mean+max with learned frame weights):** Removed the max-pooling component, which was capturing the peak wingbeat energy frame. Softmax over all frames averaged it away.

---

## What We Just Tried: FDA (complete)

**Fourier Domain Adaptation (Yang et al. ECCV 2020, adapted for spectrograms):**
At training time, for each D5 clip, with 50% probability swap the low-spatial-frequency 2D Fourier amplitude components with a randomly drawn D1–D4 clip. Species label preserved. β=0.05 swaps ~1.2% of Fourier coefficients.

**Result: BAunseen=0.2490 — worse than Exp 4's 0.2626.** FDA hurt when stacked on top of C-DANN.

**Key insight discovered:** val_BA peaked at 0.838 — the highest we've seen — but BAunseen fell. The validation set is ~99% D5 clips. **Val balanced accuracy and BAunseen are decoupled.** Any technique that improves D5 performance will raise val_BA without helping field generalisation. We have no reliable training signal for BAunseen during training — we are essentially blind to what we care about.

**Why FDA + C-DANN likely conflicted:** C-DANN's adversarial gradient is calibrated for a natural D5-heavy distribution. FDA creates synthetic "hybrid" clips that are neither clean D5 nor real field recordings. The model adapts to FDA artifacts that don't appear at test time. Testing FDA without C-DANN (plain cross-entropy) would have been the more controlled comparison.

---

## Insights from Bee & Insect Bioacoustics Research

We looked at the recent insect sound classification literature for ideas. Three findings stood out:

**Bricout et al. 2024 — "Bee Together"** (*Sensors*) — directly analogous to our problem:
- A standard CNN gets 99.2% on seen beehives but collapses to 34–84% on unseen hives — the same failure mode as CD-MSC.
- Their fix: **pairwise comparison** instead of absolute classification. Instead of "what species is this?", ask "do these two clips sound like the same species?" Domain identity cancels out in relative comparisons.
- Getting 99.98% on unseen hives by comparing against reference clips of known species.
- **Implication for us:** We could build a prototype classifier at test time — no retraining needed. Compute the mean 32-dim embedding per species from training clips. Classify test clips by cosine similarity to prototypes instead of softmax. Softmax is calibrated for D5; relative distance comparison is domain-agnostic.

**Faiß & Stowell 2023** (*PMC*) — **RIR augmentation** for insects:
- Convolving lab recordings with room impulse responses (different room reflections/reverb) improved cross-environment generalisation.
- Directly complementary to FDA: FDA changes spectral coloration, RIR changes reverberation.
- Package: `openair.markbijl.co.uk` has free RIR recordings.

**Ferreira et al. 2023/2025** — 16 bee species, similar small-dataset regime:
- Mixup + random crop essential for small datasets. Pre-training (ImageNet/AudioSet) helps when species data is scarce.

---

## New Directions Being Considered

1. **Prototype test-time inference (no retraining, quick to try):** Compute per-species mean embeddings from training clips; classify evaluation clips by cosine distance. Addresses softmax calibration failure under domain shift.

2. **D1-heavy batch weighting:** An. arabiensis, An. gambiae, Cx. quinquefasciatus all have D1 as their unseen domain and together account for 3310 unseen test clips but ~0 current recall. Increasing D1 weight in the sampler is a one-line change and could be the highest-leverage single intervention.

3. **Room impulse response (RIR) augmentation:** Convolve D5 clips with field room impulse responses. Complements FDA by targeting reverb rather than spectral coloration.

4. **Domain identification at test time:** Build a domain classifier (5-class) trained on training data. Apply to evaluation clips to estimate domain, then apply domain-specific normalisation. No labels needed — uses acoustic features only.

---

## Key Open Questions

1. **Is An. dirus fundamentally unsolvable?** 76 training clips, D4 as unseen domain (D4 has only 80 total training clips across all species, spectral peak at 223 Hz vs 595 Hz in D5 — `spectral_peak_domain_shift.png`). Is there any intervention that can rescue 40 zero-shot test clips?

2. **What does D4 actually represent?** Spectral peak 223 Hz median vs 595 Hz in D5 is a large shift. Equipment artefact? Background insects? Knowing what D4 is might suggest targeted approaches.

3. **Is the representation fundamentally limited?** Field domains have low voiced fraction — many noisy frames with no clean wingbeat (`f0_voiced_fraction.png`). Can any model classify from noisy partial wingbeats, or do we need denoising upstream?

4. **Prototype inference — worth implementing?** If test-time cosine similarity can close even half the gap for D1 species, it requires no new training.

5. **Is there additional clip metadata** (equipment model, GPS, weather) that could serve as a domain supervision signal?

---

## Next Steps (prioritised)

**A — Prototype inference on Exp 4 checkpoint (no GPU needed, ~20 min)**
Load the existing Exp 4 model. Compute the mean 32-dim embedding per species across all training clips (9 "prototype" vectors). At test time, classify by cosine similarity to prototypes instead of the softmax head. No retraining. Inspired by Bricout 2024's pairwise comparison approach for unseen beehives.

**B — D1-heavy batch weighting (~50 min training run)**
An. arabiensis, An. gambiae, Cx. quinquefasciatus all have D1 as their unseen test domain and together account for 3310 unseen clips currently scoring near zero. There are 634 D1 training clips. One-line change: give D1 clips 3× weight in the sampler vs D2–D4.

**C — Per-clip CMVN (~50 min training run)**
Normalize each clip to zero mean + unit variance per mel bin (at both train and test time). Removes gain and dynamic range differences between domains. No train/test mismatch — symmetric operation.

**D — FDA in isolation, smaller β=0.01 (~50 min training run)**
Test Fourier augmentation without C-DANN to see if the mechanism has merit independently, and with a much smaller swap fraction.

**E — 10-seed sweep of Exp 4 (~8 hrs)**
Exp 4 was run once (seed 42). Single-seed results have high variance — baseline seeds ranged 0.147–0.209. Need the mean to know if 0.2626 is representative.

---

## Questions for You

- What do you know about D4 as a recording environment? The 223 Hz spectral peak and near-zero performance across all species suggests something unusual — equipment coloration, background insects, or a fundamentally different microphone?
- Have you seen domain adaptation succeed for species with truly zero cross-domain training data (like An. dirus)? Is there a principled lower bound on what's achievable?
- Any experience with FDA or RIR augmentation for bioacoustics specifically?
- Is prototype / nearest-centroid inference commonly used in bioacoustic challenges? Any reason it would be against the spirit of the challenge?
- Is there any additional metadata for the training clips (recording equipment, GPS, weather, time of day) that could serve as weak supervision for domain adaptation?
