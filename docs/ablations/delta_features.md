# Delta Features Ablation — Temporal Dynamics from Spectrogram Gradients

**Experiment tag:** `_delta`  
**Config key:** `use_delta: true`  
**Feature shape:** `[T, 128]` (was `[T, 64]`)  
**Model key:** `model_n_mels: 128` (set automatically in `train.py` when `use_delta=True`)

---

## Origin: Applying HOG to Audio

The idea came from asking whether **Histogram of Oriented Gradients (HOG)** — a classical computer vision feature descriptor (Dalal & Triggs, CVPR 2005) — could be applied to spectrograms.

HOG works by computing the gradient magnitude and direction at each pixel, then aggregating them into orientation histograms over local spatial patches. It captures *edge and texture information* independent of absolute brightness level, which made it extremely powerful for pedestrian detection and general object recognition before deep learning.

The key insight for transfer to audio: **spectrograms are 2D images**, with:
- Rows ↔ time frames
- Columns ↔ frequency bins
- Pixel intensity ↔ log-energy at that time-frequency point

So spatial gradient concepts from image processing have natural analogues in the time-frequency plane.

---

## What We Actually Implemented: Option B (Temporal Delta)

Full HOG on spectrograms (Option A — computing oriented gradients in local patches and binning them) would require a significant pipeline change and lose the simple interpretability of mel features. We chose Option B: a simpler but principled adaptation.

**Temporal delta**: compute the first-order time-axis difference of the mel spectrogram and concatenate it as an additional feature channel:

```
delta[t, f] = mel[t+1, f] - mel[t-1, f]   (central difference; frame 0 uses prepend approximation)
```

Result: instead of `[T, 64]`, the model sees `[T, 128]` — the first 64 columns are the original mel spectrogram, the next 64 are the rate of change of each mel bin over time.

### Implementation: where delta fits in the normalization pipeline

Delta is computed *after* global z-score normalization (`_normalize`). This is deliberate: the z-score subtracts a global mean per frequency bin. Since delta is a difference, that constant offset cancels exactly:

```
delta[t, f] = norm(mel)[t+1, f] - norm(mel)[t-1, f]
            = (mel[t+1,f] - μ_f)/σ_f - (mel[t-1,f] - μ_f)/σ_f
            = (mel[t+1,f] - mel[t-1,f]) / σ_f
```

No separate delta-specific normalization statistics are needed. The model's `input_bn` (BatchNorm2d) handles any residual scale differences between the mel and delta channels.

### Model adaptation

`train.py` automatically sets `config["model_n_mels"] = config["n_mels"] * 2 = 128` when `use_delta=True`. The `frequency_projection` layer inside each `MTRCNNBranch` infers its input dimension via `_infer_frequency_bins(n_mels=128)` — no other architecture changes are needed. The embedding `Linear(192, 32)` remains unchanged because each branch always outputs `[B, 64]` regardless of mel width.

---

## Core Hypothesis: Spectral Envelope vs Temporal Dynamics

### Spectral envelope

The shape of the spectrum at any given moment — which frequency bins carry how much energy. In mosquito audio, this corresponds to the harmonic structure of the wingbeat: the fundamental frequency (species-specific, ~300–800 Hz) and its overtones.

**Why the envelope shifts across domains:**

A lab recorder (D5) in a quiet room with close mic placement captures a clean, bright spectrum. A field recorder (D1–D4) captures the same mosquito but with:
- Background noise smearing energy across frequency bins
- Distance effects attenuating high frequencies (air absorption rolls off at roughly 6 dB per doubling of distance at high frequencies)
- Room/outdoor reflections adding reverberation
- Microphone hardware differences changing the frequency response curve

So the same wingbeat at 600 Hz looks spectrally "different" in D5 vs D1, even though the mosquito is identical. A model that learns "bright high-frequency content → this species" will fail when tested on field recordings where that brightness relationship doesn't hold.

### Temporal dynamics

How fast each frequency bin is *changing* from frame to frame. Properties captured by this include:
- Wingbeat periodicity (the oscillation rate of the energy envelope over time)
- Onset and offset sharpness of wingbeat pulses
- Modulation patterns in the harmonic structure

**Why this is more domain-invariant:**

A microphone that adds 5 dB of gain across the board shifts the envelope but leaves the differences between consecutive frames unchanged — the additive offset cancels in the difference. A stationary background noise floor similarly cancels. This makes temporal dynamics robust to two specific types of domain shift:

1. **Additive offset** — constant background noise level
2. **Multiplicative gain** — microphone sensitivity / recording level differences

**The ECG analogy:** Absolute heart rate level corresponds to the spectral envelope. The rhythm pattern — the shape of each beat, regularity of intervals — corresponds to temporal dynamics. Two ECG machines with different calibration give different absolute readings but the same rhythm. Recording conditions in audio are partly like miscalibrated ECG machines.

### Limitations of delta features

Delta features are *not* robust to all domain shift types:

- **Reverberation (convolutive channel effects):** Distorts the temporal shape of each transient — it smears onset and offset edges, which shows up as changed delta values even in signal frames. This is likely present in field recordings.
- **Fluctuating background noise:** Non-stationary noise (wind gusts, rain bursts) creates spurious deltas in frames that should be silent.

So this ablation is really asking: *how much of the CD-MSC domain shift is additive/gain-type vs convolutive/reverberant type?* If most shift is the former, delta features should help. If most is reverberant, they won't.

---

## Experiment Setup

**Recommended ablation configuration (cleanly isolated):**

```python
DANN_ALPHA_MAX  = 0.0    # no DANN — isolate delta contribution
CDANN           = False
BALANCE_BATCHES = True   # keep balanced sampling (proven helpful in Exp 4)
CMN             = False  # no CMN — isolate delta vs CMN separately
D5_NOISE_STD    = 0.0
USE_DELTA       = True
SEED            = 42
BATCH_SIZE      = 64
SPEC_AUGMENT    = False
```

This compares directly against the balanced-only baseline (same config but `USE_DELTA=False`).

**Output dir will be:**
```
outputs/MTRCNN_seed42_B64_E100_earlystop_min10_pati5_balanced_delta/
```

---

## How to Interpret Results

Compare against the balanced-only baseline (no DANN, no delta) as the clean reference point.

| Outcome | Interpretation |
|---|---|
| BAunseen ↑ vs balanced-only | Temporal dynamics carry domain-invariant species signal — delta features help |
| BAunseen ≈ balanced-only | Domain shift in this dataset is not primarily additive/gain-type |
| BAunseen ↓ vs balanced-only | Reverberant effects may dominate; deltas add noise more than signal |
| BAseen ↑ with BAunseen ↓ | Delta features add discriminative but domain-specific information |

A small improvement in BAunseen is still a positive signal — it tells us some portion of the shift is gain/offset-type and we can stack this with other techniques. A clear drop would suggest we should look at PCEN or HPSS instead (which target convolutive effects more directly).

---

## Connection to Broader Pattern: CV→Audio Feature Transfer

Spectrograms being 2D images means many classical CV feature engineering ideas have audio analogues. This is a useful heuristic to keep in mind:

| CV concept | Audio analogue |
|---|---|
| Image gradients (Sobel, HOG) | Temporal/spectral deltas |
| Gabor filters | Oriented spectro-temporal patches |
| SIFT local descriptors | Local time-frequency region features |
| Texture features (LBP) | Local spectral pattern histograms |

**General rule:** When stuck on audio feature engineering for domain adaptation, ask what the equivalent spatial operation would be for a 2D image under the same kind of domain shift, then map time↔rows and frequency↔columns.

**Next escalation if temporal delta helps but isn't enough:**
1. Add spectral delta (difference along frequency axis) for a third 64-dim channel → `[T, 192]`
2. Add delta-delta (second-order temporal difference) — standard in MFCC pipelines for the same reason
3. Try PCEN (per-channel energy normalisation) which handles convolutive effects that delta can't
