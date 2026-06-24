# CD-MSC Brainstorming & Research Notes

Branch: `aaron/preprocessing` | Companion to `PROGRESS.md`

---

## A Framework for Understanding Domain Shift

Domain shift in CD-MSC comes from three independent sources. Every technique
we try is attacking one or more of these:

**1. Frequency content**
Non-wingbeat frequencies carry domain information — background noise, ecological
soundscape, equipment hum, microphone frequency response. The model learns to
exploit these as shortcuts. D5 has a clean, flat noise floor. D1-D4 have variable
background sounds (birds, wind, rain, traffic) that differ per recording environment.

**2. Statistical distribution**
Even within the wingbeat frequency band, D5 features have different mean, variance,
and histogram shape than field features. The model trained on D5 statistics doesn't
recognise the same species when the feature distribution shifts.

**3. Temporal structure**
Within a single clip, some frames are cleaner signal than others. Field recordings
have intermittent noise bursts, variable background activity, and periods where the
mosquito is not flying. The model's mean+max pooling treats all frames equally,
diluting the embedding with noisy frames.

The real domain shift is all three simultaneously. A robust solution addresses all three.

---

## Preprocessing & Augmentation Techniques

Grouped by what they target and what they require.

### Group 1 — Frequency Content

**Mel Bin Masking**
Zero out or downweight mel bins outside the mosquito wingbeat band (~200–1500 Hz,
roughly bins 5–30 of 64). Removes non-wingbeat frequencies before the model sees them.
- Applies to existing pkl features — no re-extraction needed
- 30 minutes to implement
- Tests hypothesis: "Is the model exploiting non-wingbeat frequencies as domain shortcuts?"
- Risk: some species-discriminative harmonics above 1500 Hz may be clipped

**Bandpass Filter (raw audio)**
Apply 150–1500 Hz bandpass to raw audio before mel feature extraction. Audio-domain
version of mel bin masking. Removes low-frequency hum (traffic, equipment) and
high-frequency electronic noise.
- Requires re-running extract_features.py
- Fast to implement, needs re-extraction pipeline
- Natural complement to HPSS

**HPSS (Harmonic-Percussive Source Separation)**
Decompose audio into harmonic (tonal, periodic) and percussive (transient, broadband)
components. Keep only the harmonic component. Mosquito wingbeats are tonal — dominant
harmonic at ~300–800 Hz. Field background noise (wind, rain, traffic) is mostly
broadband percussive.
- Strongest theoretical fit for our task
- Requires raw audio + re-running extract_features.py (~100ms overhead per clip)
- Conceptually subsumes mel bin masking — HPSS selectively keeps harmonics everywhere,
  not just in a fixed band
- Implementation: librosa.effects.hpss(y) → keep y_harmonic

### Group 2 — Statistical Distribution

**CMN (Cepstral Mean Normalisation)**
Subtract per-clip time-axis mean in mel space. Removes microphone and channel DC
offset effects that vary by recording device.
- Applies to existing pkl features
- 30 minutes to implement
- Lower expected impact than histogram matching but zero cost to try
- Risk: removes low-frequency content that may carry species signal

**PCEN (Per-Channel Energy Normalisation)**
Replaces log compression in feature extraction. Uses an exponential moving average
to normalise against the local noise floor — robust to variable background noise levels.
- Requires changing extract_features.py and re-extracting (new config_signature)
- librosa.pcen() drop-in replacement
- Natural companion to HPSS: HPSS isolates the harmonic signal, PCEN normalises
  its energy level

**Histogram Matching / Feature Distribution Alignment**
Match the mel-spectrogram statistics of D5 clips to D1-D4 domain statistics during
training augmentation. Two variants:
1. Mean/variance alignment: shift and scale each frequency bin of D5 clips to match
   D1-D4 mean and std. Computable from existing pkl statistics.
2. Full histogram matching: match per-bin CDF. More aggressive but preserves ordering.
Applies to existing pkl features. Probably the most impactful feature-space method.

### Group 3 — Temporal Structure

**Frame Gating (planned Exp 6)**
Learned per-frame attention weights gate each frame's contribution to the final
embedding. Implementation inside MTRCNNBranch:

```python
frame_scores = Linear(64 → 1)(branch_output)   # [B, T, 1] gate logits
gate = softmax(frame_scores, dim=1)             # normalised attention over time
embedding = sum(gate * branch_output, dim=1)    # weighted pooling
```

Trained end-to-end — no separate objective. Model learns to focus on wingbeat frames
and ignore variable background noise frames. Particularly important for long D4
field recordings where acoustic conditions vary throughout the clip.

### Group 4 — Domain Transfer Augmentation

**Gaussian Noise Injection**
Add random white/pink noise to D5 clips during training to simulate field SNR.
Simple, fast, applies to existing features. Low expected impact alone but useful
combined with other methods.

**Fourier Domain Adaptation (FDA — Yang & Soatto 2020)**
Swap low-frequency STFT amplitude of a D5 clip with a random D1-D4 clip; keep D5
phase. Result: D5 temporal structure + D1-D4 spectral "clothing." Well-studied in
vision domain adaptation. Audio analogue is less explored.
- Requires raw audio in training loop
- Eval can still use precomputed pkl
- Most principled domain transfer approach; high engineering cost

**Time Stretching / Speed Perturbation**
±5-10% playback speed variation. Broadens training distribution.
⚠️ CAUTION: wingbeat frequency IS the primary species feature. Stretching >10%
may shift a sample across species class boundaries. Use conservatively or skip.

---

## BirdNET and Pretrained Audio Models

**BirdNET** — designed for bird species identification, trained on bird vocalizations.
Not directly applicable to mosquito classification:
- Bird features are broadband, complex, melodic — opposite of narrow-band tonal wingbeats
- BirdNET cannot perform source separation or frequency extraction
- Its internal representations are optimized for avian acoustics

**The interesting version of this idea** — use BirdNET as a background characteriser:
Run BirdNET on each clip; high bird detection confidence → rich field soundscape → D4/D1
recording; near-zero confidence → quiet environment → D5. This gives a continuous
domain-proximity score without needing domain labels. Computationally expensive (271K
clips) but conceptually interesting for future work.

**PANNs / CNN14** — pretrained on AudioSet (~600 diverse sound categories including
insects). A much better pretrained backbone candidate than BirdNET for our task.
AudioSet's diversity means CNN14's features generalise across acoustic conditions.
Replace MTRCNN backbone with CNN14, keep the two-head DANN structure.

---

## D4: The Strange Domain

D4 is the smallest and hardest domain — only 200 clips total, lowest test accuracy
(Figure 2 of the baseline paper). Several things are unusual:

**Species composition:**
D4 contains An. dirus (40), An. minimus (51), An. stephensi (74), Ae. aegypti (22),
Cx. quinquefasciatus (13). Four entire species are absent (An. gambiae, An. arabiensis,
Ae. albopictus, Cx. pipiens). Three of the rarest, hardest species co-occur here.

**Almost all D4 clips went to the test set:**
- An. dirus D4: 40 clips → all 40 in test, zero in trainval
- An. stephensi D4: 74 clips → all 74 in test, zero in trainval
This appears deliberate. The challenge designers put the hardest unseen-domain clips
for the rarest species entirely in the test set.

**Long clips:**
D4 clips average 35-57 seconds per clip (Table II of the paper). D5 clips average
~0.6 seconds. D4 clips are real field recordings, not short wingbeat excerpts.

**Is it a labelling error?** Unlikely — the Oxford group designed this carefully.
The structure looks intentional.

**The contrastive learning hypothesis:**
D4 contains three acoustically similar Anopheles species recorded in the same
acoustic environment. An. dirus, An. minimus, and An. stephensi are all Anopheles —
similar genus, overlapping wingbeat frequency ranges, hard to distinguish. Having
them co-occur in D4 creates natural hard negatives: different species, same domain,
similar acoustics.

The Oxford baseline authors' companion paper (Ref [6]) is titled "contrastive
learning and distribution alignment" for this exact task. D4's structure — rare
co-occurring Anopheles species in a challenging acoustic environment — is exactly
the hard contrastive pair structure you need to train fine-grained inter-species
discrimination that's robust to domain shift.

---

## Contrastive Learning

### The Core Idea

Standard contrastive learning pulls same-class samples together in embedding space
and pushes different-class samples apart. For our task:

- **Positive pairs**: same species from different domains
  → forces embedding to ignore recording environment, keep species signal
- **Hard negatives**: different Anopheles species from the same domain (D4)
  → forces fine-grained discrimination between acoustically similar species

The domain variation in positive pairs becomes "noise" the representation must
learn to discard. The only consistent signal across same-species pairs from
different domains is the acoustic wingbeat signature — exactly what we want.

### Supervised Contrastive Loss (SupCon — Khosla et al. 2020)

The most directly applicable formulation. Uses all samples in a batch:

```
L_SupCon = -1/|P(i)| * Σ_{p∈P(i)} log [ exp(z_i·z_p / τ) / Σ_{a≠i} exp(z_i·z_a / τ) ]
```

Where:
- `z_i` = L2-normalised embedding for sample i
- `P(i)` = all other samples in the batch with the same species label
- `τ` = temperature (typically 0.07–0.2; lower = sharper, harder contrast)
- The denominator sums over ALL other samples (positives + negatives)

Positives (same species) increase the loss if they are far from the anchor.
Negatives (different species) increase the loss if they are close to the anchor.
The loss is zero only when same-species embeddings cluster tightly and
different-species embeddings are well separated.

### Integration with Our Existing Pipeline

Add SupCon as an auxiliary loss alongside the existing species CE and DANN domain loss:

```python
L = L_species_CE + λ_dann * L_domain + λ_con * L_supcon
```

No architectural changes needed — SupCon operates on the existing 32-dim embedding z.
Only change: add L2 normalisation of z before computing the contrastive loss
(separate from the classification head path, which uses unnormalised z).

```python
# In train_one_epoch, after getting embeddings:
z_norm = F.normalize(z, dim=-1)                          # L2 normalise
loss_con = supcon_loss(z_norm, species_labels)           # contrastive
loss_species = CE(species_logits, species_labels)        # classification
loss_domain = CE(domain_logits, domain_labels)           # adversarial
loss = loss_species - alpha * loss_domain + lambda_con * loss_con
```

### Domain-Aware Variant

Weight positive pairs by whether they cross domain boundaries:

```python
# Same species, DIFFERENT domain → higher weight (cross-domain positive)
# Same species, SAME domain → lower weight (within-domain positive)
cross_domain_weight = (domain_i != domain_j).float() * (1 - w) + w
```

This explicitly rewards the model for bringing together cross-domain same-species
pairs, making domain invariance a direct objective rather than an emergent property.

### Batch Composition Requirement

SupCon requires multiple species AND multiple domains per batch to form meaningful
pairs. Our existing joint species×domain WeightedRandomSampler already ensures this —
every batch contains field domain clips from multiple species. This is the ideal
setup for contrastive learning without any additional changes to the data loader.

### Temperature Tuning

Temperature τ controls how hard the contrast is:
- τ too low → only the nearest negative matters, gradient vanishes for easy cases
- τ too high → loss treats all negatives equally, hard negatives don't dominate
- Start with τ = 0.1–0.2 for our small embedding size (32-dim)

### What to Expect

SupCon directly addresses the core problem for An. dirus, An. minimus, An. stephensi:
these species need to be discriminated from each other despite similar acoustics and
limited domain diversity. The hard negatives from D4 (same domain, different Anopheles)
are exactly the pairs SupCon is designed to exploit.

For D5-trained species, SupCon provides a different benefit: the positive pairs
spanning D5 and D1-D4 directly encode the requirement "same species should look
the same regardless of recording environment."

---

## Recommended Experimental Phases

### Phase 1 — Diagnose the domain shift
Run each of these separately to understand which source of shift dominates:
1. Mel bin masking + C-DANN + balanced
2. Histogram matching + C-DANN + balanced
3. Frame gating + C-DANN + balanced

Results tell us: frequency content (1), statistics (2), or temporal structure (3)?

### Phase 2 — Combine what works
Stack Phase 1 winners. Add SupCon loss as an additional component.
Best candidate: mel bin masking + histogram matching + frame gating + C-DANN + balanced + SupCon

### Phase 3 — Source separation
HPSS + PCEN as the most principled frequency-domain approach.
Subsumes mel bin masking but requires re-extraction.

### Phase 4 — Full pipeline
HPSS + PCEN + frame gating + C-DANN + balanced batches + SupCon

### Phase 5 — Domain transfer augmentation
FDA as training augmentation on D5 clips.
Most aggressive domain adaptation; benefits from a known-good architecture.

---

## Fundamental Frequency (f0) as a Domain-Invariant Feature

### Why f0 is Special

f0 is the wingbeat rate — a **biological property of the insect** that doesn't change
based on recording device, room acoustics, noise floor, or geographic location. Every
other feature in our mel spectrogram is contaminated by domain. f0 is not. It is the
most inherently domain-invariant signal we have access to.

Mosquito f0 ranges (approximate, females):

| Species | f0 range |
|---|---|
| Ae. aegypti | ~450–500 Hz |
| An. gambiae | ~380–450 Hz |
| Cx. quinquefasciatus | ~350–500 Hz |
| An. dirus | ~300–450 Hz |

Ranges overlap substantially — f0 alone can't fully discriminate species — but it is a
strong prior, particularly for separating genus groups.

### The Within-Species f0 Variation Problem

Within a single species, f0 varies due to temperature, sex, age, and body size
(noted explicitly in the baseline paper). Temperature is the most relevant for our
domain shift problem: D5 is a controlled lab (constant temperature), D1-D4 are field
recordings with variable temperature. Roughly 5-10 Hz shift per °C — a 10-20°C
field/lab temperature difference could shift f0 by 50-200 Hz. For species whose
ranges already overlap, this matters.

This means raw Hz values are not directly comparable across domains. Two approaches
address this:

**1. Encode f0 in cents (logarithmic pitch scale)**
Convert Hz to cents relative to a reference frequency:
```
cents = 1200 * log2(f0 / f_ref)
```
If temperature shifts ALL species' f0 proportionally, their separation in cents space
remains constant — temperature-induced absolute Hz drift cancels out. This makes the
feature more robust to systematic domain-induced frequency shifts.

**2. Use mean + variance (Option 4 below)**
Mean f0 over a full clip smooths short-term variation within a clip. Combined with
voiced frame fraction and encoded in cents, it captures the species-typical wingbeat
rate while being tolerant of clip-to-clip variation.

### What We Need to Know First: Visualise f0 Across Domains

Before relying on f0 as a feature, we need to understand how much it shifts between
domains for the same species. Proposed visualisation:

- For each species, plot violin plots or box plots of per-clip mean f0 across every
  domain it appears in
- Also plot f0 distributions D5 vs. D1-D4 pooled — how much does the lab vs. field
  shift matter?
- Use `librosa.pyin()` on raw audio to extract f0 tracks; compute mean over voiced frames

This tells us definitively:
1. How separable species are in f0 space
2. How much domain shifts the f0 distribution per species
3. Whether mean f0 is informative or dominated by domain-induced noise

### Pipeline Options (Option 1 deferred)

**Option 2 — f0-guided adaptive mel bin masking**
Estimate f0 per clip and mask everything except a window around f0 and its first 3-4
harmonics. Adapts to the actual wingbeat frequency rather than a fixed band. Smarter
than fixed mel bin masking, especially for species at frequency extremes.

**Option 3 — f0 track as an additional feature channel**
Stack the f0 time series as an extra channel alongside the mel spectrogram. The model
explicitly sees "the wingbeat frequency is X Hz right now." Compact, domain-invariant
signal added to the existing 2D representation.

**Option 4 — f0 statistics as auxiliary species features (recommended starting point)**
Extract per-clip:
- Mean f0 over voiced frames (encoded in cents)
- Std of f0 over voiced frames (spread of wingbeat rate within clip)
- Voiced frame fraction (what proportion of frames have detectable wingbeat)

Feed these 3 scalars as additional inputs into the species classification head alongside
embedding z. No architectural changes needed — just concatenate to z before the species
linear layer: `[z (32-dim) ; f0_mean ; f0_std ; voiced_frac]` → species head.

Mean f0 in cents is the most directly interpretable species signal. Std captures
within-clip variation (nervous/active mosquito vs. steady flight). Voiced fraction
is a clip quality score — high fraction = reliable recording, low = mostly noise.

### Encoding Hz to Cents

The user raised the idea of encoding Hz to cents — this is correct and important.

**Why not raw Hz:**
- Linear Hz scale treats 50 Hz difference at 400 Hz the same as 50 Hz at 4000 Hz
- Temperature-induced shifts are proportional (multiplicative), not additive
- Species separation is better captured on a log scale

**Why cents:**
- 100 cents = 1 semitone; 1200 cents = 1 octave
- Equal perceptual intervals correspond to equal numeric differences
- If species A and B are 200 cents apart in D5, they remain ~200 cents apart in D4
  even if both shift by 10% due to temperature — the separation is preserved

**Implementation:**
```python
import librosa
f0_hz = librosa.pyin(y, fmin=100, fmax=1200)[0]  # f0 in Hz per frame
voiced = ~np.isnan(f0_hz)
f0_cents = 1200 * np.log2(f0_hz[voiced] / 440.0)  # cents relative to A=440
mean_f0_cents = np.mean(f0_cents)
std_f0_cents = np.std(f0_cents)
voiced_frac = voiced.mean()
```

A=440Hz is a convenient reference (musical A4, within mosquito f0 range) but any
fixed reference works since it only shifts the mean, not the variance or separability.

---

## Natural Technique Combinations

| Combo | What it addresses |
|---|---|
| HPSS + PCEN | Isolate harmonic signal (HPSS) + normalise its energy (PCEN) — natural pair |
| Mel bin masking + Histogram matching | Filter irrelevant frequencies + align remaining distributions |
| Frame gating + HPSS | HPSS cleans spectrogram, gating learns which cleaned frames are most informative |
| SupCon + C-DANN + balanced | Contrastive inter-species discrimination + adversarial domain removal + balanced exposure |
| FDA + C-DANN + balanced | Synthesise field-like training data + adversarially remove residual domain info |

---

## f0 Visualisation Results & Updated Experimental Priority
*(Added after running visualize_spectral_peak.ipynb — 2026-06-10)*

### What the visualisations showed

We ran two f0 approaches:

**Attempt 1 — pyin (fmin=100 Hz):** Detected many voiced frames but values were unreliable.
D1 values for Ae. aegypti (105 Hz), An. gambiae (137 Hz), Cx. quinquefasciatus (100 Hz)
all sat at the fmin floor — pyin was latching onto low-frequency background noise.

**Attempt 2 — pyin (fmin=300 Hz):** D5 completely disappeared. Zero voiced frames detected
in any D5 clip with fmin=300. This is contradictory — if D5 had real wingbeat signals at
400–800 Hz (as fmin=100 suggested), raising fmin to 300 should not erase them. Most likely
explanation: D5's apparent high voiced fraction at fmin=100 was pyin picking up low-frequency
noise, and the raw D5 audio clips may be sparse/brief when randomly cropped.

**Attempt 3 — Spectral peak via argmax on pkl features (no pyin):**
Directly find the dominant frequency bin in 200–1000 Hz from the precomputed log-mel features.
Much more interpretable. Key findings:

- **Cx. pipiens**: D1=633, D2=633, D5=595 Hz — nearly identical across three domains.
  Strongest evidence of a domain-invariant wingbeat signal in the dataset.
- **Ae. aegypti**: D1=558, D4=521, D5=595 Hz — consistent within ~7%.
- **D4 problem persists**: Cx. quinquefasciatus D4=223 Hz, An. minimus D4=223 Hz.
  223 Hz is near the band floor — D4 clips for these species are dominated by
  low-frequency noise. Ae. aegypti D4=521 Hz is an exception — real wingbeat signal.
- **D5 broad distribution** (400–1000 Hz, median=595 Hz) — multiple species, each
  contributing their own peak. Healthy spread.
- **D1/D2/D3 single-bin histograms** — sparse data (few species per domain), not an
  artefact of the method.

**Mean spectra (Figure 3)** were the most diagnostic figure. D5 (orange) has a clearly
different spectral shape from field domains: higher overall energy, cleaner harmonic peaks,
different low-frequency profile. Field domains look "flatter" with more low-bin energy.
This is a **spectral envelope shift** — the dominant source of domain shift visible in
the features we already have.

### Is f0/spectral peak worth adding as a model feature?

**Probably not as a priority.** The species where spectral peak is consistent
(Cx. pipiens, Ae. aegypti) are already the easier ones with more training data.
The hard cases (An. dirus, An. gambiae, An. arabiensis) have no cross-domain
validation or show 223 Hz noise in D4. Adding an auxiliary spectral peak head would
give wrong supervision for D4 clips of those species.

If pursued: add a scalar regression head on the 32-dim bottleneck predicting
normalised peak Hz, training only on clips where spectral peak > 300 Hz (mask out
noise). But risk-reward is poor given cheaper fixes available.

### D4 — Labelling error or intentional?

**Not a labelling error.** Ae. aegypti in D4 shows a real wingbeat signal at 521 Hz,
close to D1=558 and D5=595. A blanket labelling error would mean ALL species in D4
show noise — but they don't. D4 is a genuinely hard domain with variable recording
quality. Some species have audible wingbeat signal (Ae. aegypti), others are buried
in noise (Cx. quinquefasciatus, An. minimus). The contrastive learning hypothesis
(rare co-occurring Anopheles as hard negatives) remains the best explanation for
D4's structure.

### Updated Experimental Priority

The mean spectra figure directly motivates the ordering. D5 vs field spectral envelope
shift is **visible in the existing features** — cheap fixes first.

**Tier 1 — Feature-space, no re-extraction, implement today**

1. **CMN (Cepstral Mean Normalisation)** — subtract per-clip time-axis mean.
   Directly removes the spectral level difference between D5 and field visible in
   Figure 3. One line in `MosquitoFeatureDataset.__getitem__`.
   `feat = feat - feat.mean(axis=0)`

2. **Gaussian noise on D5 training clips** — add random noise to D5 features
   during training. Closes the "clean D5 vs noisy field" gap seen in Figure 4.
   One line: `feat += np.random.normal(0, noise_std, feat.shape)` for D5 clips only.

3. **Histogram matching** — align D5 mel bin statistics (mean + std per bin) to
   match D1-D4 statistics during training. Computable directly from pkl statistics.
   More principled than Gaussian noise but same cost.

Try CMN + Gaussian noise together as **Exp 6** (one experiment, two one-liners).
This is faster to implement than frame gating and addresses the root cause directly.

**Tier 2 — Model-level changes**

4. **Frame gating** — learned per-frame attention. Originally Exp 6, now Exp 7.
   Addresses D4 temporal noise problem. Architecturally interesting but more work.

5. **SupCon** — auxiliary contrastive loss. Pulls same-species embeddings together
   across domains. Directly leverages the D4 hard negative structure.

**Tier 3 — Requires re-extraction**

6. **HPSS** — removes broadband noise, keeps harmonic wingbeat. Strong theoretical
   fit. Would address the D4 noise problem at the feature level.

7. **PCEN** — better noise floor normalisation. Natural companion to HPSS.

**Tier 4 — Skip for now**

8. **FDA** — most principled but highest engineering cost.
9. **Spectral peak auxiliary feature** — marginal benefit, unreliable for hard cases.
