# BioAcoustic Domain Generalization Benchmark (BioAcDG) — Design Notes

*Status: Idea stage. Separate from CD-MSC challenge work.*  
*Last updated: 2026-06-04*

---

## Motivation

No standard bioacoustic DG benchmark exists analogous to DomainBed for vision. The CD-MSC challenge is the closest thing for mosquitoes, but:
- Domain labels (D1–D5) are opaque — no explicit device or location metadata per recording
- Single taxon (mosquito), single axis of domain shift
- D5 imbalance (99.4% of training) makes it a data starvation problem as much as a DG problem

A proper benchmark should:
1. Cover multiple taxa (generality across bioacoustic signals)
2. Have explicit, structured domain metadata on **two axes** — device and location — which are physically distinct sources of shift requiring different methods
3. Support controlled evaluation along each axis independently and jointly
4. Expose the physical structure of domain shift (see *Domain-Specific Insights* section) so method contributions can be mechanistically validated

---

## Core Design: 2D Domain Structure

Domain is not a single axis in bioacoustics. Two independent sources of covariate shift exist:

| Axis | What shifts | Example |
|---|---|---|
| **Device** | Spectral envelope, noise floor, frequency response, gain | Smartphone vs. SM4 recorder vs. lab microphone |
| **Location** | Background noise profile, reverberation, habitat acoustics | Rainforest vs. urban wetland vs. lab chamber |

These co-vary in practice (field recorder in forest vs. phone in suburb) but are conceptually separable and may require different invariance mechanisms.

**Key claim:** A benchmark with both axes explicitly labeled enables a richer evaluation than any existing DG benchmark, including DomainBed.

---

## Three Evaluation Modes

Given recordings indexed by `(device d, location l)` with species label `y`:

### Mode 1 — Device-DG
- **Train:** all locations, all devices except `d*`
- **Test:** device `d*` across all locations
- **Measures:** robustness to unseen recording hardware (spectral envelope shift)
- **Analogous to:** standard domain generalization

### Mode 2 — Location-DG
- **Train:** all devices, all locations except `l*`
- **Test:** location `l*` across all devices
- **Measures:** robustness to unseen habitat/background acoustics
- **Analogous to:** standard domain generalization, different axis

### Mode 3 — Compositional-DG *(novel)*
- **Train:** all `(d, l)` combinations except `(d*, l*)`
- **Test:** unseen combination `(d*, l*)`
- **Measures:** can the model decompose device and location factors and generalize to their unseen combination?
- **Why novel:** no existing benchmark tests compositional domain generalization. The model has seen device `d*` (at other locations) and location `l*` (with other devices) — can it combine these?

Mode 3 is the primary novel contribution of this benchmark design.

---

## Data Requirements

To support all three evaluation modes, each dataset must provide:

- [ ] Per-recording **device label** (structured, not free-text) — at minimum 3 distinct devices
- [ ] Per-recording **location label** — at minimum 3 distinct locations
- [ ] Sufficient `(device × location)` coverage — ideally ≥ 3×3 = 9 combinations with species overlap
- [ ] Species labels (closed-set within taxon)
- [ ] Audio recordings (WAV, consistent sample rate or resampleable)

---

## Domain-Specific Insights — Why Bioacoustic DG Is Physically Structured

*These are application-driven insights that distinguish bioacoustic DG from generic vision/NLP DG. Each is a potential demonstrable claim.*

The central argument: **bioacoustic domain shift decomposes along physically distinct axes, and this decomposition is measurable and method-relevant.** A benchmark with explicit device × location labels is the only way to expose this structure cleanly.

---

### Insight 1: Device Shift Is a Filter — Analytically Removable

A microphone is a Linear Time-Invariant (LTI) system. In the frequency domain:

```
X_recorded(f) = H_device(f) · X_animal(f) + N_environment(f)
```

`H_device(f)` is a slowly-varying multiplicative spectral envelope. In log-mel space this becomes **additive per-bin bias** — which per-clip CMVN (subtract mean per bin) directly removes.

**Implication:** Device shift is, in principle, *analytically removable* given correct normalization. This is fundamentally different from vision DG, where the domain transform has no clean physical interpretation. Learning-based DG methods applied to device shift are solving the residual of what CMVN doesn't catch — not the full problem.

**What to show:** CMVN alone closes a measurable fraction of the device gap. The residual is what DANN/DiCL are actually learning. A 2D benchmark lets you measure this decomposition: how much of the gap is analytically removable vs. requires representation learning.

**Connection to existing results:** `clip_normalize` in CD-MSC is exactly this — per-bin CMVN. Its effect on device-only vs. location-only folds can be measured with explicit domain labels.

---

### Insight 2: Location Shift Is Additive Noise — Physically Different Problem

Location shift is NOT a filter — it is **additive non-stationary noise** in the linear (power) domain:

```
X_recorded(f,t) = X_animal(f,t) + N_location(f,t)
```

Noise types have distinct spectral profiles: wind (broadband), traffic (low-frequency), dawn chorus (species-specific tonal), water (broadband mid). CMVN does not help — noise is additive, not multiplicative.

**Implication:** Methods designed for device shift (CMVN, style mixing of spectral envelopes like MixStyle) are physically wrong for location shift. They solve different problems. A 1D "domain" label confounds both mechanisms and makes it impossible to know which a method is solving.

**Connection to existing results:** FBS-Mix's variance ratio analysis on CD-MSC independently rediscovered this: mel bins 0–8 (domain-dominated = noise floor) respond to style mixing; bins 9–35 (species signal) must be protected. The physical reason is that bins 0–8 are where additive location noise dominates SNR.

**What to show:** CMVN improves device-DG but not location-DG. HPSS (harmonic–percussive separation) improves location-DG but not device-DG. A 2D benchmark makes this dissociation measurable — a result impossible to demonstrate on any existing single-axis benchmark.

---

### Insight 3: Signal Bandwidth Determines Device Sensitivity — Taxa-Dependent by Physics

Device shift severity depends on where a species' calls fall relative to device frequency response range:

| Taxa | Call frequency range | Device sensitivity | Notes |
|---|---|---|---|
| Mosquitoes | 100–1000 Hz | Low | All consumer mics respond well here |
| Frogs | 100 Hz – 5 kHz | Low–medium | Some rolloff at high end |
| Birds | 200 Hz – 10 kHz | Medium | High-freq rolloff varies strongly by device |
| Bats | 20–200 kHz | Extreme | Requires specialized detectors; nonlinear transforms across detector types |

Bat echolocation across detector types (heterodyne, time-expansion, frequency-division) is not even a simple filter — it is a nonlinear frequency compression. Methods that work for mosquito device shift will almost certainly fail for bats.

**What to show:** A multi-taxa benchmark spanning this bandwidth range would reveal that device-DG difficulty scales predictably with the call frequency / device response mismatch. This is a taxa-specific finding that only appears in a multi-taxa benchmark — and it is a testable physical prediction, not an empirical observation.

---

### Insight 4: Background Frames Are Free Domain Calibration Data

In passive acoustic monitoring, the majority of any recording is background (no animal vocalizing). These silent frames contain pure domain signal — `H_device(f) · N_location(f,t)` — with no species information.

**This is unique to bioacoustics.** In ImageNet or text benchmarks there is no analogous "pure domain, zero content" signal available at test time. In bioacoustic recordings it is always present and trivially segmentable (energy threshold or VAD).

**What to show:** A test-time adaptation method that estimates the device spectral envelope and location noise statistics *from background frames only*, then applies corrections prior to classification. This is source-free domain adaptation with a principled physical justification — no labels needed at test time, only the silent segments that are always available in PAM deployments. Demonstrating this on a benchmark with explicit device/location labels allows verification that the calibration is targeting the correct axis.

**Practical value:** This is directly deployable — any PAM system has background frames by definition. A method that uses them for calibration requires zero additional data collection or annotation.

---

### Insight 5: Temperature Is a Biological Domain Axis (Ectotherms Only)

For frogs and insects, call frequency and call rate drift with ambient temperature (Q10 effect: roughly +10% frequency per +10°C for many species). This is a **biological domain shift** — entirely orthogonal to device and location, caused by physiology rather than recording conditions.

**What to show:** A model trained on warm-season frog recordings failing on cold-season recordings of the same species, at the same location, with the same device. This is a third domain axis unique to ectotherm bioacoustics. No vision or general audio DG benchmark has this.

**Feasibility:** Requires seasonal recordings with temperature metadata. Viable with existing frog datasets (AnuraSet, some DCASE data) if temperature/season metadata is available. Lower priority than Insights 1–4 but worth documenting as a future axis.

---

### Which Insights Are Immediately Demonstrable

| Insight | Required data | Feasibility | Priority |
|---|---|---|---|
| 1. Device = LTI filter, CMVN closes it | 2D benchmark + CMVN ablation | High — demonstrable on HumBugDB now | **P1** |
| 2. Location ≠ filter, needs different methods | Same 2D benchmark, HPSS vs CMVN | High — demonstrable on HumBugDB now | **P1** |
| 3. Bandwidth → device sensitivity (multi-taxa) | Birds + bats needed | Medium — needs dataset survey | **P2** |
| 4. Background frames as calibration | Any PAM dataset | High — implementable on HumBugDB | **P1** |
| 5. Temperature as biological axis | Seasonal frog data with temperature | Low — needs new metadata | **P3** |

**P1 insights (1, 2, 4) together form the core demonstrable claim:** bioacoustic domain shift has physical structure, the structure is measurable with the right benchmark design, and it enables physically-grounded methods (CMVN for device, noise suppression for location, background calibration for both) that complement learning-based DG.

---

## Related Work — Bioacoustic Domain Gap Field Survey

*(Surveyed 2026-06-04)*

The literature clusters into four framings of the same underlying problem:

### 1. Focal → Passive Recording Gap *(most studied)*

Training on citizen science focal recordings (Xeno-canto: one mic, target species foregrounded, high SNR) but deploying on passive acoustic monitoring (PAM) soundscapes (omnidirectional mic, overlapping species, variable SNR).

- **ProtoCLR** — Moummad et al., arXiv:2409.08589 (2024). Birds. Supervised contrastive learning enforcing domain invariance across focal/passive pairs; prototypical comparisons instead of pairwise SupCon. Evaluated on BIRB benchmark.
- **"Birds, Bats and Beyond"** — Frontiers Bird Science, 2024. Systematic evaluation of BirdNET/Perch across taxa. Key finding: focal→soundscape gap is the dominant failure mode; device/mic type are secondary (frequency response, polar pattern, SNR differences).
- **"Studying Domain Shifts with Information Theory"** — Perry et al., SSRN 2025. Uses EGCI (entropy-based acoustic complexity index) to quantify focal/soundscape domain shift. SVM separates focal vs. soundscape recordings at 81% accuracy from acoustic complexity alone.

### 2. Few-Shot + Domain Adaptation *(DCASE lineage)*

Detect/classify previously unseen species from 5 examples, across recordings from different environments. Domain shift is a nuisance variable, not the primary target.

- **DCASE Task 5** (2022–2024) — Few-shot bioacoustic event detection. 66 eval files from 8 subsets (birds, bats, marine, insects). F1 improved 40→63% over 3 years. Domain adaptation via negative hard sampling and transductive learning.
- **"Mind the Domain Gap"** — arXiv:2403.18638 (2024). Systematic analysis of domain gap in the DCASE few-shot setting. Proposes negative hard sampling + transductive learning for prototypical networks.

### 3. Foundation Models as Domain-Agnostic Encoders

Pre-train on massive multi-taxa, multi-condition audio and rely on scale to achieve implicit domain invariance. DG is evaluated as a downstream property, not explicitly designed for.

- **AVEX / "What Matters for Bioacoustic Encoding"** — arXiv:2508.11845. 26 datasets, in-dist + OOD evaluation. Finding: self-supervised pre-training + supervised post-training on mixed bioacoustic + general audio corpus gives best OOD generalization. Data diversity > architecture choice.
- **NatureLM-audio** — arXiv:2411.07186 (2024). Audio-language foundation model. Zero-shot on unseen taxa. Transfers from speech/music domains to animal vocalizations.
- **Perch/BirdNET** — Transfer to marine (SurfPerch). Geography and device shift handled implicitly at scale.

### 4. Taxon-Specific DG *(directly relevant)*

Closed-set species classification under explicit domain shift, treated as the primary problem. Smallest cluster — almost entirely mosquito so far.

- **DR-BioL** — Hou et al., arXiv:2510.00346 (2025). Mosquito. DiCL + SdaL. Only paper that treats bioacoustic DG as a first-class problem with structured domain splits and a DG-specific metric.
- **CD-MSC / BioDCASE 2026** — arXiv:2603.20118. The challenge itself — first competition with a DG-focused metric (BA_unseen, DSG).

### Field Gap Summary

| Framing | Domain splits explicit? | DG as primary task? | Multi-taxa? | Benchmark reusable? |
|---|---|---|---|---|
| Focal→Passive | Implicit (recording type) | No (few-shot primary) | Yes | Partially |
| DCASE few-shot | No (random) | No | Yes | Yes (DCASE protocol) |
| Foundation models | No | No | Yes | No standard protocol |
| Taxon-specific DG | **Yes** | **Yes** | No (mosquito only) | Partial (CD-MSC) |

**The gap this benchmark would fill:** explicit domain splits + DG as primary task + multi-taxa. Nothing currently does all three together. The information-theoretic framing (cluster 1) and focal→passive gap are mature enough that reviewers will ask whether the benchmark captures them — a benchmark spanning both device domains and location/recording-style domains addresses both concerns.

---

## Candidate Datasets

### Mosquito
| Dataset | Device metadata | Location metadata | Notes |
|---|---|---|---|
| **HumBugDB** (Kiskin et al. 2021, arXiv:2110.07607) | `device_type` + `mic_type` per recording ✓ | `country`, `place`, `location_type` per recording ✓ | All 9 CD-MSC species present. 6 sites, 5 countries. **BUT device × location perfectly confounded** — each site used one device exclusively. Mode 3 not feasible. Valid for single-axis LODO (6 domains). |
| **CD-MSC** (BioDCASE 2026) | Opaque domain label only ✗ | Opaque domain label only ✗ | Cannot support 2D evaluation without dataset authors releasing metadata. |

### Birds
| Dataset | Device metadata | Location metadata | Notes |
|---|---|---|---|
| **Xeno-canto** | Free-text (inconsistent) ~✗ | GPS coordinates ✓ | Massive scale. Device field unreliable for structured use. |
| **BirdCLEF** (Kaggle/LifeCLEF) | Partially known | Location known | Year-over-year variation could be treated as a third axis |

### Frogs
| Dataset | Device metadata | Location metadata | Notes |
|---|---|---|---|
| **AnuraSet** | Unknown — needs check | Brazilian sites ✓ | Multi-species, consistent recording protocol |

### Bats
| Dataset | Device metadata | Location metadata | Notes |
|---|---|---|---|
| **ChiroVox** | Unknown — needs check | Unknown — needs check | Echolocation calls, very different signal type |

### Marine mammals
| Dataset | Device metadata | Location metadata | Notes |
|---|---|---|---|
| **DCLDE datasets** | Hydrophone type ✓ (likely) | Ocean location ✓ | Different enough domain shift to be interesting |

---

## Key Open Questions

1. ~~**HumBugDB metadata schema**~~ — **Verified (2026-06-04).** Has `device_type`, `mic_type`, `country`, `place`, `location_type` per recording. All 9 CD-MSC species present. However, device × location are perfectly confounded (each site used one device). Mode 3 (Compositional-DG) not feasible with HumBugDB alone. Usable for single-axis LODO (6 domains).

2. **Label harmonization** — Species label spaces don't overlap across taxa. Options:
   - *Within-taxa evaluation* (preferred): evaluate per-taxon, aggregate across taxa (e.g. mean LODO BA_unseen). Avoids label harmonization entirely.
   - *Family/genus coarsening*: collapse to higher taxonomic level for cross-taxa comparison. Loses species-level signal.
   - *Task reformulation*: open-set or verification rather than closed-set classification.

3. **What counts as "device"?** — Smartphone model vs. recorder model vs. just "field vs. lab"? Granularity matters for Mode 3 coverage.

4. **Minimum coverage for Mode 3** — How many `(device × location)` combinations are needed for a meaningful compositional hold-out? Likely ≥ 3×3 with ≥ 50 recordings per cell.

5. **Evaluation metric** — BA_unseen (used in CD-MSC) works per-mode. Need to define how to aggregate across modes and across taxa for a single benchmark score.

---

## Candidate Paper Venue

- **NeurIPS Datasets & Benchmarks track** — primary target. Requires: dataset(s), standardized splits, baseline results, clear task definition.
- **ICML workshop on DG** — lighter-weight venue for early version.
- **DCASE / BioDCASE workshop** — domain-specific, lower bar, good for establishing the protocol before a full benchmark paper.

---

## Immediate Next Steps (before committing scope)

1. ~~**Read HumBugDB paper**~~ — Done. Has device + location fields but device × location confounded. Supports single-axis LODO only.
2. **Decide on 2D benchmark scope** — Two paths: (a) drop 2D design, build multi-taxa single-axis DG benchmark using HumBugDB + birds + frogs; (b) keep 2D as desideratum, frame current as single-axis + recommend 2D data collection to community.
3. **Check AnuraSet and BirdCLEF** — metadata coverage for device + location fields.
4. **Decide on label harmonization strategy** — within-taxa evaluation is cleanest; validate this is acceptable for the target venue.
5. **Survey DCASE few-shot datasets** — assess whether any have structured device × location coverage sufficient for Mode 3.

---

## Relationship to Current CD-MSC Work

- CD-MSC work (LODO, balanced_dann, FBS-Mix, DiCL) is the **methods contribution** — independent of this benchmark.
- If a 2D benchmark paper is built, CD-MSC results become one evaluation within it (once authors release explicit device/location metadata, or this is treated as "opaque domain label" baseline).
- The two papers are complementary: methods paper (CD-MSC) + benchmark paper (BioAcDG). Do not block one on the other.
