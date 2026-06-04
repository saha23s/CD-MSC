# References — BioDCASE 2026 CD-MSC

---

## Core / Baseline

- **BioDCASE 2026 CD-MSC Baseline** — arXiv:2603.20118  
  "BioDCASE 2026 Challenge Baseline for Cross-Domain Mosquito Species Classification"  
  ICASSP 2026. doi:10.1109/ICASSP55912.2026.11464393  
  *This repo is built directly on this baseline.*

- **MTRCNN** — doi:10.1109/ICASSP49660.2025.10890031  
  Multi-temporal-resolution CNN backbone. ICASSP 2025.  
  *Used as the primary classifier architecture (`framework/model.py`).*

---

## Domain Generalization / Adaptation

- **DANN** — Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.  
  GRL + λ annealing schedule.  
  *Implemented in `framework/gradient_reversal.py`.*

- **MixStyle** — Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.  
  Feature-level (μ, σ) style mixing between random batch pairs.  
  *Implemented in `framework/mixstyle.py`.*

- **DR-BioL** — Hou et al., arXiv:2510.00346 (2025).  
  Introduces DiCL (domain-invariant contrastive loss) and SdaL (species-conditional MMD). Same mosquito problem. Their DANN ablation: −0.45% (consistent with our D1: +0.5pp). τ=0.07 used (vs 0.01 in paper — more stable with small minority batches).  
  *DiCL and SdaL implemented in `framework/losses.py`.*

- **GroupDRO** — Sagawa et al., arXiv:1911.08731 (2020).  
  Per-domain exponential loss reweighting (η=0.01).  
  *Implemented in `framework/group_dro.py`.*

- **TENT** — Wang et al., arXiv:2006.10726 (2021).  
  Test-time entropy minimization via BN/LN affine adaptation.  
  *Implemented in `framework/tent.py`.*

---

## Bioacoustics / Wingbeat

- **Mukundarajan et al. (2017)** — Science Translational Medicine.  
  Per-species wingbeat frequency reference values.  
  *Used in `framework/metadata.py` for auxiliary regression head targets.*

- **Brogdon (1994)** — Wingbeat frequency literature values for mosquito species.  
  *Used in `framework/metadata.py`.*

- **Kiskin et al. (2020) — HumBugDB** — Mosquito audio dataset.  
  *Wingbeat frequency priors sourced from here (`framework/metadata.py`).*
