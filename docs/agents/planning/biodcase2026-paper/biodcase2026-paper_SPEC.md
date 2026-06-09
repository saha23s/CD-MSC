# SPEC: BioDCASE 2026 Workshop Paper

## 1. Overview

- **Goal:** Produce a camera-ready 4–6 page DCASE 2026 workshop paper with all LaTeX source, figure generation scripts, and bibliography.
- **Title:** *"Domain-Balanced Adversarial Training with Physics-Grounded Augmentation for Cross-Domain Mosquito Species Classification"*
- **Venue:** BioDCASE 2026 / DCASE 2026 Workshop. IEEE one-column format.
- **Problem Statement:** Existing DG methods applied naively to mosquito audio fail because one domain (D5) dominates 99.4% of training data, effectively starving the model of D1–D4 signal. We diagnose this as a prerequisite failure and propose a stacked system that fixes it.

---

## 2. Requirements

### Functional

- [ ] `paper/main.tex` compiles to PDF with `pdflatex` (or `latexmk`) without errors
- [ ] All 6 figure scripts run standalone from the repo root with `source .venv/bin/activate && python scripts/paper/figN_*.py`
- [ ] Each script saves `paper/figures/figN_name.{pdf,png,svg}` using the shared save helper
- [ ] Table 1 numbers match `plan.md` exactly (seed 42, D1–D4 mean)
- [ ] Paper fits within 6 pages including references (target: 5.5 pages)
- [ ] Abstract ≤ 150 words
- [ ] Every equation is numbered; every symbol defined on first use
- [ ] All citations resolve in `refs.bib`

### Non-Functional

- Colorblind-safe palette throughout (seaborn `colorblind`)
- Type-1 / vector fonts in all PDFs (no rasterized text)
- Figure captions: ≥ 2 sentences (what + takeaway)
- No hardcoded absolute paths in any script

---

## 3. Technical Constraints & Assumptions

- Style file: DCASE 2026 uses IEEE `IEEEtran` one-column. Use `\documentclass[a4paper]{IEEEtran}` with DCASE-specific overrides (download from DCASE official page; scaffold with a standard IEEEtran preamble until obtained).
- Python env: `.venv` already present; all deps (matplotlib, seaborn, numpy, pandas, sklearn, scipy) are installed.
- Variance ratio data: must be re-computed on-the-fly in `fig2_freq_variance.py` by loading the test-split feature pickle and computing within/cross-domain per-bin variance. Alternatively, hard-code from notes (bins 0–8: 0.30–0.93; bins 9–35: 1.50–4.12; bins 36–63: 0.73–1.47).
- t-SNE data: embeddings already saved in `technical_report_assets_current_split/` — reuse directly.
- Results data: all numbers come from `plan.md` table, not from re-running jobs.
- Author block: **placeholder** — `\author{Sulagna Saha \\ Mila -- Quebec AI Institute}`. Fill final names before submission.

---

## 4. Paper Content Specification

### 4.1 Abstract (≤ 150 words)

Structure: background (1 sentence) → problem statement (2 sentences) → method (2 sentences) → results (2 sentences) → impact (1 sentence).

Draft:
> Cross-domain mosquito species classification is a critical step toward scalable,
> low-cost disease vector surveillance. The BioDCASE 2026 challenge exposes a severe
> domain imbalance: one recording domain (D5) constitutes 99.4% of training data,
> rendering standard domain generalization methods nearly ineffective — and masking
> the true cross-domain gap in the official evaluation metric.
> We propose a stacked system: (i) domain-balanced sampling as a mandatory prerequisite,
> (ii) domain-adversarial training (DANN) over the balanced signal, (iii) domain-invariant
> contrastive learning (DiCL) with a projection head, and (iv) Frequency-Band-Selective
> Mix (FBS-Mix), a physics-grounded augmentation that selectively randomizes
> domain-dominated low-frequency bins while protecting the species-discriminative
> wingbeat band.
> On the LODO (Leave-One-Domain-Out) protocol, our system achieves BA_unseen = 0.378
> (+21 pp) and DSG = 0.202 (−0.210) over the MTRCNN baseline.
> These results demonstrate that addressing sampling bias is a prerequisite for
> domain generalization in severely imbalanced bioacoustic settings.

### 4.2 Introduction

Paragraphs:
1. Motivation — mosquito-borne disease, acoustic monitoring, deployment gap
2. Challenge task — 9 species, 5 domains, BA_unseen as metric, LODO as honest eval
3. Problem diagnosis — D5 dominance, why standard DG fails, why standard split misleads
4. Contributions (bulleted):
   - Diagnosis of domain imbalance as the primary failure mode
   - Stacked system: balanced sampling → DANN → DiCL → FBS-Mix
   - FBS-Mix: novel frequency-band-selective augmentation motivated by per-bin variance analysis
   - Negative results: TTBN, unbalanced DANN, MixStyle — benefit practitioners

### 4.3 Data & Problem Analysis

Subsections:
- **2.1 Dataset** — 9 species, 5 domains, Table A (domain clip counts: D1=458, D2=184, D3=158, D4=508, D5=212,339 training), clip duration statistics
- **2.2 LODO Evaluation Protocol** — definition, D5 exclusion rationale (212k → 1.3k training starvation), metrics BA_unseen/BA_seen/DSG
- **2.3 Why the Standard Split Misleads** — BA_unseen=0.175 on official split vs. LODO baseline 0.165; test set 84% D5-like
- **2.4 Frequency-Band Variance Analysis** — within/cross-domain variance ratio per mel bin; three zones (domain 0–8, species 9–35, mixed 36–63); motivates FBS-Mix

### 4.4 Proposed System

Subsections with equations:
- **3.1 Domain-Balanced Sampling** — WeightedRandomSampler, weight_i = 1/count(domain_i), effect: D1–D4 upsampled ~200–2600×
- **3.2 Domain-Adversarial Training (DANN)** — GRL, λ schedule Eq. (1): λ(p) = 2/(1+exp(−10p))−1 · λ_max, p = epoch/total. Loss: L = L_species − λ · L_domain
- **3.3 Domain-Invariant Contrastive Learning (DiCL)** — Eq. (2): InfoNCE with positives = same-species different-domain pairs; projection head 32→128→128; τ = 0.2. Combined loss Eq. (3).
- **3.4 FBS-Mix** — Algorithm 1 (pseudocode box): (a) compute masked mean/std on bins 0–8 for each sample in batch; (b) sample λ ~ Beta(0.1, 0.1); (c) interpolate statistics for bins 0–8 only; (d) restore padding zeros. Formal Eq. (4) for the bin-selective interpolation.

### 4.5 Experimental Setup

- MTRCNN backbone (3 branches, kernel sizes 3/5/7, 32-dim embedding, two classification heads)
- Training: Adam lr=1e-3, batch=64, early stop min_epochs=10 patience=5, max=100
- LODO: 4 folds (D1–D4), seed 42, D5 excluded
- DANN λ_max = 1.0; DiCL weight = 0.1, τ = 0.2, proj_dim = 128
- FBS-Mix: β(0.1, 0.1), bins 0–8 mixed, bins 9–35 protected
- Hardware: 1 × NVIDIA A100 (Mila cluster), ~27 min per fold

### 4.6 Results & Ablations

**Table 1 — Main Ablation (LODO D1–D4 mean, seed 42)**

| Method | BA\_unseen | Δ | BA\_seen | DSG |
|---|---|---|---|---|
| MTRCNN baseline | 0.165 | — | 0.578 | 0.412 |
| + Domain-balanced sampling | 0.321 | +15.6 | 0.511 | 0.245 |
| + Species-only (no domain head) | 0.324 | +15.9 | 0.547 | 0.267 |
| + MixStyle | 0.167 | +0.2 | 0.592 | 0.425 |
| + DANN (unbalanced) | 0.169 | +0.4 | 0.621 | 0.453 |
| Balanced + DANN | 0.368 | +20.3 | 0.540 | 0.276 |
| Balanced + DiCL | 0.321 | +15.6 | 0.547 | 0.262 |
| Balanced + DANN + DiCL | 0.340 | +17.5 | 0.517 | 0.247 |
| Balanced + DANN + DiCL + proj128 | 0.371 | +20.6 | 0.540 | 0.235 |
| **Balanced + DANN + DiCL + proj128 + τ0.2** | **0.378** | **+21.3** | **0.534** | **0.202** |
| Balanced + DANN + FBS-Mix | 0.345 | +18.0 | 0.516 | 0.229 |
| Best + augmentations (†) | — | — | — | — |

† Combination/delta/ScoL results pending; will be updated before camera-ready.

Analysis paragraphs:
1. Balance as prerequisite (Table 1 rows 1–5 comparison)
2. DANN synergy with balance
3. DiCL projection head geometry
4. FBS-Mix: best DSG/BA tradeoff among augmentations
5. Negative results: TTBN (−3 pp), AST (−5 pp BA_unseen vs MTRCNN balanced), MixStyle (null without balance)

### 4.7 Conclusion

- Summary: balance is prerequisite; DANN+DiCL+FBS-Mix achieve 0.378 BA_unseen (+21 pp)
- Limitations: single backbone (MTRCNN), single seed for ablations, opaque domain labels
- Future: FBS-Mix for AST (patch-level), TENT on eval set, compositional domain splits

---

## 5. Figure Specifications

### Fig 1 — Data Analysis (two panels, ~3.5" × 2.5")
- **Panel A:** Horizontal bar chart. X = number of clips (log scale). Y = domain labels D1–D5.
  Colors: D5 highlighted in orange-red, D1–D4 in blue. Annotation: "99.4%" on D5 bar.
  Data: `domain_distribution.json` from `technical_report_assets_current_split/`.
- **Panel B:** LODO fold schematic. 5 rows (D1–D5), 2 columns (train / test).
  Grid of colored squares. Held-out fold = red, training folds = blue, D5-excluded marker.
  Built with matplotlib patches — no external data needed.
- Caption: "Dataset domain imbalance (left) and LODO evaluation protocol (right). D5 contains 99.4% of training clips, making it unsuitable as a held-out domain; all reported LODO means exclude D5."

### Fig 2 — Frequency Variance Ratio (~3.5" × 2.0")
- X: mel bin index 0–63. Y: within-domain / cross-domain variance ratio.
- Single line plot. Three shaded regions:
  - bins 0–8: light orange, label "Domain-dominated"
  - bins 9–35: light blue, label "Species-dominated (wingbeat)"
  - bins 36–63: light gray, label "Mixed"
- Horizontal dashed line at y=1.0.
- Data: hard-coded from `note.md` variance analysis (sufficient for paper — full recompute is Fig script bonus).
- Caption: "Per-mel-bin within-domain vs. cross-domain variance ratio on the test split. Bins 9–35 (500–2.2 kHz) are consistently species-dominated — the wingbeat core. FBS-Mix exploits this structure."

### Fig 3 — System Diagram (~5" × 2.5")
- Left: spectrogram strip labeled "Input log-mel (64 × T)".
- Arrow → FBS-Mix block (shaded box): "Mix bins 0–8" annotation + "Protect bins 9–35".
- Arrow → MTRCNN block (3 parallel conv towers with kernel annotations 3/5/7).
- Arrow → embedding circle "z (32-dim)".
- Two arrows diverging: species head (→ "9-class softmax") and domain head via GRL (→ "5-class softmax").
- DiCL projection arc: z → "proj MLP" → contrastive loss symbol.
- All text in matplotlib patches; no Inkscape/tikz dependency.
- Caption: "Proposed system. FBS-Mix randomizes domain-dominated low-frequency statistics before feature extraction. DANN enforces domain-invariant embeddings via gradient reversal. DiCL pulls same-species embeddings across domains."

### Fig 4 — Ablation Bar Chart (~3.5" × 4.0")
- Horizontal bars, sorted by BA_unseen ascending (baseline at bottom, best at top).
- Color coding by method category:
  - gray: baselines (MTRCNN, species-only, MixStyle, DANN-unbalanced)
  - blue: balanced variants
  - teal: balanced + DANN
  - purple: balanced + DANN + DiCL variants
  - orange: augmentation variants (FBS-Mix)
- Secondary axis or inline annotation: DSG value.
- Vertical dashed line at baseline BA_unseen = 0.165.
- Caption: "LODO D1–D4 mean BA_unseen (seed 42). All methods above the dashed line improve over the unbalanced MTRCNN baseline. Domain-balanced sampling (+15.6 pp) is the dominant single gain."

### Fig 5 — t-SNE (two panels, ~5" × 2.5")
- Reuse `technical_report_assets_current_split/fig5a_tsne_domain.pdf` concept but:
  Panel A = baseline embeddings, Panel B = best system embeddings.
  Both domain-colored (5 colors from colorblind palette).
  Shared legend. Title: "Baseline" / "Balanced + DANN + DiCL + τ0.2".
- Data: need to check if both embedding TSNEs were saved; if only one, use that.
- Caption: "t-SNE of 32-dim embeddings, colored by domain. Baseline (left): D1–D4 collapse to an isolated corner. Best system (right): domain clusters intermix while retaining species structure."

### Fig 6 — Per-Species BA (~5" × 2.5")
- Grouped bar chart: 9 species on X. 3 bar groups: baseline, balanced, best.
- Colors from colorblind palette (3 shades).
- Species abbreviated to 4-letter codes (e.g., "Cp" = Cx. pipiens).
- Annotations: Cx. pipiens best generalizer (BA_unseen ≈ 0.804 in baseline).
- Data: from `technical_report_assets_current_split/per_species_official_best_model.json` + manually compiled LODO per-species (may need to pull from checkpoint eval JSONs).
- Caption: "Per-species balanced accuracy on LODO D1–D4 mean. Cx. pipiens is the only species with strong generalization in the baseline; the proposed method improves rarely seen species most."

---

## 6. refs.bib Entries

```bibtex
@article{biodcase2026,
  title   = {{BioDCASE} 2026 Challenge Baseline for Cross-Domain Mosquito Species Classification},
  author  = {[challenge authors]},
  journal = {arXiv:2603.20118},
  year    = {2026}
}

@inproceedings{mtrcnn2025,
  title     = {[MTRCNN full title]},
  author    = {[authors]},
  booktitle = {ICASSP},
  year      = {2025},
  doi       = {10.1109/ICASSP49660.2025.10890031}
}

@article{dann2016,
  title   = {Domain-Adversarial Training of Neural Networks},
  author  = {Ganin, Yaroslav and others},
  journal = {JMLR},
  volume  = {17},
  year    = {2016}
}

@inproceedings{mixstyle2021,
  title     = {Domain Generalization with {MixStyle}},
  author    = {Zhou, Kaiyang and others},
  booktitle = {ICLR},
  year      = {2021}
}

@article{drbiol2025,
  title   = {{DR-BioL}: [full title]},
  author  = {Hou and others},
  journal = {arXiv:2510.00346},
  year    = {2025}
}

@article{groupdro2020,
  title   = {Distributionally Robust Neural Networks},
  author  = {Sagawa, Shiori and others},
  journal = {arXiv:1911.08731},
  year    = {2020}
}

@inproceedings{tent2021,
  title     = {Tent: Fully Test-Time Adaptation by Entropy Minimization},
  author    = {Wang, Dequan and others},
  booktitle = {ICLR},
  year      = {2021}
}

@article{humbugdb2021,
  title   = {{HumBugDB}: A Large-scale Acoustic Mosquito Dataset},
  author  = {Kiskin, Ivan and others},
  journal = {arXiv:2110.07607},
  year    = {2021}
}

@article{mukundarajan2017,
  title   = {Using mobile phones as acoustic sensors for high-throughput
             mosquito surveillance},
  author  = {Mukundarajan, Haripriya and others},
  journal = {Science Translational Medicine},
  year    = {2017}
}

@inproceedings{supcon2020,
  title     = {Supervised Contrastive Learning},
  author    = {Khosla, Prannay and others},
  booktitle = {NeurIPS},
  year      = {2020}
}
```

---

## 7. Out of Scope

- Running new experiments to fill pending results — paper is written around existing numbers
- AST model results as a primary system — AST is an ablation / negative result only
- BioAcDG benchmark paper — separate document (`bioacoustic_dg_benchmark.md`), do not conflate
- Multi-seed confidence intervals in Table 1 — single seed (42) only; note this as a limitation

---

## 8. Acceptance Criteria

- [ ] `pdflatex paper/main.tex` exits 0 and produces a PDF
- [ ] PDF is ≤ 6 pages (including references)
- [ ] Table 1 has ≥ 10 rows and all numbers match `plan.md`
- [ ] All 6 figure scripts run and produce output in `paper/figures/`
- [ ] Abstract is ≤ 150 words
- [ ] `refs.bib` has ≥ 10 entries, all cited in text
- [ ] No LaTeX `\undefined` or `undefined reference` warnings in final compile

---

## 9. Dependencies

- `paper/` and `scripts/paper/` directories created
- `.venv` activated for all Python scripts
- `technical_report_assets_current_split/` — source for Fig 1 domain dist data, Fig 5 t-SNE, Fig 6 per-species data
- `plan.md` — authoritative source for all Table 1 numbers
- DCASE 2026 style file (IEEEtran-based) — scaffold with standard IEEEtran until official file obtained

---

## 10. Verification Plan

| Step | Verification command |
|---|---|
| LaTeX compiles | `cd paper && pdflatex main.tex 2>&1 \| grep -E "Error\|Warning"` |
| All fig scripts run | `for f in scripts/paper/fig*.py; do python $f && echo "OK: $f"; done` |
| Page count | `pdfinfo paper/main.pdf \| grep Pages` |
| Abstract word count | `detex paper/sections/intro.tex \| head -20 \| wc -w` |
| Table numbers match | Manual cross-check against `plan.md` |
