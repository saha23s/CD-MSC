# PLAN: BioDCASE 2026 Workshop Paper

## 1. Scope & Context

**Task:** Write a 4–6 page workshop paper for the BioDCASE 2026 / DCASE 2026 challenge
(Cross-Domain Mosquito Species Classification, Task 5).

**Primary metric:** `species_balanced_accuracy` on unseen domains (LODO BA_unseen, D1–D4 mean).
**Baseline result:** BA_unseen = 0.165.  **Best system:** BA_unseen = 0.378 (+21 pp, −0.210 DSG).

### Constraints

| Constraint | Detail |
|---|---|
| Page budget | 4–6 pages (DCASE one-column IEEE format) |
| Results available | LODO D1–D4, seed 42. Multi-seed still running (jobs 9733089, 9733096, 9733146). |
| Running experiments | delta features, ScoL, combo runs, random-split × 3 seeds — NOT yet in paper (include only when results land) |
| Existing assets | `technical_report_assets_current_split/` has figs 1–5b, t-SNE, per-species bars |
| Novel contribution | FBS-Mix (frequency-band-selective augmentation) — full variance-ratio motivation exists |

---

## 2. Narrative Architecture

The paper tells **one coherent story** in three beats:

> **Beat 1 — Diagnose:** The challenge is not just domain shift — it is *domain imbalance*.
> D5 holds 99.4 % of training data. Standard DG methods (DANN, MixStyle) are nearly
> useless because they never see D1–D4 during training. The standard evaluation
> metric (BA_unseen = 0.175) is also misleading because the test set is 84 % D5.

> **Beat 2 — Fix the prerequisite:** Domain-balanced sampling restores D1–D4 visibility
> (+15.6 pp alone). This is a *prerequisite*, not an ablation — everything else is
> built on top.

> **Beat 3 — Stack the methods:** With balance in place, DANN (+4.7 pp) and
> DiCL-proj128-τ0.2 (+1.0 pp further) push the system to BA_unseen = 0.378, DSG = 0.202.
> FBS-Mix (novel) is the physics-grounded augmentation: per-frequency variance analysis
> shows bins 0–8 are domain-dominated, bins 9–35 are species-dominated — so mix only bins 0–8.

This narrative makes the paper read as a principled progression, not a bag of tricks.

---

## 3. Section Outline (target page allocation)

| # | Section | Pages | Key content |
|---|---|---|---|
| 1 | Introduction | 0.5 | Task, challenge, contribution bullets |
| 2 | Data & Problem Analysis | 1.0 | Domain imbalance, LODO protocol, why standard split misleads, per-freq variance ratio |
| 3 | Proposed System | 1.25 | Balanced sampling, DANN, DiCL, FBS-Mix (with diagram) |
| 4 | Experimental Setup | 0.5 | LODO protocol, metrics (BA_unseen, BA_seen, DSG), configs |
| 5 | Results & Ablations | 1.5 | Main table + key ablations + per-species / t-SNE analysis |
| 6 | Conclusion | 0.25 | Summary, limitations, future work |
| Refs | References | ~0.5 | ~12 key citations |

---

## 4. Figure Plan

Six figures total (each in PDF + PNG + SVG from a dedicated script).

| Fig | Title | Source | Script | Reuse? |
|---|---|---|---|---|
| **Fig 1** | Dataset imbalance & evaluation protocol | Domain dist + LODO diagram | `fig1_data_analysis.py` | Partial reuse of `fig1_domain_imbalance.pdf` + new LODO fold diagram |
| **Fig 2** | Per-frequency variance ratio analysis | Variance ratio per mel bin | `fig2_freq_variance.py` | **New** — motivates FBS-Mix |
| **Fig 3** | System diagram (FBS-Mix + model) | Schematic | `fig3_system_diagram.py` | **New** — matplotlib schematic |
| **Fig 4** | Ablation bar chart | LODO results table | `fig4_ablation_bars.py` | **New** — horizontal bar chart of BA_unseen per method |
| **Fig 5** | t-SNE embeddings (baseline vs best) | Embeddings | `fig5_tsne.py` | Reuse `fig5a_tsne_domain.pdf` side-by-side |
| **Fig 6** | Per-species BA breakdown | Per-species analysis | `fig6_per_species.py` | Reuse `per_species_official_best_model_main_figure.pdf` adapted |

**Decision rationale:**
- Fig 1 captures Beat 1 visually.
- Fig 2 is the scientific justification for FBS-Mix — essential for novelty claim.
- Fig 3 makes the method self-explanatory without extensive prose.
- Fig 4 is the "money table" as a figure (more skimmable for workshop readers).
- Figs 5–6 go in Results to give analytical depth.

---

## 5. LaTeX Folder Structure

```
paper/
├── main.tex                  ← master document (includes all sections)
├── dcase2026.sty             ← DCASE/IEEE style file (to be downloaded)
├── refs.bib                  ← all citations (BibTeX)
├── sections/
│   ├── intro.tex
│   ├── data_analysis.tex
│   ├── method.tex
│   ├── experiments.tex
│   ├── results.tex
│   └── conclusion.tex
└── figures/                  ← generated PDFs/PNGs from scripts/paper/
    ├── fig1_data_analysis.{pdf,png,svg}
    ├── fig2_freq_variance.{pdf,png,svg}
    ├── fig3_system_diagram.{pdf,png,svg}
    ├── fig4_ablation_bars.{pdf,png,svg}
    ├── fig5_tsne.{pdf,png,svg}
    └── fig6_per_species.{pdf,png,svg}

scripts/paper/
├── fig1_data_analysis.py
├── fig2_freq_variance.py
├── fig3_system_diagram.py
├── fig4_ablation_bars.py
├── fig5_tsne.py
└── fig6_per_species.py
```

`paper/figures/` and generated outputs are **not committed** (gitignored).
Only `paper/*.tex`, `paper/refs.bib`, and `scripts/paper/*.py` go into version control.

---

## 6. References (.bib entries needed)

| Key | Paper | Where used |
|---|---|---|
| baseline2026 | BioDCASE 2026 CD-MSC Baseline (arXiv:2603.20118) | Intro, Experiments |
| mtrcnn2025 | MTRCNN (ICASSP 2025) | Method — backbone |
| dann2016 | Ganin et al. DANN (JMLR 2016) | Method — adversarial |
| mixstyle2021 | Zhou et al. MixStyle (ICLR 2021) | Method — ablation |
| drbiol2025 | DR-BioL / DiCL (arXiv:2510.00346) | Method — DiCL |
| groupdro2020 | Sagawa et al. GroupDRO (arXiv:1911.08731) | Ablation |
| tent2021 | Wang et al. TENT (arXiv:2006.10726) | Ablation |
| humbugdb2021 | Kiskin et al. HumBugDB (arXiv:2110.07607) | Related work |
| mukundarajan2017 | Mukundarajan et al. STM 2017 | Wingbeat freq priors |
| supcon2020 | Khosla et al. SupCon (NeurIPS 2020) | DiCL / ScoL background |
| fitzgerald2010 | Fitzgerald 2010 HPSS | HPSS method note |
| brogdon1994 | Brogdon 1994 (wingbeat freqs) | Wingbeat priors |

---

## 7. Key Design Decisions (explicitly stated)

1. **D5 exclusion in all tables.** Every LODO mean excludes D5 (data starvation, not domain shift). This must be explained once clearly and consistently referenced.

2. **Primary metric = LODO D1–D4 mean BA_unseen.** BA_seen and DSG are secondary columns. Do not report standard-split BA_unseen as a main result — only as context for "why standard split is misleading."

3. **FBS-Mix as the novel contribution.** All other methods (DANN, DiCL, balanced sampling) are prior work used as components. FBS-Mix has physics justification (wingbeat frequency band) + variance-ratio analysis + concrete implementation novelty (padding-aware statistics).

4. **Negative results to report explicitly:** TTBN harmful (−3 pp), MixStyle null without balance, AST hurts BA_unseen. These strengthen the narrative and prevent reviewers from asking "why didn't you try X?"

5. **Running experiments (combo, delta, ScoL, random-split).** Add a placeholder row in Table 1 labelled "Best + augmentations (†)" with a footnote "† results pending; will be updated before camera-ready." This is honest and does not block paper writing.

---

## 8. Granular Implementation Steps

Each step is independently completable and verifiable.

### Step 1 — Scaffold LaTeX
- Create `paper/` and `scripts/paper/` directories
- Write `paper/main.tex` skeleton (title, authors, abstract placeholder, `\input{}` all sections)
- Write `paper/refs.bib` with all 12 BibTeX entries
- Write stub `paper/sections/*.tex` files (title comment + `\section{}` only)
- **Verify:** `pdflatex main.tex` compiles without errors (empty sections OK)

### Step 2 — Write prose: Introduction + Data Analysis
- `intro.tex`: task statement, challenge severity, 3-bullet contribution list
- `data_analysis.tex`: domain distribution table, LODO protocol box, D5 exclusion rationale, per-frequency variance ratio narrative
- **Verify:** section reads coherently as standalone; no orphaned citations

### Step 3 — Write prose: Method
- `method.tex`: balanced sampling (paragraph), DANN + GRL schedule (paragraph + eq), DiCL projection head (paragraph + eq), FBS-Mix (paragraph + algorithm box or eq)
- **Verify:** every equation is numbered; every symbol defined on first use

### Step 4 — Write prose: Experiments + Results + Conclusion
- `experiments.tex`: LODO setup, hyperparams table, hardware
- `results.tex`: Table 1 (ablation), analysis paragraphs, pointer to figures
- `conclusion.tex`: 2-paragraph summary + limitations + future
- **Verify:** Table 1 matches numbers in plan.md exactly

### Step 5 — Figure scripts (Fig 1, 2, 4 first — these are in prose)
- `fig1_data_analysis.py`: two-panel (domain dist bar + LODO fold schematic)
- `fig2_freq_variance.py`: line plot of within/cross-domain variance ratio per mel bin (bins 0–63), shaded regions
- `fig4_ablation_bars.py`: horizontal bar chart, methods sorted by BA_unseen, color-coded by category (baseline / balance only / + DANN / + DiCL / + aug)
- **Verify:** each script runs standalone, saves to `paper/figures/`, checks out visually

### Step 6 — Figure scripts (Fig 3, 5, 6)
- `fig3_system_diagram.py`: matplotlib schematic of input → FBS-Mix split → model → heads
- `fig5_tsne.py`: side-by-side t-SNE (baseline vs best), domain-colored + species-marker
- `fig6_per_species.py`: grouped bar chart, per-species BA across baseline / balanced / best
- **Verify:** same as Step 5

### Step 7 — Wire figures into LaTeX + polish
- Add `\includegraphics` calls in correct sections
- Write captions (each ≥ 2 sentences: what it shows + key takeaway)
- Final pass: consistent notation, spell-check, page count check
- **Verify:** full PDF compiles, figures render, fits within 6 pages

### Step 8 — Abstract + final read
- Write 150-word structured abstract (background / problem / method / result)
- Read entire PDF as a reviewer would
- **Verify:** narrative is coherent, no dangling references

---

## 9. Next Step

> **Request approval to proceed to Phase 2 (SPEC)**, which will define exact figure
> content (axis labels, color palettes, data sources), the LaTeX preamble and style
> choices, and the full Table 1 layout with all numbers filled in.
