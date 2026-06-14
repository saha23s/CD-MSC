"""Per-species seen/unseen balanced accuracy for balanced_dann_dicl_proj128_tau02."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
REPO     = Path(__file__).resolve().parents[2]
ROOT     = REPO / "technical_report_assets_current_split"
CSV      = ROOT / "per_species_balanced_dann_dicl_proj128_tau02.csv"
OUT_DIR  = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)
PAPER_OUT = REPO / "paper" / "figures"   # paper-ready copy (no in-figure title)
PAPER_OUT.mkdir(parents=True, exist_ok=True)
STEM       = "per_species_dann_dicl"
PAPER_STEM = "fig5_per_species"

# ── data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)

labels     = df["species_short"].tolist()
seen_mean  = df["BA_seen_mean"].to_numpy()
seen_std   = df["BA_seen_std"].to_numpy()
unseen_mean = df["BA_unseen_mean"].to_numpy()
unseen_std  = df["BA_unseen_std"].to_numpy()
dsg_mean   = df["DSG_mean"].to_numpy()

n = len(labels)
x = np.arange(n)
w = 0.35  # bar half-width

# ── colorblind palette (matches existing figures: steel-blue / orange) ────────
C_SEEN   = "#4878CF"   # muted blue
C_UNSEEN = "#EF8636"   # muted orange
C_DSG    = "#6ACC65"   # green for DSG line

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(11, 4.8), constrained_layout=True)

# bars
bars_seen   = ax1.bar(x - w/2, seen_mean,   w, yerr=seen_std,
                      label="Seen",   color=C_SEEN,   capsize=4,
                      error_kw=dict(lw=1.2, capthick=1.2), zorder=3)
bars_unseen = ax1.bar(x + w/2, unseen_mean, w, yerr=unseen_std,
                      label="Unseen", color=C_UNSEEN, capsize=4,
                      error_kw=dict(lw=1.2, capthick=1.2), zorder=3)

ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.set_ylabel("Balanced Accuracy", fontsize=12)
ax1.set_ylim(0, 1.12)
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax1.tick_params(axis="y", labelsize=10)
ax1.set_xlabel("Species", fontsize=12)
ax1.spines["top"].set_visible(False)

# light horizontal gridlines behind bars
ax1.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax1.set_axisbelow(True)

# ── secondary axis: DSG ───────────────────────────────────────────────────────
ax2 = ax1.twinx()
ax2.plot(x, dsg_mean, color=C_DSG, marker="D", markersize=5,
         linewidth=1.8, linestyle="-", label="DSG", zorder=4)
ax2.set_ylabel("DSG  |BA$_{seen}$ − BA$_{unseen}$|", fontsize=11, color=C_DSG)
ax2.set_ylim(0, 1.12)
ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax2.tick_params(axis="y", labelsize=10, colors=C_DSG)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_color(C_DSG)

# ── legend ────────────────────────────────────────────────────────────────────
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2,
           fontsize=10, loc="upper right", framealpha=0.85)

# Title kept for the standalone technical-report asset; the paper version
# below is saved without it (the LaTeX caption carries the description).
TITLE = (
    "Per-Species Evaluation: balanced_dann_dicl_proj128_τ=0.2\n"
    "(LODO D1–D4, best checkpoint, error bars = std across folds)"
)

# ── save ──────────────────────────────────────────────────────────────────────
ax1.set_title(TITLE, fontsize=11)
for ext, kw in [("pdf", {}), ("png", {"dpi": 150}), ("svg", {})]:
    fig.savefig(OUT_DIR / f"{STEM}.{ext}", bbox_inches="tight", **kw)
    print(f"saved → {OUT_DIR / f'{STEM}.{ext}'}")

ax1.set_title("")  # paper convention: no in-figure title
for ext, kw in [("pdf", {}), ("png", {"dpi": 150}), ("svg", {})]:
    fig.savefig(PAPER_OUT / f"{PAPER_STEM}.{ext}", bbox_inches="tight", **kw)
    print(f"saved → {PAPER_OUT / f'{PAPER_STEM}.{ext}'}")

plt.close(fig)
