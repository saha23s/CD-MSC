"""
D4 species discrimination analysis.

All three D4 unseen-domain species (An. dirus, An. stephensi, An. minimus)
share F0 ≈ 300-307 Hz in D4. This script asks: is there anything ELSE
that distinguishes them?

Figures produced:
  1. d4_spectrogram_grid.png        — 5 example spectrograms per D4 species
  2. d4_mean_spectra.png            — mean mel spectra overlaid (D4 only)
  3. dirus_d1_vs_d4.png             — An. dirus training domain (D1) vs test domain (D4)
  4. d4_harmonic_power.png          — mean power spectrum 0-1500 Hz (harmonic content)
"""

import re
import random
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Config ─────────────────────────────────────────────────────────────────────
RAW_AUDIO_DIR = Path("Development_data/raw_audio")
N_EXAMPLES    = 5       # spectrogram examples per species
MAX_MEAN      = 40      # clips for mean spectrum computation
SR            = 8000
N_MELS        = 64
FMIN_MEL      = 100
FMAX_MEL      = 4000
CROP_S        = 2.0     # seconds to show in spectrograms
SEED          = 42
random.seed(SEED)

# Species of interest: name → 1-indexed species ID used in filename
D4_SPECIES = {
    "An. dirus":     6,
    "An. stephensi": 9,
    "An. minimus":   8,
}
COLORS = {
    "An. dirus":     "#e41a1c",
    "An. stephensi": "#984ea3",
    "An. minimus":   "#377eb8",
}

FILE_RE = re.compile(r"S_(\d+)_D_(\d+)_")


# ── Helpers ────────────────────────────────────────────────────────────────────
def find_files(species_sid: int, domain_did: int) -> list[Path]:
    prefix = f"S_{species_sid}_D_{domain_did}_"
    return sorted(f for f in RAW_AUDIO_DIR.iterdir()
                  if f.suffix.lower() == ".wav" and f.name.startswith(prefix))


def mel_db(path: Path, duration: float = CROP_S) -> np.ndarray | None:
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=duration)
        S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                            fmin=FMIN_MEL, fmax=FMAX_MEL)
        return librosa.power_to_db(S, ref=np.max)
    except Exception as e:
        print(f"  skip {path.name}: {e}")
        return None


def mean_mel(files: list[Path], max_clips: int = MAX_MEAN) -> np.ndarray | None:
    sample = random.sample(files, min(max_clips, len(files)))
    rows = []
    for f in sample:
        try:
            y, _ = librosa.load(f, sr=SR, mono=True)
            S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                                fmin=FMIN_MEL, fmax=FMAX_MEL)
            rows.append(S.mean(axis=1))
        except Exception:
            pass
    return np.mean(rows, axis=0) if rows else None


def mean_power_spectrum(files: list[Path], max_clips: int = MAX_MEAN,
                         fft_len: int = SR * 2) -> tuple:
    """Returns (freqs_hz, mean_magnitude) for 0-2000 Hz."""
    sample = random.sample(files, min(max_clips, len(files)))
    specs = []
    for f in sample:
        try:
            y, _ = librosa.load(f, sr=SR, mono=True, duration=CROP_S)
            if len(y) < fft_len:
                y = np.pad(y, (0, fft_len - len(y)))
            mag = np.abs(np.fft.rfft(y[:fft_len], n=fft_len))
            specs.append(mag)
        except Exception:
            pass
    if not specs:
        return None, None
    freqs = np.fft.rfftfreq(fft_len, d=1.0 / SR)
    return freqs, np.mean(specs, axis=0)


# ── Collect files ──────────────────────────────────────────────────────────────
print("Scanning audio directory …")
d4_files   = {name: find_files(sid, 4) for name, sid in D4_SPECIES.items()}
dirus_d1   = find_files(6, 1)
dirus_d4   = d4_files["An. dirus"]

for name, files in d4_files.items():
    print(f"  {name} D4: {len(files)} files")
print(f"  An. dirus D1: {len(dirus_d1)} files")

if not any(d4_files.values()):
    raise RuntimeError("No D4 audio files found. Check RAW_AUDIO_DIR path.")


# ── Figure 1: Spectrogram grid ─────────────────────────────────────────────────
print("\nFigure 1: spectrogram grid …")
n_rows = len(D4_SPECIES)
fig, axes = plt.subplots(n_rows, N_EXAMPLES, figsize=(N_EXAMPLES * 3, n_rows * 2.8))
fig.suptitle("D4 test-domain spectrograms — An. dirus vs An. stephensi vs An. minimus\n"
             f"({CROP_S}s window, log-mel, fmin={FMIN_MEL} Hz)",
             fontsize=11, fontweight="bold")

for row, (name, sid) in enumerate(D4_SPECIES.items()):
    files = d4_files[name]
    sample = random.sample(files, min(N_EXAMPLES, len(files)))
    for col in range(N_EXAMPLES):
        ax = axes[row, col]
        if col < len(sample):
            S = mel_db(sample[col])
            if S is not None:
                librosa.display.specshow(S, sr=SR, x_axis="time", y_axis="mel",
                                          fmin=FMIN_MEL, fmax=FMAX_MEL,
                                          ax=ax, cmap="magma")
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(labelsize=6)
            else:
                ax.text(0.5, 0.5, "load error", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="grey")
            ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(name, fontsize=8, color=COLORS[name], fontweight="bold")
        if row == 0:
            ax.set_title(f"ex {col+1}", fontsize=7)

plt.tight_layout()
plt.savefig("d4_spectrogram_grid.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → d4_spectrogram_grid.png")


# ── Figure 2: Mean mel spectra overlaid ────────────────────────────────────────
print("Figure 2: mean mel spectra …")
mel_bin_hz = librosa.mel_frequencies(n_mels=N_MELS, fmin=FMIN_MEL, fmax=FMAX_MEL)

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_title("Mean log-mel spectrum in D4 — three unseen-domain species\n"
             "Do their spectral envelopes differ?", fontweight="bold")

for name, sid in D4_SPECIES.items():
    files = d4_files[name]
    if not files:
        continue
    mm = mean_mel(files)
    if mm is not None:
        mm_db = librosa.power_to_db(mm, ref=np.max)
        ax.plot(mel_bin_hz, mm_db, color=COLORS[name], lw=2, label=f"{name} D4 (n={len(files)})")

# Also show An. dirus D1 for reference
if dirus_d1:
    mm_d1 = mean_mel(dirus_d1)
    if mm_d1 is not None:
        mm_d1_db = librosa.power_to_db(mm_d1, ref=np.max)
        ax.plot(mel_bin_hz, mm_d1_db, color=COLORS["An. dirus"], lw=1.5,
                ls="--", alpha=0.6, label=f"An. dirus D1 — training ref (n={len(dirus_d1)})")

ax.set_xlabel("Frequency (Hz)", fontsize=10)
ax.set_ylabel("Mean log energy (dB, normalised)", fontsize=10)
ax.legend(fontsize=9)
ax.axvspan(200, 400, alpha=0.08, color="gold", label="_F0 zone (~300 Hz)")
ax.axvspan(550, 700, alpha=0.06, color="lightblue", label="_2nd harmonic zone")
ax.axvspan(900, 1100, alpha=0.06, color="lightgreen", label="_3rd harmonic zone")
ax.set_xlim(FMIN_MEL, 2000)
ax.grid(True, alpha=0.3)

# Annotate harmonic bands
for f, label in [(300, "F0 ~300 Hz"), (600, "2nd harm."), (900, "3rd harm.")]:
    ax.axvline(f, color="grey", lw=0.8, ls=":", alpha=0.7)
    ax.text(f + 10, ax.get_ylim()[0] + 1, label, fontsize=7, color="grey", rotation=90)

plt.tight_layout()
plt.savefig("d4_mean_spectra.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → d4_mean_spectra.png")


# ── Figure 3: An. dirus D1 (training) vs D4 (test) ────────────────────────────
print("Figure 3: An. dirus D1 vs D4 …")
n_cols = 4
fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3, 5.5))
fig.suptitle("An. dirus — training domain D1 (top) vs unseen test domain D4 (bottom)\n"
             "Same species, completely different recording conditions",
             fontsize=10, fontweight="bold")

for col, (files, label) in enumerate([(dirus_d1, "D1 (train)"),
                                       (dirus_d4, "D4 (test)")]):
    sample = random.sample(files, min(n_cols, len(files))) if files else []
    row_idx = 0 if label.startswith("D1") else 1
    for c in range(n_cols):
        ax = axes[row_idx, c]
        if c < len(sample):
            S = mel_db(sample[c])
            if S is not None:
                librosa.display.specshow(S, sr=SR, x_axis="time", y_axis="mel",
                                          fmin=FMIN_MEL, fmax=FMAX_MEL,
                                          ax=ax, cmap="magma")
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(labelsize=6)
            else:
                ax.text(0.5, 0.5, "load error", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="grey")
            ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(label, fontsize=9, fontweight="bold",
                          color="#e41a1c" if "D1" in label else "#984ea3")
        if row_idx == 0:
            ax.set_title(f"ex {c+1}", fontsize=7)

plt.tight_layout()
plt.savefig("dirus_d1_vs_d4.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → dirus_d1_vs_d4.png")


# ── Figure 4: Power spectrum — harmonic content ────────────────────────────────
print("Figure 4: harmonic power spectrum …")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Mean power spectrum 0–1500 Hz — harmonic content in D4\n"
             "If species differ in overtone ratios or envelope shape, it should appear here",
             fontsize=10, fontweight="bold")

ax_lin, ax_log = axes
ax_lin.set_title("Linear amplitude")
ax_log.set_title("Log amplitude (dB)")

for name, sid in D4_SPECIES.items():
    files = d4_files[name]
    if not files:
        continue
    freqs, mag = mean_power_spectrum(files)
    if freqs is None:
        continue
    mask = freqs <= 1500
    mag_db = 20 * np.log10(mag[mask] + 1e-10)
    ax_lin.plot(freqs[mask], mag[mask], color=COLORS[name], lw=1.5,
                label=f"{name} (n={len(files)})")
    ax_log.plot(freqs[mask], mag_db, color=COLORS[name], lw=1.5,
                label=f"{name} (n={len(files)})")

# Mark expected harmonics at 300 Hz
for ax in (ax_lin, ax_log):
    for i, f in enumerate([300, 600, 900, 1200], start=1):
        ax.axvline(f, color="gold", lw=0.9, ls="--", alpha=0.8,
                   label=f"_{i}×300 Hz" if i == 1 else "_")
        ax.text(f + 5, ax.get_ylim()[0], f"{i}×", fontsize=7, color="goldenrod")
    ax.set_xlabel("Frequency (Hz)", fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlim(50, 1500)
    ax.grid(True, alpha=0.3)

ax_lin.set_ylabel("Mean magnitude", fontsize=9)
ax_log.set_ylabel("Mean magnitude (dB)", fontsize=9)

plt.tight_layout()
plt.savefig("d4_harmonic_power.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → d4_harmonic_power.png")

print("\nDone. Figures saved:")
print("  d4_spectrogram_grid.png")
print("  d4_mean_spectra.png")
print("  dirus_d1_vs_d4.png")
print("  d4_harmonic_power.png")
