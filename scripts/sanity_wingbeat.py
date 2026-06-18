"""CPU sanity test for the wingbeat descriptor + model forward.

Run: .venv/bin/python3 scripts/sanity_wingbeat.py
Verifies, with no real data:
  1. wingbeat_band_bins picks the expected band for n_mels=64, 0-4000 Hz.
  2. compute_wingbeat_descriptor centroids a synthetic in-band tone correctly
     and is invariant to a global amplitude scaling.
  3. MTRCNNClassifier(use_wingbeat_feature=True) forwards a dummy batch with a
     descriptor and produces correct logit shapes.
  4. Omitting the descriptor raises ValueError.
"""

import numpy as np
import torch

from framework.dataset import (
    WINGBEAT_DESCRIPTOR_DIM,
    compute_wingbeat_descriptor,
    wingbeat_band_bins,
)
from framework.metadata import DOMAIN_NAMES, SPECIES_NAMES
from framework.model import MTRCNNClassifier


def test_band_bins():
    import librosa

    n_mels, fmin, fmax = 64, 0.0, 4000.0
    start, end = wingbeat_band_bins(n_mels, fmin, fmax)
    centers = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax, htk=False)
    lo_hz, hi_hz = centers[start], centers[end - 1]
    print(f"[1] band bins [{start},{end}) -> {lo_hz:.0f}-{hi_hz:.0f} Hz (n_band={end-start})")
    assert 0 < start < end <= n_mels
    assert 450 < lo_hz < 600 and 2000 < hi_hz < 2300


def test_descriptor():
    n_mels = 64
    start, end = wingbeat_band_bins(n_mels, 0.0, 4000.0)
    T = 50
    # dB power floor everywhere; inject a tone at a known bin in some frames.
    feat = np.full((T, n_mels), -80.0, dtype=np.float32)
    tone_bin = start + (end - start) // 4  # 25% into band -> centroid ~0.25
    feat[T // 2 :, tone_bin] = 0.0  # loud tone only in the second half of clip
    d = compute_wingbeat_descriptor(feat, start, end)
    print(f"[2] descriptor={d}  (centroid~0.25 expected, gated to loud frames)")
    assert d.shape == (WINGBEAT_DESCRIPTOR_DIM,)
    expected_centroid = (tone_bin - start) / (end - start - 1)
    assert abs(d[0] - expected_centroid) < 0.05, (d[0], expected_centroid)
    assert 0.0 <= d[2] <= 1.0
    # Amplitude invariance: +20 dB global scale leaves centroid/bandwidth/frac fixed.
    d_scaled = compute_wingbeat_descriptor(feat + 20.0, start, end)
    assert np.allclose(d, d_scaled, atol=1e-5), (d, d_scaled)
    print("    amplitude-invariance OK")


def _dummy_config():
    return {
        "n_mels": 64,
        "embed_dim": 32,
        "dropout": 0.1,
        "use_wingbeat_feature": True,
    }


def test_model_forward():
    cfg = _dummy_config()
    model = MTRCNNClassifier(
        cfg, num_species_classes=len(SPECIES_NAMES), num_domain_classes=len(DOMAIN_NAMES)
    )
    model.eval()
    B, T = 4, 60
    features = torch.randn(B, T, cfg["n_mels"])
    lengths = torch.tensor([T, T - 5, T - 10, T], dtype=torch.long)
    wb = torch.randn(B, WINGBEAT_DESCRIPTOR_DIM)
    with torch.no_grad():
        out = model(features, lengths, wb)
    print(f"[3] species_logits={tuple(out['species_logits'].shape)} "
          f"domain_logits={tuple(out['domain_logits'].shape)}")
    assert out["species_logits"].shape == (B, len(SPECIES_NAMES))
    assert out["domain_logits"].shape == (B, len(DOMAIN_NAMES))

    try:
        with torch.no_grad():
            model(features, lengths)
        raise AssertionError("[4] expected ValueError when wb_descriptor omitted")
    except ValueError as e:
        print(f"[4] ValueError on missing descriptor OK: {e}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    test_band_bins()
    test_descriptor()
    test_model_forward()
    print("\nALL SANITY CHECKS PASSED")
