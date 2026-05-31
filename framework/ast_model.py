"""Audio Spectrogram Transformer (AST) for the DCASE2026 mosquito baseline.

Architecture:
  - Non-overlapping patch embedding via Conv2d (default 8×8 time×freq)
  - Learnable [CLS] token
  - 2D sinusoidal positional embeddings (fixed; generalise to variable lengths)
  - Standard pre-norm transformer encoder
  - Dual classification heads (species + domain), same interface as MTRCNNClassifier

Variable-length handling:
  Padding frames are excluded from self-attention via key_padding_mask derived
  from the per-sample true frame counts passed in `lengths`.

Config keys (all optional, resolved from experiment JSON):
    ast_embed_dim   int   192     patch projection / attention dimension
    ast_depth       int   4       number of transformer blocks
    ast_num_heads   int   4       attention heads (embed_dim % num_heads == 0)
    ast_mlp_ratio   int   4       FFN hidden = embed_dim * mlp_ratio
    ast_patch_time  int   8       patch height in time frames
    ast_patch_freq  int   8       patch width in mel bins (n_mels % ast_patch_freq == 0)
    ast_dropout     float 0.1     dropout applied in attention and FFN

Author: Sulagna Saha
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

def _sinusoidal_1d(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """Standard sinusoidal encoding. Returns [length, dim]."""
    position = torch.arange(length, device=device).unsqueeze(1).float()   # [L, 1]
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim)
    )                                                                        # [dim/2]
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term[:dim // 2])
    return pe


def build_2d_sinusoidal_pos_embed(
    max_time_patches: int,
    freq_patches: int,
    embed_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """2D sinusoidal positional embeddings for a (time_patch, freq_patch) grid.

    Time and frequency encodings each occupy half the embedding dimension, then
    they are concatenated so every position is uniquely identified.

    Returns: [1, max_time_patches * freq_patches, embed_dim]
    """
    assert embed_dim % 2 == 0
    half = embed_dim // 2
    pe_t = _sinusoidal_1d(max_time_patches, half, device)   # [T_p, D/2]
    pe_f = _sinusoidal_1d(freq_patches,     half, device)   # [F_p, D/2]

    # Broadcast to full grid [T_p, F_p, D]
    pe = torch.cat([
        pe_t.unsqueeze(1).expand(-1, freq_patches, -1),
        pe_f.unsqueeze(0).expand(max_time_patches, -1, -1),
    ], dim=-1)                                               # [T_p, F_p, D]

    return pe.reshape(1, max_time_patches * freq_patches, embed_dim)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN→MHA→residual → LN→FFN→residual."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden     = embed_dim * mlp_ratio
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        normed, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                              key_padding_mask=key_padding_mask)
        x = x + normed
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# AST classifier
# ---------------------------------------------------------------------------

class ASTClassifier(nn.Module):
    """Audio Spectrogram Transformer with variable-length support.

    Smoke test::

        if __name__ == "__main__":
            cfg = {"n_mels": 64, "dropout": 0.1}
            model = ASTClassifier(cfg, num_species_classes=9, num_domain_classes=5)
            feats   = torch.randn(4, 200, 64)       # [B, T, F]
            lengths = torch.tensor([200, 160, 120, 80])
            out = model(feats, lengths)
            assert out["species_logits"].shape == (4, 9)
            assert out["domain_logits"].shape  == (4, 5)
            print("ASTClassifier OK — params:",
                  sum(p.numel() for p in model.parameters()))
    """

    # Precomputed positional embedding table covers clips up to this many
    # time-patches (128 × 8 frames/patch = 1024 frames ≈ 10 s at 10 ms/frame).
    MAX_TIME_PATCHES = 128

    def __init__(self, config: dict, num_species_classes: int, num_domain_classes: int) -> None:
        super().__init__()

        embed_dim = int(config.get("ast_embed_dim",  192))
        depth     = int(config.get("ast_depth",      4))
        num_heads = int(config.get("ast_num_heads",  4))
        mlp_ratio = int(config.get("ast_mlp_ratio",  4))
        p_t       = int(config.get("ast_patch_time", 8))
        p_f       = int(config.get("ast_patch_freq", 8))
        dropout   = float(config.get("ast_dropout",  0.1))
        n_mels    = int(config["n_mels"])

        if n_mels % p_f != 0:
            raise ValueError(f"n_mels ({n_mels}) must be divisible by ast_patch_freq ({p_f})")
        if embed_dim % num_heads != 0:
            raise ValueError(f"ast_embed_dim ({embed_dim}) must be divisible by ast_num_heads ({num_heads})")

        self.patch_time  = p_t
        self.patch_freq  = p_f
        self.freq_patches = n_mels // p_f    # = 8 for default 64 mels / 8

        # Patch projection: [B, 1, T, F] → [B, embed_dim, T_p, F_p]
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(p_t, p_f),
            stride=(p_t, p_f),
            bias=True,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Fixed 2D sinusoidal positional embeddings [1, MAX*F_p, D]
        # Registered as buffer so they move with .to(device) and are not trained.
        pos = build_2d_sinusoidal_pos_embed(
            self.MAX_TIME_PATCHES, self.freq_patches, embed_dim,
            device=torch.device("cpu"),
        )
        self.register_buffer("pos_embed", pos)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.species_head = nn.Linear(embed_dim, num_species_classes)
        self.domain_head  = nn.Linear(embed_dim, num_domain_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)
        for block in self.blocks:
            nn.init.trunc_normal_(block.ffn[0].weight, std=0.02)
            nn.init.trunc_normal_(block.ffn[3].weight, std=0.02)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: padded spectrograms  [B, T, F]
            lengths:  true frame counts    [B]

        Returns:
            dict with ``species_logits`` and ``domain_logits``, each [B, n_classes]
        """
        B, T, F = features.shape

        # ---- patch embedding ------------------------------------------------
        # [B, T, F] → [B, 1, T, F] → Conv2d → [B, D, T_p, F_p]
        x   = self.patch_embed(features.unsqueeze(1))
        T_p = x.size(2)                                    # actual time patches

        # [B, D, T_p, F_p] → [B, T_p*F_p, D]
        x = x.flatten(2).transpose(1, 2)

        # ---- positional embeddings ------------------------------------------
        n_patches = T_p * self.freq_patches
        x = x + self.pos_embed[:, :n_patches, :]

        # ---- CLS token ------------------------------------------------------
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)                  # [B, 1+n_patches, D]

        # ---- attention mask from true lengths --------------------------------
        # patch is valid if its time-patch index < true number of time patches
        patch_time_lengths = (lengths // self.patch_time).clamp(min=1)   # [B]

        time_patch_idx = torch.arange(T_p, device=x.device)              # [T_p]
        patch_valid    = time_patch_idx.unsqueeze(0) < patch_time_lengths.unsqueeze(1)  # [B, T_p]

        # Expand to all freq patches: [B, T_p] → [B, T_p*F_p]
        patch_valid = patch_valid.unsqueeze(2).expand(-1, -1, self.freq_patches).reshape(B, n_patches)

        # CLS is always valid; combine and invert for key_padding_mask
        cls_valid        = patch_valid.new_ones(B, 1)
        key_padding_mask = ~torch.cat([cls_valid, patch_valid], dim=1)   # True = ignore

        # ---- transformer blocks ---------------------------------------------
        for block in self.blocks:
            x = block(x, key_padding_mask)
        x = self.norm(x)

        # ---- classification heads on CLS token ------------------------------
        cls_out = x[:, 0]
        return {
            "species_logits": self.species_head(cls_out),
            "domain_logits":  self.domain_head(cls_out),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = {"n_mels": 64, "dropout": 0.1}
    model = ASTClassifier(cfg, num_species_classes=9, num_domain_classes=5)
    feats   = torch.randn(4, 200, 64)
    lengths = torch.tensor([200, 160, 120, 80])
    out = model(feats, lengths)
    assert out["species_logits"].shape == (4, 9), out["species_logits"].shape
    assert out["domain_logits"].shape  == (4, 5), out["domain_logits"].shape
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ASTClassifier OK — trainable params: {n_params:,}")
