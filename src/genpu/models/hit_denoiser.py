"""Transformer-based denoiser for hit set generation.

Predicts noise eps given:
- x_t: noisy hit features (B, N, D_hit)
- t: diffusion timestep (B,)
- cond: particle embedding (B, D_cond)
- mask: valid hit positions (B, N)

Architecture: sinusoidal timestep encoding + cross-attention conditioning +
self-attention over hit positions. Permutation-equivariant by design.
"""

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding, same as original Transformer / DDPM."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class HitDenoiserBlock(nn.Module):
    """One transformer block: self-attention + conditioning + FFN."""

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        # Conditioning injection via adaptive layer norm (scale + shift)
        self.cond_proj = nn.Linear(dim, dim * 2)

        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond, mask):
        """
        Args:
            x: (B, N, D) hit features
            cond: (B, D) conditioning (particle_emb + timestep_emb)
            mask: (B, N) bool, True = valid hit
        """
        # Self-attention with mask
        # key_padding_mask: True = ignore, so invert
        key_padding_mask = ~mask
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h

        # Adaptive conditioning (scale & shift)
        scale_shift = self.cond_proj(cond).unsqueeze(1)  # (B, 1, 2D)
        scale, shift = scale_shift.chunk(2, dim=-1)
        x = self.norm2(x) * (1 + scale) + shift

        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


class HitDenoiser(nn.Module):
    """Transformer denoiser for hit set diffusion.

    Separate instances for tracker and calo (different D_hit).
    """

    def __init__(
        self,
        hit_dim: int,
        cond_dim: int = 128,
        model_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        timesteps: int = 1000,
        dropout: float = 0.0,
    ):
        """
        Args:
            hit_dim: dimension of each hit feature vector (6 for tracker, 5 for calo)
            cond_dim: dimension of particle embedding from ParticleEncoder
            model_dim: internal transformer dimension
            n_layers: number of transformer blocks
            n_heads: attention heads
            timesteps: max diffusion timesteps (for embedding)
            dropout: dropout rate
        """
        super().__init__()
        self.model_dim = model_dim

        # Input projection: hit features -> model_dim
        self.input_proj = nn.Linear(hit_dim, model_dim)

        # Timestep embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # Conditioning: combine particle_emb + timestep_emb
        self.cond_combine = nn.Sequential(
            nn.Linear(cond_dim + model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            HitDenoiserBlock(model_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Output projection: model_dim -> hit_dim (predict noise)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, hit_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t: (B, N, D_hit) noisy hit features
            t: (B,) integer timesteps
            cond: (B, D_cond) particle embedding
            mask: (B, N) bool mask

        Returns:
            noise_pred: (B, N, D_hit) predicted noise
        """
        # Embed timestep
        t_emb = sinusoidal_embedding(t, self.model_dim)
        t_emb = self.time_mlp(t_emb)

        # Combine particle conditioning + timestep
        combined = self.cond_combine(torch.cat([cond, t_emb], dim=1))  # (B, model_dim)

        # Project hits
        h = self.input_proj(x_t)  # (B, N, model_dim)

        # Transformer blocks
        for block in self.blocks:
            h = block(h, combined, mask)

        # Output
        noise_pred = self.output_proj(h)

        # Zero out masked positions
        noise_pred = noise_pred * mask.unsqueeze(-1).float()
        return noise_pred
