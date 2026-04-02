"""Hit count prediction head.

Predicts the number of tracker and calo hits for a given particle.
This runs first at generation time — if n=0, we skip the diffusion step.

Framed as classification over discrete counts (0..max_hits), since the
distributions are highly non-Gaussian (spike at 0, long tail).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HitCountHead(nn.Module):
    """Predict (n_tracker, n_calo) as categorical distributions."""

    def __init__(
        self,
        embed_dim: int = 128,
        max_tracker_hits: int = 32,
        max_calo_hits: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.max_tracker_hits = max_tracker_hits
        self.max_calo_hits = max_calo_hits

        # Shared hidden layer
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Separate heads: predict count as classification
        # +1 for the 0 class, last bin is "max_hits or more"
        self.tracker_head = nn.Linear(hidden_dim, max_tracker_hits + 1)
        self.calo_head = nn.Linear(hidden_dim, max_calo_hits + 1)

    def forward(self, particle_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            particle_emb: (B, embed_dim) from ParticleEncoder

        Returns:
            dict with:
                tracker_logits: (B, max_tracker_hits+1)
                calo_logits: (B, max_calo_hits+1)
        """
        h = self.shared(particle_emb)
        return {
            "tracker_logits": self.tracker_head(h),
            "calo_logits": self.calo_head(h),
        }

    def loss(
        self,
        particle_emb: torch.Tensor,
        n_tracker: torch.Tensor,
        n_calo: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute cross-entropy loss for hit count prediction.

        Args:
            particle_emb: (B, embed_dim)
            n_tracker: (B,) ground truth tracker hit counts
            n_calo: (B,) ground truth calo hit counts
        """
        out = self.forward(particle_emb)

        # Clamp targets to valid range
        trk_target = n_tracker.clamp(0, self.max_tracker_hits)
        cal_target = n_calo.clamp(0, self.max_calo_hits)

        trk_loss = F.cross_entropy(out["tracker_logits"], trk_target)
        cal_loss = F.cross_entropy(out["calo_logits"], cal_target)

        return {
            "tracker_count_loss": trk_loss,
            "calo_count_loss": cal_loss,
            "total_count_loss": trk_loss + cal_loss,
        }

    @torch.no_grad()
    def sample(self, particle_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample hit counts from predicted distributions.

        Returns:
            n_tracker: (B,) sampled tracker hit counts
            n_calo: (B,) sampled calo hit counts
        """
        out = self.forward(particle_emb)
        n_trk = torch.distributions.Categorical(
            logits=out["tracker_logits"]
        ).sample()
        n_cal = torch.distributions.Categorical(
            logits=out["calo_logits"]
        ).sample()
        return n_trk, n_cal
