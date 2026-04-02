"""Shared particle feature encoder.

Encodes particle kinematics (log_pt, eta, phi, pdg_class, charge, mass)
into a conditioning vector used by both the hit count head and the
diffusion denoiser.
"""

import torch
import torch.nn as nn

from genpu.preprocessing import N_PDG_CLASSES


class ParticleEncoder(nn.Module):
    """Encode particle features into a dense conditioning vector.

    Continuous features (log_pt, eta, phi, charge, mass) go through an MLP.
    The discrete pdg_class gets a learned embedding.
    Both are concatenated and projected to the output dimension.
    """

    def __init__(self, embed_dim: int = 128, pdg_embed_dim: int = 16):
        super().__init__()
        self.pdg_embedding = nn.Embedding(N_PDG_CLASSES, pdg_embed_dim)

        # 5 continuous features: log_pt, eta, phi, charge, mass
        n_continuous = 5
        self.encoder = nn.Sequential(
            nn.Linear(n_continuous + pdg_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, particle_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            particle_features: (B, 6) — log_pt, eta, phi, pdg_class, charge, mass

        Returns:
            (B, embed_dim) conditioning vector
        """
        continuous = torch.cat([
            particle_features[:, :3],   # log_pt, eta, phi
            particle_features[:, 4:6],  # charge, mass
        ], dim=1)

        pdg_class = particle_features[:, 3].long().clamp(0, N_PDG_CLASSES - 1)
        pdg_emb = self.pdg_embedding(pdg_class)

        x = torch.cat([continuous, pdg_emb], dim=1)
        return self.encoder(x)
