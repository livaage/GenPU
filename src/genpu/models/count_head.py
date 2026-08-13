"""Conditioned count head: P(n_hits | particle [, d0]) as a categorical over 1..MAXB.
Gives an emergent, honest hit count (no truth n_hits) for the tracker pipeline. d0 is the
geometric impact parameter (charged primary/secondary discriminator; defined for all particles)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from genpu.conditioning import ParticleConditioning

MAXB = 48


class CountHead(nn.Module):
    def __init__(self, embed_dim=64, hidden=128, use_d0=True, n_cont=7):
        super().__init__()
        self.use_d0 = use_d0
        self.cond = ParticleConditioning(embed_dim=embed_dim, use_pdg=True)
        self.net = nn.Sequential(nn.Linear(embed_dim + (1 if use_d0 else 0), hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, MAXB))
        self.register_buffer("d0_mean", torch.zeros(1))
        self.register_buffer("d0_std", torch.ones(1))
        # cont standardization (its OWN training stats) so callers pass RAW physical cont and can't
        # feed the wrong normalization. Default 0/1 = no-op (back-compat: treats input as pre-std).
        self.register_buffer("cont_mean", torch.zeros(n_cont))
        self.register_buffer("cont_std", torch.ones(n_cont))

    def std_cont(self, cont_phys):
        return (cont_phys - self.cont_mean) / self.cont_std

    def logits_raw(self, cont_phys, pdg, d0):
        """Logits from RAW physical cont (standardizes internally with the count head's own stats)."""
        return self.logits(self.std_cont(cont_phys), pdg, d0)

    @torch.no_grad()
    def sample_raw(self, cont_phys, pdg, d0):
        return torch.distributions.Categorical(logits=self.logits_raw(cont_phys, pdg, d0)).sample() + 1

    def logits(self, cont_std, pdg, d0):
        h = self.cond(cont_std, pdg)
        if self.use_d0:
            h = torch.cat([h, ((d0.abs() - self.d0_mean) / self.d0_std).unsqueeze(-1)], -1)
        return self.net(h)

    def loss(self, cont_std, pdg, d0, n_hits):
        return F.cross_entropy(self.logits(cont_std, pdg, d0), torch.clamp(n_hits - 1, 0, MAXB - 1))

    @torch.no_grad()
    def sample(self, cont_std, pdg, d0):
        return torch.distributions.Categorical(logits=self.logits(cont_std, pdg, d0)).sample() + 1

    @torch.no_grad()
    def expected(self, cont_std, pdg, d0):
        p = torch.softmax(self.logits(cont_std, pdg, d0), -1)
        return (p * (torch.arange(MAXB, device=p.device) + 1)).sum(-1)
