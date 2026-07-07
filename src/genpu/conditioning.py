"""Shared per-particle conditioning contract.

Both response heads (calo flow, tracker AR) condition on the SAME per-particle
representation so the system is consistent and a single model can eventually be
trained on a species mixture. Built now with a PDG-type embedding even though we
still train per-species (single-species slices just see one type — harmless);
this avoids ripping out the interface later.

Continuous feature layout (order matters — slices must match CONT_FEATURES):
    log_pt, eta, log_E, charge, mass, vr, vz
  - kinematics: log_pt, eta, log_E
  - charge: sets B-field bend direction (matters for hadron shower offset)
  - mass: weak species hint alongside the PDG embedding
  - vr, vz: production vertex — which layers are reachable; also the hook for
            decay-daughter injection (daughters get a displaced vr).
phi is deliberately excluded: the detector response is phi-invariant by symmetry
and the heads work in particle-relative frames.

trunk latent (M3): an optional per-particle vector from the event-level
permutation-equivariant trunk, concatenated in. Zero-width until M3.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from genpu.preprocessing import N_PDG_CLASSES

CONT_FEATURES = ["log_pt", "eta", "log_E", "charge", "mass", "vr", "vz"]
N_CONT = len(CONT_FEATURES)


class ParticleConditioning(nn.Module):
    """Raw per-particle features -> dense conditioning embedding.

    Args:
        embed_dim: output conditioning dimension
        pdg_embed_dim: width of the learned PDG-class embedding
        use_pdg: include the PDG embedding (set False only for ablations)
        trunk_dim: width of the M3 trunk latent (0 = not used yet)
    """

    def __init__(self, embed_dim: int = 64, pdg_embed_dim: int = 8,
                 use_pdg: bool = True, trunk_dim: int = 0):
        super().__init__()
        self.use_pdg = use_pdg
        self.trunk_dim = trunk_dim
        self.pdg_embedding = nn.Embedding(N_PDG_CLASSES, pdg_embed_dim) if use_pdg else None
        in_dim = N_CONT + (pdg_embed_dim if use_pdg else 0) + trunk_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_dim = embed_dim

    def forward(self, cont_std: torch.Tensor, pdg_class: torch.Tensor | None = None,
                trunk: torch.Tensor | None = None) -> torch.Tensor:
        parts = [cont_std]
        if self.use_pdg:
            if pdg_class is None:
                raise ValueError("use_pdg=True but pdg_class not provided")
            parts.append(self.pdg_embedding(pdg_class.long().clamp(0, N_PDG_CLASSES - 1)))
        if self.trunk_dim:
            if trunk is None:
                raise ValueError("trunk_dim>0 but trunk not provided")
            parts.append(trunk)
        return self.mlp(torch.cat(parts, dim=-1))
