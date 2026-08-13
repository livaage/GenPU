"""Tracker response head: shared conditioning + tokenized AR tracker.

Mirrors genpu.flow.calo_flow.CaloFlow: bundles the shared
ParticleConditioning embedding with the TrackerARModel and carries the
per-particle conditioning (un)standardisation buffers so a checkpoint is
self-contained.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from genpu.conditioning import ParticleConditioning
from genpu.models.tracker_ar import TrackerARModel


class TrackerModel(nn.Module):
    """Shared conditioning + AR tracker, with cont (un)standardisation buffers."""

    def __init__(self, norm, embed_dim=64, model_dim=128, n_layers=4, n_heads=4,
                 max_hits=32, use_pdg=True, use_vertex=False, use_helix=False,
                 use_mom_feat=False, z_dev_std=1.0):
        super().__init__()
        self.cond = ParticleConditioning(embed_dim=embed_dim, use_pdg=use_pdg)
        self.tracker = TrackerARModel(
            cond_dim=embed_dim, model_dim=model_dim, n_layers=n_layers,
            n_heads=n_heads, max_hits=max_hits, use_vertex=use_vertex,
            use_helix=use_helix, use_mom_feat=use_mom_feat, z_dev_std=z_dev_std,
        )
        for k, v in norm.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    @staticmethod
    def helix_params_from_cont(cont: torch.Tensor) -> torch.Tensor:
        """(B,3) physical [vr, vz, sinh(eta)] from PHYSICAL cont. CONT_FEATURES order:
        [log_pt, eta, log_E, charge, mass, vr, vz] -> eta=1, vr=5, vz=6."""
        return torch.stack([cont[:, 5], cont[:, 6], torch.sinh(cont[:, 1])], dim=-1)

    def std_cont(self, cont):
        return (cont - self.cont_mean) / self.cont_std

    def cond_embed(self, cont_std, pdg):
        return self.cond(cont_std, pdg)
