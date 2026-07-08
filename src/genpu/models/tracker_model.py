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
                 max_hits=32, use_pdg=True, use_vertex=False):
        super().__init__()
        self.cond = ParticleConditioning(embed_dim=embed_dim, use_pdg=use_pdg)
        self.tracker = TrackerARModel(
            cond_dim=embed_dim, model_dim=model_dim, n_layers=n_layers,
            n_heads=n_heads, max_hits=max_hits, use_vertex=use_vertex,
        )
        for k, v in norm.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def std_cont(self, cont):
        return (cont - self.cont_mean) / self.cont_std

    def cond_embed(self, cont_std, pdg):
        return self.cond(cont_std, pdg)
