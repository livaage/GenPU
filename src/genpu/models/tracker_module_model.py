"""Tracker v3 response head: shared conditioning + surface-local AR tracker.

The v3 analogue of tracker_model.TrackerModel — bundles ParticleConditioning with the
TrackerModuleARModel (hierarchical module head + local Cartesian coords) and carries the
per-particle cont (un)standardisation buffers so a checkpoint is self-contained. The module
geometry (vocab, local frames, layer<->surface maps) is loaded from module_geometry.npz;
its arrays live inside the AR model as buffers, so they travel with the checkpoint.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from genpu.conditioning import ParticleConditioning
from genpu.module_geometry import ModuleGeometry
from genpu.models.tracker_module_ar import TrackerModuleARModel


class TrackerModuleModel(nn.Module):
    def __init__(self, norm, module_geometry_path=None, embed_dim=64, model_dim=128,
                 n_layers=4, n_heads=4, max_hits=32, use_pdg=True):
        super().__init__()
        self.cond = ParticleConditioning(embed_dim=embed_dim, use_pdg=use_pdg)
        mg = ModuleGeometry(module_geometry_path)
        hy = mg.build_hierarchy()
        self.tracker = TrackerModuleARModel(
            hy, mg.means, mg.stds, cond_dim=embed_dim, model_dim=model_dim,
            n_layers=n_layers, n_heads=n_heads, max_hits=max_hits)
        for k, v in norm.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def std_cont(self, cont):
        return (cont - self.cont_mean) / self.cont_std

    def cond_embed(self, cont_std, pdg):
        return self.cond(cont_std, pdg)
