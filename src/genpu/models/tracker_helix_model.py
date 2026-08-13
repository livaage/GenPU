"""Tracker v4 wrapper: shared conditioning + helix-anchored AR. The helix reference (B,48,3) is
computed outside (from cont + rotated vertex) and passed to the tracker's loss/generate."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

from genpu.conditioning import ParticleConditioning
from genpu.models.tracker_helix_ar import TrackerHelixARModel
from genpu.helix import layer_references


class TrackerHelixModel(nn.Module):
    def __init__(self, norm, embed_dim=64, model_dim=128, n_layers=4, n_heads=4, max_hits=32, use_pdg=True):
        super().__init__()
        self.cond = ParticleConditioning(embed_dim=embed_dim, use_pdg=use_pdg)
        self.tracker = TrackerHelixARModel(cond_dim=embed_dim, model_dim=model_dim, n_layers=n_layers,
                                           n_heads=n_heads, max_hits=max_hits)
        for k, v in norm.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def cond_embed(self, cont_std, pdg):
        return self.cond(cont_std, pdg)

    @staticmethod
    def helix_ref(cont_phys, vxr, vyr):
        """(B,7) physical cont [logpt,eta,logE,charge,mass,vr,vz] + rotated vertex (B,) -> (B,48,3).
        Particle frame: phi0=0 (momentum along +x). CONT order: pT=exp(0), eta=1, charge=3, vz=6."""
        cp = np.asarray(cont_phys)
        return layer_references(np.exp(cp[:, 0]), np.zeros(len(cp)), cp[:, 1], cp[:, 3],
                                np.asarray(vxr), np.asarray(vyr), cp[:, 6])
