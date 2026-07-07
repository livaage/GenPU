"""CaloClouds-lite calo shower generator for the M2 spike.

Two heads, both conditioned on the SHARED per-particle embedding produced by
genpu.conditioning.ParticleConditioning (kinematics + vertex + charge + PDG):
  GlobalHead  — cond_embed -> diagonal Gaussian over standardised globals
                [total_logE, log_n_points]. Trained with Gaussian NLL.
  PointCFM    — per-point conditional flow-matching velocity field over
                standardised points (d_eta, d_phi, log_ecell), conditioned on
                [cond_embed, global]. Linear-OT CFM; sampled by Euler ODE.

Points within a shower are i.i.d. given (cond, global) — the CaloClouds
per-point assumption. Shower localisation comes from the shower-centred frame
(positions relative to the particle direction), not from a latent.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from genpu.conditioning import ParticleConditioning


def mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class GlobalHead(nn.Module):
    """cond_embed -> Gaussian(mu, logvar) over 2 standardised globals."""

    def __init__(self, embed_dim=64, hidden=128, n_glob=2):
        super().__init__()
        self.net = mlp([embed_dim, hidden, hidden, 2 * n_glob])

    def forward(self, cond_embed):
        mu, logvar = self.net(cond_embed).chunk(2, dim=-1)
        return mu, logvar.clamp(-8, 6)

    def nll(self, cond_embed, glob_std):
        mu, logvar = self(cond_embed)
        return 0.5 * (logvar + (glob_std - mu) ** 2 / logvar.exp()).sum(-1).mean()

    @torch.no_grad()
    def sample(self, cond_embed):
        mu, logvar = self(cond_embed)
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()


def timestep_embed(t, dim=64):
    half = dim // 2
    freqs = torch.exp(torch.linspace(0, 8, half, device=t.device))
    ang = t * freqs[None, :]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class PointCFM(nn.Module):
    """Conditional flow-matching velocity field over 3-D standardised points."""

    def __init__(self, pt_dim=3, embed_dim=64, glob_dim=2, hidden=256, t_dim=64, cond_hidden=64):
        super().__init__()
        self.cond_enc = mlp([embed_dim + glob_dim, cond_hidden, cond_hidden])
        self.t_dim = t_dim
        self.net = mlp([pt_dim + t_dim + cond_hidden, hidden, hidden, hidden, pt_dim])

    def velocity(self, x, t, c):
        return self.net(torch.cat([x, timestep_embed(t, self.t_dim), c], dim=-1))

    def cfm_loss(self, x1, cond_embed, glob_std):
        c = self.cond_enc(torch.cat([cond_embed, glob_std], dim=-1))
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], 1, device=x1.device)
        xt = (1 - t) * x0 + t * x1
        return F.mse_loss(self.velocity(xt, t, c), x1 - x0)

    @torch.no_grad()
    def sample(self, cond_embed, glob_std, steps=50):
        c = self.cond_enc(torch.cat([cond_embed, glob_std], dim=-1))
        x = torch.randn(cond_embed.shape[0], 3, device=cond_embed.device)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((x.shape[0], 1), k * dt, device=x.device)
            x = x + self.velocity(x, t, c) * dt
        return x


class CaloFlow(nn.Module):
    """Shared conditioning + global head + per-point CFM, with (un)standardisation buffers."""

    def __init__(self, norm, embed_dim=64, hidden_pt=256, use_pdg=True):
        super().__init__()
        self.cond = ParticleConditioning(embed_dim=embed_dim, use_pdg=use_pdg)
        self.glob = GlobalHead(embed_dim=embed_dim)
        self.points = PointCFM(embed_dim=embed_dim, hidden=hidden_pt)
        for k, v in norm.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    # standardisation helpers
    def std_cont(self, cont):
        return (cont - self.cont_mean) / self.cont_std

    def std_glob(self, glob):
        return (glob - self.glob_mean) / self.glob_std

    def std_pts(self, pts):
        return (pts - self.pts_mean) / self.pts_std

    def unstd_glob(self, g):
        return g * self.glob_std + self.glob_mean

    def unstd_pts(self, p):
        return p * self.pts_std + self.pts_mean

    def cond_embed(self, cont_std, pdg):
        return self.cond(cont_std, pdg)
