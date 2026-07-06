"""CaloClouds-lite calo shower generator for the M2 spike.

Two parts:
  GlobalHead  — MLP(cond) -> diagonal Gaussian over standardised globals
                [total_logE, log_n_points]. Trained with Gaussian NLL.
  PointCFM    — per-point conditional flow-matching velocity field over
                standardised points (d_eta, d_phi, log_efrac), conditioned on
                [cond_embed, global]. Trained with the linear-OT CFM objective;
                sampled by Euler-integrating the ODE from noise (t=0) to data (t=1).

Points within a shower are i.i.d. given (cond, global) — the CaloClouds
per-point assumption. Shower localisation comes from the shower-centred frame
(positions are relative to the particle direction), not from a latent.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class GlobalHead(nn.Module):
    """cond -> Gaussian(mu, logvar) over 2 standardised globals."""

    def __init__(self, cond_dim=4, hidden=128, n_glob=2):
        super().__init__()
        self.net = mlp([cond_dim, hidden, hidden, 2 * n_glob])
        self.n_glob = n_glob

    def forward(self, cond):
        mu, logvar = self.net(cond).chunk(2, dim=-1)
        logvar = logvar.clamp(-8, 6)
        return mu, logvar

    def nll(self, cond, glob_std):
        mu, logvar = self(cond)
        return 0.5 * (logvar + (glob_std - mu) ** 2 / logvar.exp()).sum(-1).mean()

    @torch.no_grad()
    def sample(self, cond):
        mu, logvar = self(cond)
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()


def timestep_embed(t, dim=64):
    # t: (B,1) in [0,1] -> sinusoidal (B, dim)
    half = dim // 2
    freqs = torch.exp(torch.linspace(0, 8, half, device=t.device))
    ang = t * freqs[None, :]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


class PointCFM(nn.Module):
    """Conditional flow-matching velocity field over 3-D standardised points."""

    def __init__(self, pt_dim=3, cond_dim=4, glob_dim=2, hidden=256, t_dim=64, cond_embed=64):
        super().__init__()
        self.cond_enc = mlp([cond_dim + glob_dim, cond_embed, cond_embed])
        self.t_dim = t_dim
        self.net = mlp([pt_dim + t_dim + cond_embed, hidden, hidden, hidden, pt_dim])

    def velocity(self, x, t, c):
        te = timestep_embed(t, self.t_dim)
        return self.net(torch.cat([x, te, c], dim=-1))

    def cfm_loss(self, x1, cond, glob_std):
        """x1: (P,3) target points; cond/glob_std: (P,·) per-point (repeated per shower)."""
        c = self.cond_enc(torch.cat([cond, glob_std], dim=-1))
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], 1, device=x1.device)
        xt = (1 - t) * x0 + t * x1          # linear OT path
        target = x1 - x0                    # constant velocity
        v = self.velocity(xt, t, c)
        return F.mse_loss(v, target)

    @torch.no_grad()
    def sample(self, cond, glob_std, steps=50):
        c = self.cond_enc(torch.cat([cond, glob_std], dim=-1))
        x = torch.randn(cond.shape[0], 3, device=cond.device)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((x.shape[0], 1), k * dt, device=x.device)
            x = x + self.velocity(x, t, c) * dt
        return x


class CaloFlow(nn.Module):
    """Convenience wrapper bundling both heads + (un)standardisation buffers."""

    def __init__(self, norm, hidden_pt=256):
        super().__init__()
        self.glob = GlobalHead(cond_dim=4)
        self.points = PointCFM(hidden=hidden_pt)
        for k, v in norm.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def std_cond(self, cond):
        return (cond - self.cond_mean) / self.cond_std

    def std_glob(self, glob):
        return (glob - self.glob_mean) / self.glob_std

    def std_pts(self, pts):
        return (pts - self.pts_mean) / self.pts_std

    def unstd_glob(self, g):
        return g * self.glob_std + self.glob_mean

    def unstd_pts(self, p):
        return p * self.pts_std + self.pts_mean
