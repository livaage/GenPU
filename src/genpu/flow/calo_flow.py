"""CaloClouds-lite calo shower generator.

Heads (all conditioned on the shared genpu.conditioning.ParticleConditioning
embedding — kinematics + vertex + charge + PDG):
  GlobalHead  — cond_embed -> Gaussian over standardised globals [total_logE, log_n].
  PointCFM    — per-point conditional flow-matching over standardised POSITIONS
                (d_eta, d_phi). Linear-OT CFM, Euler-sampled.
  EnergyHead  — per-cell energy, conditioned on (cond_embed, global, cell position):
                an at-floor Bernoulli + a Gaussian over log-E above the floor.
                This replaces the earlier "model log-E in the flow then clamp"
                approach, which over-piled the 50 keV floor (event gate: that pile
                was the dominant real-vs-gen discriminator, AUC 0.99).

Points within a shower are i.i.d. given (cond, global); positions localise via the
shower-centred frame. Energy correlates with position through EnergyHead's pos input.
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
    """Conditional flow-matching velocity field over 2-D standardised positions."""

    def __init__(self, pt_dim=2, embed_dim=64, glob_dim=2, hidden=256, t_dim=64, cond_hidden=64):
        super().__init__()
        self.pt_dim = pt_dim
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
        x = torch.randn(cond_embed.shape[0], self.pt_dim, device=cond_embed.device)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((x.shape[0], 1), k * dt, device=x.device)
            x = x + self.velocity(x, t, c) * dt
        return x


class EnergyHead(nn.Module):
    """Per-cell log-E: at-floor Bernoulli + Gaussian MIXTURE above the floor.

    Conditioned on cond_embed (the PARTICLE features) ONLY — deliberately NOT on
    cell position or the global (total_logE, log_n). Both of those are SAMPLED /
    generated quantities; conditioning energy on them makes it a sharp function of
    a noisy input, so at generation the marginal energy spectrum blows up (event
    gate: too broad + too many floor cells, even after dropping position). cond is
    the only NOISE-FREE conditioning (truth particle features at generation), so
    conditioning energy on cond alone makes the generated per-cell energy marginal
    match truth by construction. log-E is PHYSICAL; floor = zero-suppression thresh.
    """

    def __init__(self, embed_dim=64, hidden=128, floor_eps=0.05, n_mix=3):
        super().__init__()
        self.n_mix = n_mix
        # floor_logit + n_mix*(weight_logit, mu, log_sigma): a K-Gaussian MIXTURE over
        # above-floor log-E captures the skewed tail a single Gaussian misses (the
        # event-gate residual was logE_p90 / frac_near_floor).
        self.net = mlp([embed_dim, hidden, hidden, 1 + 3 * n_mix])
        self.floor_eps = floor_eps

    def _out(self, cond_embed):
        o = self.net(cond_embed)
        m = o[:, 1:].view(-1, self.n_mix, 3)
        return o[:, 0], m[..., 0], m[..., 1], m[..., 2].clamp(-4, 3)  # floor_logit, w_logit, mu, log_sigma

    def loss(self, cond_embed, logE, log_floor):
        is_floor = (logE <= log_floor + self.floor_eps).float()
        floor_logit, w_logit, mu, ls = self._out(cond_embed)
        bce = F.binary_cross_entropy_with_logits(floor_logit, is_floor)
        logw = F.log_softmax(w_logit, dim=-1)                            # (N,K)
        comp = -0.5 * ((logE.unsqueeze(-1) - mu) / ls.exp()) ** 2 - ls    # (N,K), up to const
        logp = torch.logsumexp(logw + comp, dim=-1)                       # (N,)
        above = 1.0 - is_floor
        gnll = (-logp * above).sum() / above.sum().clamp(min=1.0)
        return bce + gnll

    @torch.no_grad()
    def sample(self, cond_embed, log_floor):
        floor_logit, w_logit, mu, ls = self._out(cond_embed)
        at_floor = torch.rand_like(floor_logit) < torch.sigmoid(floor_logit)
        k = torch.distributions.Categorical(logits=w_logit).sample()     # (N,)
        mu_k = mu.gather(1, k[:, None])[:, 0]; ls_k = ls.gather(1, k[:, None])[:, 0]
        logE = mu_k + torch.randn_like(mu_k) * ls_k.exp()
        logE = torch.where(at_floor, torch.full_like(logE, log_floor), logE)
        return logE.clamp(min=log_floor)


class CaloFlow(nn.Module):
    """Shared conditioning + global/position/energy heads, with (un)standardisation buffers."""

    def __init__(self, norm, embed_dim=64, hidden_pt=256, use_pdg=True, log_floor=None):
        super().__init__()
        self.cond = ParticleConditioning(embed_dim=embed_dim, use_pdg=use_pdg)
        # global = [total_logE, log_n] (+ [core_eta, core_phi] when the slice carries a
        # per-shower core for the compactness fix); infer width from the norm buffers.
        self.glob = GlobalHead(embed_dim=embed_dim, n_glob=int(len(norm["glob_mean"])))
        self.points = PointCFM(embed_dim=embed_dim, hidden=hidden_pt)
        self.energy = EnergyHead(embed_dim=embed_dim)
        for k, v in norm.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
        if log_floor is None:
            log_floor = float(torch.log(torch.tensor(5e-5)))
        self.register_buffer("log_floor", torch.tensor(float(log_floor)))

    # standardisation helpers
    def std_cont(self, cont):
        return (cont - self.cont_mean) / self.cont_std

    def std_glob(self, glob):
        return (glob - self.glob_mean) / self.glob_std

    def std_pos(self, pos):
        return (pos - self.pts_mean[:2]) / self.pts_std[:2]

    def unstd_glob(self, g):
        return g * self.glob_std + self.glob_mean

    def unstd_pos(self, p):
        return p * self.pts_std[:2] + self.pts_mean[:2]

    def cond_embed(self, cont_std, pdg):
        return self.cond(cont_std, pdg)
