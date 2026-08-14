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

# anchor conditioning features: [a_eta, a_phi, |a|] (standardised) + 4-way branch one-hot
ANCHOR_FEAT_DIM = 7
N_ANCHOR_MODES = 4


def mlp(sizes, act=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class GlobalHead(nn.Module):
    """cond_embed -> diagonal Gaussian MIXTURE over the standardised globals
    [total_logE, log_n, core_eta, core_phi]. A single Gaussian smears the shower
    core, whose real distribution is sharply peaked at the particle direction with
    heavy tails (kurtosis ~8-20); a K-component mixture captures peak + tails.

    `anchor_dim > 0` appends per-shower ANCHOR features (see CaloFlow.anchor_feats) to the
    conditioning of THIS head only. Measured 2026-08-14 on the Phase 1 checkpoints: the real
    residual core scale differs 10x (pion) / 15x (e±) between a shower whose particle reaches the
    calo face and a curler anchored at its turning point, and that branch is a hard threshold
    (2R vs r_calo, arc length vs pi*R) the trunk's smooth features cannot express — so the mixture
    blended them, producing face showers 1.13-1.48x too wide and curlers 0.85-0.92x too narrow.
    Appended AFTER the trunk on purpose: with a shared trunk, feeding this in upstream would move
    what the point and energy heads see (measured twice on 2026-08-13).
    """

    def __init__(self, embed_dim=64, hidden=128, n_glob=2, n_mix=4, anchor_dim=0):
        super().__init__()
        self.n_glob = n_glob; self.n_mix = n_mix; self.anchor_dim = anchor_dim
        self.net = mlp([embed_dim + anchor_dim, hidden, hidden, n_mix * (1 + 2 * n_glob)])

    def _out(self, cond_embed, anchor=None):
        if self.anchor_dim:
            if anchor is None:
                raise ValueError("GlobalHead built with anchor_dim>0 needs anchor features")
            cond_embed = torch.cat([cond_embed, anchor], dim=-1)
        o = self.net(cond_embed); K, G = self.n_mix, self.n_glob
        w = o[:, :K]
        r = o[:, K:].view(-1, K, 2 * G)
        return w, r[:, :, :G], r[:, :, G:].clamp(-8, 6)   # w_logit (N,K), mu (N,K,G), log_sigma (N,K,G)

    def nll(self, cond_embed, glob_std, anchor=None):
        w, mu, ls = self._out(cond_embed, anchor)
        logw = F.log_softmax(w, dim=-1)                                   # (N,K)
        comp = (-0.5 * ((glob_std.unsqueeze(1) - mu) / ls.exp()) ** 2 - ls).sum(-1)  # (N,K)
        return -(torch.logsumexp(logw + comp, dim=-1)).mean()

    @torch.no_grad()
    def sample(self, cond_embed, anchor=None):
        w, mu, ls = self._out(cond_embed, anchor)
        k = torch.distributions.Categorical(logits=w).sample()           # (N,)
        idx = k[:, None, None].expand(-1, 1, self.n_glob)
        mu_k = mu.gather(1, idx)[:, 0]; ls_k = ls.gather(1, idx)[:, 0]
        return mu_k + torch.randn_like(mu_k) * ls_k.exp()


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

    Conditioning (`glob_dim > 0`, the default since the 2026-08-13 energy audit): the
    cond_embed (PARTICLE features) **plus the standardised global** (total_logE, log_n).
    The particle features alone cannot set a shower's energy SCALE — measured on the
    slices, the per-shower mean cell log-E has R^2 = 0.05 from particle features but
    R^2 = 0.85 (pion) / 0.93 (photon) once (total_logE, log_n) are known. A cond-only
    head therefore emits that shower-level variance as INDEPENDENT per-cell noise, which
    is why pion per-shower energy resolution came out ~40% too broad (1.3-1.8 vs real
    0.9-1.2). Pairing this conditioning with `partition_energies` (renormalise the cells
    to the sampled total) is what makes per-shower energy right.

    An earlier version conditioned on the global and blew up, because the cells were left
    as free absolute energies: errors in a sharp function of a noisy input landed directly
    in the marginal. The renormalisation is what removes that failure mode — the global
    sets the total exactly, and the head only has to model the SHAPE of the split.

    log-E is PHYSICAL; floor = zero-suppression threshold.
    """

    def __init__(self, embed_dim=64, hidden=128, floor_eps=0.05, n_mix=3, glob_dim=0):
        super().__init__()
        self.n_mix = n_mix
        self.glob_dim = glob_dim
        # floor_logit + n_mix*(weight_logit, mu, log_sigma): a K-Gaussian MIXTURE over
        # above-floor log-E captures the skewed tail a single Gaussian misses (the
        # event-gate residual was logE_p90 / frac_near_floor).
        self.net = mlp([embed_dim + glob_dim, hidden, hidden, 1 + 3 * n_mix])
        self.floor_eps = floor_eps

    def _out(self, cond_embed, glob_std=None):
        if self.glob_dim:
            if glob_std is None:
                raise ValueError("EnergyHead built with glob_dim>0 needs glob_std")
            cond_embed = torch.cat([cond_embed, glob_std[:, :self.glob_dim]], dim=-1)
        o = self.net(cond_embed)
        m = o[:, 1:].view(-1, self.n_mix, 3)
        return o[:, 0], m[..., 0], m[..., 1], m[..., 2].clamp(-4, 3)  # floor_logit, w_logit, mu, log_sigma

    def loss(self, cond_embed, logE, log_floor, glob_std=None):
        # at-floor = NARROW band around the exact zero-suppression pile (single-contributor
        # cells at 5e-5). Everything else — including the physical SUB-floor tail (shared-cell
        # contributions below 5e-5) — is modelled by the mixture, so we don't clamp it away.
        is_floor = (torch.abs(logE - log_floor) < self.floor_eps).float()
        floor_logit, w_logit, mu, ls = self._out(cond_embed, glob_std)
        bce = F.binary_cross_entropy_with_logits(floor_logit, is_floor)
        logw = F.log_softmax(w_logit, dim=-1)                            # (N,K)
        comp = -0.5 * ((logE.unsqueeze(-1) - mu) / ls.exp()) ** 2 - ls    # (N,K), up to const
        logp = torch.logsumexp(logw + comp, dim=-1)                       # (N,)
        cont = 1.0 - is_floor                                            # continuum (above AND below pile)
        gnll = (-logp * cont).sum() / cont.sum().clamp(min=1.0)
        return bce + gnll

    @torch.no_grad()
    def sample(self, cond_embed, log_floor, glob_std=None, logE_max=None):
        floor_logit, w_logit, mu, ls = self._out(cond_embed, glob_std)
        at_floor = torch.rand_like(floor_logit) < torch.sigmoid(floor_logit)
        k = torch.distributions.Categorical(logits=w_logit).sample()     # (N,)
        mu_k = mu.gather(1, k[:, None])[:, 0]; ls_k = ls.gather(1, k[:, None])[:, 0]
        logE = mu_k + torch.randn_like(mu_k) * ls_k.exp()
        logE = torch.where(at_floor, torch.full_like(logE, log_floor), logE)
        # BOUND THE TAIL. A mixture Gaussian is unbounded above, so a rare draw lands cells far
        # past anything physical (measured: pion gen max log-E +0.28 vs real -1.76, i.e. a 1.3 GeV
        # cell where the hardest real cell is 0.17 GeV). 167 such cells in 4.7M were enough to put
        # <E_reco/E_true> at 4.7 vs 0.016 in the lowest energy bin. Clamping at the data max is the
        # same "bounded inverse" argument as the quantile position transform: worst case saturates.
        logE = logE.clamp(min=log_floor - 12.0)                          # allow physical sub-floor tail
        if logE_max is None:
            return logE
        # scalar bound, or a PER-CELL bound (per-species maxima — a photon's hardest real cell is
        # 0.078 GeV where a hadronic one is 0.32, so one shared bound is 4x too loose for EM)
        return torch.minimum(logE, torch.as_tensor(logE_max, dtype=logE.dtype, device=logE.device))


class CaloFlow(nn.Module):
    """Shared conditioning + global/position/energy heads, with (un)standardisation buffers."""

    def __init__(self, norm, embed_dim=64, hidden_pt=256, use_pdg=True, log_floor=None,
                 qt_z=None, qt_pos_x=None, qt_glob_x=None, qt_glob_mask=None,
                 energy_use_glob=True, logE_max=None, logE_max_pdg=None, width_norm=False,
                 ctx_pt_edges=None, ctx_loc=None, ctx_scale=None, ctx_mask=None,
                 separate_trunks=False, core_anchored=False, anchor_cond=False,
                 anchor_mean=None, anchor_std=None):
        super().__init__()
        # CONDITIONING TRUNK(S). Shared (the original) means one ParticleConditioning MLP feeds all
        # three heads and the summed loss back-propagates every head's gradient into it — so changing
        # ONE head's task silently moves what the OTHERS see. Measured twice on 2026-08-13: altering
        # only the GlobalHead task (width_norm, then ctx_norm) degraded the untouched energy head
        # identically (e± cell_logE 0.017 -> 0.081, frac_near_floor AUC 0.616 -> 0.858).
        # Separate trunks make that coupling impossible, for ~+11k params on a 240k model (~4%).
        self.separate_trunks = separate_trunks
        if separate_trunks:
            self.cond_glob = ParticleConditioning(embed_dim=embed_dim, use_pdg=use_pdg)
            self.cond_points = ParticleConditioning(embed_dim=embed_dim, use_pdg=use_pdg)
            self.cond_energy = ParticleConditioning(embed_dim=embed_dim, use_pdg=use_pdg)
        else:
            self.cond = ParticleConditioning(embed_dim=embed_dim, use_pdg=use_pdg)
        # global = [total_logE, log_n] (+ [core_eta, core_phi] when the slice carries a
        # per-shower core for the compactness fix); infer width from the norm buffers.
        # n_mix=8: the per-shower core is sharply peaked (leptokurtic); more components let
        # the mixture fit the narrow center on plain-normalised data (no coordinate warp).
        # anchor conditioning for the GlobalHead: [a_eta, a_phi, |a|] standardised + a 4-way
        # branch one-hot (barrel / endcap / turning point / none). See GlobalHead's docstring.
        self.anchor_cond = bool(anchor_cond)
        anchor_dim = ANCHOR_FEAT_DIM if anchor_cond else 0
        self.glob = GlobalHead(embed_dim=embed_dim, n_glob=int(len(norm["glob_mean"])), n_mix=8,
                               anchor_dim=anchor_dim)
        self.points = PointCFM(embed_dim=embed_dim, hidden=hidden_pt)
        # energy sees (total_logE, log_n) — the shower energy scale particle features can't predict
        self.energy = EnergyHead(embed_dim=embed_dim, n_mix=4,   # extra component for the sub-floor tail
                                 glob_dim=2 if energy_use_glob else 0)
        for k, v in norm.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))
        if log_floor is None:
            log_floor = float(torch.log(torch.tensor(5e-5)))
        self.register_buffer("log_floor", torch.tensor(float(log_floor)))
        # hardest cell log-E seen in training; the sampling clamp (see EnergyHead.sample).
        # inf = no clamp, for loading pre-audit checkpoints unchanged.
        self.register_buffer("logE_max", torch.tensor(float("inf") if logE_max is None else float(logE_max)))
        # per-PDG-class version of the same bound; the multi-species head needs it because one
        # shared max is set by the hardest hadronic cell and is far too loose for EM showers.
        from genpu.preprocessing import N_PDG_CLASSES
        self.register_buffer("logE_max_pdg", torch.full((N_PDG_CLASSES,), float("inf"))
                             if logE_max_pdg is None else
                             torch.as_tensor(logE_max_pdg, dtype=torch.float32))
        # points were stored in units of each shower's own RMS width, with log_width as glob dim 4;
        # generation must scale the sampled cloud back up by the sampled width.
        self.register_buffer("width_norm", torch.tensor(1.0 if width_norm else 0.0))
        # CORE ANCHOR (Phase 1). When the slice stored the core as a residual from the truth-helix
        # extrapolation to the calo front face, glob dims 2,3 are that residual and generation MUST
        # add the anchor back (`sample_showers(core_anchor=...)`, computed by genpu.calo_geom from
        # truth conditioning). Recorded as a buffer so the generation path can refuse to run without
        # it — silently omitting the anchor would place pion showers ~1.5 rad off in phi and still
        # produce plausible-looking marginals.
        self.register_buffer("core_anchored", torch.tensor(1.0 if core_anchored else 0.0))
        self.register_buffer("anchor_cond_on", torch.tensor(1.0 if anchor_cond else 0.0))
        self.register_buffer("anchor_mean", torch.zeros(3) if anchor_mean is None
                             else torch.as_tensor(anchor_mean, dtype=torch.float32))
        self.register_buffer("anchor_std", torch.ones(3) if anchor_std is None
                             else torch.as_tensor(anchor_std, dtype=torch.float32))

        # CONTEXT NORMALISATION of selected global dims (the shower core). The pooled quantile
        # transform matches the core's MARGINAL by construction but leaves every conditional slice
        # heavy-tailed and differently scaled (measured: per-(charge,pT)-bin core spread varies 2.4x
        # for e±, 4.0x for pions), so the mixture regresses to the mean and generates only 0.58x
        # (e±) / 0.94x (pion) of the real per-bin spread. Standardising each dim by its own
        # (charge sign x pT bin) location and scale FIRST removes that variation, so the pooled
        # quantile transform then acts on a shape that is actually representative.
        # The context comes from TRUTH conditioning (charge, log_pt) and so is exactly known at
        # generation — unlike the sampled-global conditioning that failed on 2026-08-13.
        # NOTE: ctx-normalised dims MUST also be quantile-masked (the plain z-score path uses the
        # slice's raw glob_mean/std, which no longer matches a context-normalised value).
        nG = int(len(norm["glob_mean"]))
        C = 1 if ctx_loc is None else len(ctx_loc)
        self.register_buffer("ctx_pt_edges", torch.zeros(0) if ctx_pt_edges is None
                             else torch.as_tensor(ctx_pt_edges, dtype=torch.float32))
        self.register_buffer("ctx_loc", torch.zeros(C, nG) if ctx_loc is None
                             else torch.as_tensor(ctx_loc, dtype=torch.float32))
        self.register_buffer("ctx_scale", torch.ones(C, nG) if ctx_scale is None
                             else torch.as_tensor(ctx_scale, dtype=torch.float32))
        self.register_buffer("ctx_mask", torch.zeros(nG) if ctx_mask is None
                             else torch.as_tensor(ctx_mask, dtype=torch.float32))
        # NORMAL-QUANTILE normalisation. A near-delta d_eta/d_phi peak (delta) and a sharply
        # peaked per-shower core defeat both a continuous flow (can't emit a spike) and a
        # Gaussian mixture (plateaus on leptokurtic data). The quantile transform maps each
        # marginal to EXACTLY Gaussian via its empirical inverse-CDF (a monotone map), so the
        # flow/mixture only ever fit a standard Gaussian. Unlike arcsinh the inverse is
        # BOUNDED — it maps a Gaussian sample back onto the DATA's own quantile range, so a
        # mixture tail can't explode (worst case it saturates at the real min/max). Reference
        # quantiles ride in buffers (fit by the trainer); default = identity (z-score path).
        G = int(len(norm["glob_mean"]))
        K = len(qt_z) if qt_z is not None else 256
        self.register_buffer("qt_z", torch.as_tensor(
            qt_z if qt_z is not None else torch.linspace(-4.0, 4.0, K), dtype=torch.float32))
        self.register_buffer("qt_pos_x", torch.as_tensor(
            qt_pos_x if qt_pos_x is not None else torch.zeros(K, 2), dtype=torch.float32))
        self.register_buffer("qt_pos_on", torch.tensor(1.0 if qt_pos_x is not None else 0.0))
        self.register_buffer("qt_glob_x", torch.as_tensor(
            qt_glob_x if qt_glob_x is not None else torch.zeros(K, G), dtype=torch.float32))
        self.register_buffer("qt_glob_mask", torch.as_tensor(
            qt_glob_mask if qt_glob_mask is not None else torch.zeros(G), dtype=torch.float32))

    @classmethod
    def from_checkpoint(cls, sd, norm, **kw):
        """Build a model whose shape-dependent options match a saved state_dict, then load it.

        Several buffers changed shape as the model grew (per-class energy bounds, context
        normalisation tables); a size mismatch raises even under strict=False, so the options have
        to be inferred from the state dict rather than defaulted. Older checkpoints simply lack the
        keys and fall back to the pre-existing behaviour.
        """
        opts = dict(energy_use_glob=sd["energy.net.0.weight"].shape[1] > 64,
                    width_norm=bool(float(sd.get("width_norm", 0.0))) or len(norm["glob_mean"]) >= 5,
                    separate_trunks=any(k.startswith("cond_glob.") for k in sd),
                    core_anchored=bool(float(sd.get("core_anchored", 0.0))),
                    anchor_cond=bool(float(sd.get("anchor_cond_on", 0.0))))
        for key, arg in [("ctx_pt_edges", "ctx_pt_edges"), ("ctx_loc", "ctx_loc"),
                         ("ctx_scale", "ctx_scale"), ("ctx_mask", "ctx_mask")]:
            if key in sd:
                opts[arg] = sd[key]
        opts.update(kw)
        model = cls(norm, **opts)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        return model, missing, unexpected

    # standardisation helpers
    def std_cont(self, cont):
        return (cont - self.cont_mean) / self.cont_std

    def _qt_fwd(self, x, ref_x):     # (N,D) physical -> Gaussian, per-col monotone interp
        out = torch.empty_like(x)
        for d in range(x.shape[1]):
            xp = ref_x[:, d].contiguous()
            xc = x[:, d].clamp(xp[0], xp[-1]).contiguous()   # bound the tail (piles at +-max sigma)
            idx = torch.searchsorted(xp, xc).clamp(1, xp.shape[0] - 1)
            x0, x1 = xp[idx - 1], xp[idx]
            z0, z1 = self.qt_z[idx - 1], self.qt_z[idx]
            t = (xc - x0) / (x1 - x0).clamp(min=1e-9)
            out[:, d] = z0 + t * (z1 - z0)
        return out

    def _qt_inv(self, z, ref_x):     # (N,D) Gaussian -> physical, BOUNDED to the data range
        zp = self.qt_z.contiguous()
        out = torch.empty_like(z)
        for d in range(z.shape[1]):
            zc = z[:, d].clamp(zp[0], zp[-1])
            idx = torch.searchsorted(zp, zc).clamp(1, zp.shape[0] - 1)
            z0, z1 = zp[idx - 1], zp[idx]
            x0, x1 = ref_x[idx - 1, d], ref_x[idx, d]
            t = (zc - z0) / (z1 - z0).clamp(min=1e-9)
            out[:, d] = x0 + t * (x1 - x0)
        return out

    def context_of(self, cont_std):
        """(charge sign, pT bin) index per particle, from TRUTH conditioning. cont layout:
        [log_pt, eta, log_E, charge, mass, vr, vz]; cont_std is standardised, so undo that first."""
        cont = cont_std * self.cont_std + self.cont_mean
        n_pt = self.ctx_pt_edges.shape[0] + 1
        pt_bin = torch.bucketize(cont[:, 0].contiguous(), self.ctx_pt_edges)
        q_idx = torch.sign(cont[:, 3]).long() + 1                     # 0 = negative, 1 = neutral, 2 = positive
        return (q_idx * n_pt + pt_bin).clamp(0, self.ctx_loc.shape[0] - 1)

    def _ctx_norm(self, glob, ctx, invert=False):
        m = self.ctx_mask > 0
        if ctx is None or not bool(m.any()):
            return glob
        loc, sc = self.ctx_loc[ctx], self.ctx_scale[ctx].clamp(min=1e-9)
        return torch.where(m, glob * sc + loc if invert else (glob - loc) / sc, glob)

    def std_glob(self, glob, ctx=None):
        g = self._ctx_norm(glob, ctx)                                  # context location/scale first
        z = (g - self.glob_mean) / self.glob_std
        m = self.qt_glob_mask > 0
        if bool(m.any()):
            z = torch.where(m, self._qt_fwd(g, self.qt_glob_x), z)     # quantile for masked dims
        return z

    def unstd_glob(self, g, ctx=None):
        u = g * self.glob_std + self.glob_mean
        m = self.qt_glob_mask > 0
        if bool(m.any()):
            u = torch.where(m, self._qt_inv(g, self.qt_glob_x), u)
        return self._ctx_norm(u, ctx, invert=True)

    def std_pos(self, pos):
        if bool(self.qt_pos_on > 0):
            return self._qt_fwd(pos, self.qt_pos_x)
        return (pos - self.pts_mean[:2]) / self.pts_std[:2]

    def unstd_pos(self, p):
        if bool(self.qt_pos_on > 0):
            return self._qt_inv(p, self.qt_pos_x)
        return p * self.pts_std[:2] + self.pts_mean[:2]

    def anchor_feats(self, anchor, anchor_mode):
        """(S,2) anchor + (S,) branch code -> (S, ANCHOR_FEAT_DIM) GlobalHead conditioning.

        Truth-derived and deterministic, exactly like the anchor itself, so it carries no
        exposure-bias risk. Training and generation must build it the same way — hence one method.
        """
        a = torch.as_tensor(anchor, dtype=torch.float32, device=self.anchor_mean.device)
        mag = torch.hypot(a[:, 0], a[:, 1]).unsqueeze(-1)
        z = (torch.cat([a, mag], dim=-1) - self.anchor_mean) / self.anchor_std.clamp(min=1e-6)
        m = torch.as_tensor(anchor_mode, dtype=torch.long, device=z.device).clamp(0, N_ANCHOR_MODES - 1)
        return torch.cat([z, F.one_hot(m, N_ANCHOR_MODES).to(z.dtype)], dim=-1)

    def cond_embed(self, cont_std, pdg, head=None):
        """Conditioning embedding for `head` in {'glob','points','energy'}. With a shared trunk all
        heads get the same vector (head is ignored); with separate trunks each gets its own."""
        if not self.separate_trunks:
            return self.cond(cont_std, pdg)
        return getattr(self, f"cond_{head or 'glob'}")(cont_std, pdg)

    # ---- generation ------------------------------------------------------------------
    @torch.no_grad()
    def sample_showers(self, cont_std, pdg, steps=50, max_cells=128, partition=True,
                       scale_bounds=(1e-2, 1e2), e_true=None, width_renorm=False,
                       core_anchor=None, anchor_mode=None):
        """Sample complete showers for a batch of particles. THE generation path — metrics,
        eval and the full-event generator all call this so they cannot drift apart.

        Returns a dict of numpy-friendly tensors:
          n     (S,)   int64  cells per shower          src   (P,)  int64  shower index per cell
          core  (S,2)         shower core (d_eta,d_phi) offset from the particle direction
          pos   (P,2)         cell offset from the core (add core + particle eta/phi for global)
          logE  (P,)          cell log-E [GeV]          total (S,)  sampled shower total E [GeV]

        `partition=True` renormalises the cells so each shower's cells sum to the SAMPLED
        total (GlobalHead), instead of letting the total be the sum of independently drawn
        cells. Measured on the pion slice: sum-of-cells gives W/sigma 0.073 on the log total
        and 1.3-1.8 energy resolution (real 0.9-1.2), while the GlobalHead total is 0.017 —
        so the total is much better modelled as a global than as a sum. At-floor cells are
        pinned at the zero-suppression value and the above-floor cells absorb the residual,
        so the floor pile survives the rescale. The scale is bounded (`scale_bounds`) so a
        shower whose above-floor energy is negligible cannot be blown up to hit its total;
        those rare showers keep a slightly-off total rather than gaining an unphysical cell.

        `core_anchor` (S,2) is REQUIRED for core-anchored models (Phase 1): the sampled global
        holds the core as a residual from the truth-helix prediction at the calo front face, and
        the returned `core` is `residual + anchor`. Build it with
        `genpu.calo_geom.core_anchor(pt, phi, eta, q, vx, vy, vz, *load_front_face())` from the
        SAME truth conditioning the slice was built with.
        """
        if bool(self.core_anchored > 0) and core_anchor is None:
            raise ValueError(
                "this checkpoint stores the shower core as a helix RESIDUAL; pass core_anchor=(S,2) "
                "from genpu.calo_geom.core_anchor(...) or every shower lands at the particle "
                "direction (~1.5 rad off in phi for pions)")
        if core_anchor is not None and not bool(self.core_anchored > 0):
            raise ValueError("core_anchor passed to a model trained WITHOUT the anchor")
        if self.anchor_cond and anchor_mode is None:
            raise ValueError("this checkpoint conditions the GlobalHead on the anchor branch; "
                             "pass anchor_mode=(S,) from genpu.calo_geom.core_anchor(...)")
        af = self.anchor_feats(core_anchor, anchor_mode) if self.anchor_cond else None
        g_std = self.glob.sample(self.cond_embed(cont_std, pdg, "glob"), af)
        ctx = self.context_of(cont_std) if bool((self.ctx_mask > 0).any()) else None
        g = self.unstd_glob(g_std, ctx)
        S = g.shape[0]
        n = torch.exp(g[:, 1]).round().clamp(1, max_cells).long()
        src = torch.repeat_interleave(torch.arange(S, device=g.device), n)
        # per-shower embeddings, then index by cell — cheaper than embedding every cell
        ce_p = self.cond_embed(cont_std, pdg, "points")
        ce_e = self.cond_embed(cont_std, pdg, "energy")
        pos = self.unstd_pos(self.points.sample(ce_p[src], g_std[src][:, :2], steps=steps))
        if bool(self.width_norm > 0):
            # the flow emits SHAPE in units of the shower's width; scale by the sampled width so
            # the per-shower width distribution comes from the global mixture (which can fit its
            # ~50x q10-q90 spread) instead of from an i.i.d. point cloud (which cannot).
            if width_renorm:
                # project each cloud onto exactly unit RMS first. Real normalised clouds satisfy
                # RMS==1 by construction; n i.i.d. draws only satisfy it to ~1/sqrt(2n), which would
                # smear the width distribution back out. Off by default so the effect is measurable.
                r2 = torch.zeros(S, device=pos.device).index_add_(0, src, (pos ** 2).sum(-1))
                rms = (r2 / n.to(pos.dtype)).sqrt()
                pos = pos / rms[src].clamp(min=1e-6).unsqueeze(-1)
            pos = pos * g[:, 4].exp()[src].unsqueeze(-1)
        # tightest available bound per cell: the shared scalar and the cell's own species max
        lmax = torch.minimum(self.logE_max.expand(src.shape[0]),
                             self.logE_max_pdg[pdg.long().clamp(0, self.logE_max_pdg.shape[0] - 1)][src])
        logE = self.energy.sample(ce_e[src], self.log_floor, glob_std=g_std[src], logE_max=lmax)
        core = g[:, 2:4] if g.shape[1] >= 4 else torch.zeros(S, 2, device=g.device)
        if core_anchor is not None:
            # the mixture predicted a small local residual; the physics prediction supplies the
            # 1.5 m bending displacement it no longer has to learn
            core = core + torch.as_tensor(core_anchor, dtype=core.dtype, device=core.device)
        total = torch.exp(g[:, 0])

        if partition:
            e = torch.exp(logE)
            is_floor = logE <= self.log_floor + 1e-6
            sum_floor = torch.zeros(S, device=e.device).index_add_(0, src, torch.where(is_floor, e, torch.zeros_like(e)))
            sum_rest = torch.zeros(S, device=e.device).index_add_(0, src, torch.where(is_floor, torch.zeros_like(e), e))
            resid = total - sum_floor
            # normal case: above-floor cells absorb (total - floor energy). degenerate case
            # (no above-floor energy, or the floor cells alone already exceed the total):
            # scale every cell instead, so the total is still respected.
            ok = (resid > 0) & (sum_rest > 0)
            s_rest = torch.where(ok, resid / sum_rest.clamp(min=1e-30), torch.ones_like(resid))
            s_all = torch.where(ok, torch.ones_like(resid), total / (sum_floor + sum_rest).clamp(min=1e-30))
            lo, hi = scale_bounds
            sat = ((s_rest < lo) | (s_rest > hi) | (s_all < lo) | (s_all > hi))
            s_rest = s_rest.clamp(lo, hi); s_all = s_all.clamp(lo, hi)
            scale = torch.where(is_floor, s_all[src], s_all[src] * s_rest[src])
            logE = logE + scale.log()
            # showers whose total is NOT exact because the scale saturated (see docstring).
            # reported so the cap can never pass silently as "totals are exact".
            out_sat = sat
        else:
            out_sat = torch.zeros(S, dtype=torch.bool, device=g.device)

        # ENERGY CONSERVATION. A shower cannot deposit more than the incident particle carried,
        # and E_true is TRUTH conditioning at generation (noise-free), so this is a free hard
        # bound rather than a fitted one. It catches the failure the per-cell clamp cannot: a
        # very-low-E_true particle handed a normal-sized shower (measured: a photon shower at
        # 240x E_true built from perfectly ordinary cells, where no real shower exceeds 0.72).
        # Real showers sit well under 1, so this only ever touches the pathological tail.
        if e_true is not None:
            e = torch.exp(logE)
            tot_cells = torch.zeros(S, device=e.device).index_add_(0, src, e)
            over = tot_cells > e_true
            shrink = torch.where(over, e_true / tot_cells.clamp(min=1e-30), torch.ones_like(tot_cells))
            logE = logE + shrink[src].log()
            total = torch.minimum(total, e_true)
        else:
            over = torch.zeros(S, dtype=torch.bool, device=g.device)
        return {"n": n, "src": src, "core": core, "pos": pos, "logE": logE, "total": total,
                "scale_saturated": out_sat, "over_etrue": over}
