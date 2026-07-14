"""Tracker v2: state-carrying autoregressive hit generator.

Extends the v1 tokenized decoder ([tracker_ar.py]) with the two things v1 lacks (see
tracker_v2_design.md): an explicit, updated, supervised DIRECTION state, and a STOP head.

Per step the model predicts a group of tokens for the next hit:
  layer (48) | r,phi,z resid (512 bins) | time (64 bins)   -- v1 position, per-layer encoding
  dir_theta, dir_alpha (256 bins)                            -- NEW: momentum-direction STATE
  stop (binary)                                              -- NEW: END the track

The predicted direction is embedded and fed back into the next step's input (the carried
state), alongside the absolute-position feedback. The position head is a LEARNED head
conditioned on the decoder state (which now carries the explicit direction) -- there is NO
analytic geometry anywhere (hard constraint, design doc 4a). Generation stops when `stop`
fires -- truth n_hits is NOT used.

v1 encodings (per-layer standardized position, abspos, vertex seed, bin config) are reused
verbatim so the direction state is the only structural change under test.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from genpu.detector_geometry import N_LAYERS, LAYER_MEANS, LAYER_STDS
from genpu.models.tracker_ar import (
    N_BINS_SPATIAL, N_BINS_TIME, SPATIAL_RANGE, TIME_RANGE,
    _make_bin_centers, _digitize,
)

# direction angles: theta in [0, pi], alpha in [-pi, pi]; one shared 256-bin scheme over [-pi, pi]
N_BINS_DIR = 256
DIR_RANGE = (-3.1416, 3.1416)


class TrackerStateARModel(nn.Module):
    """Causal transformer decoder with tokenized position + an explicit direction state + stop."""

    def __init__(self, cond_dim=64, model_dim=128, n_layers=4, n_heads=4, max_hits=32, dropout=0.0,
                 state_feedback=True):
        super().__init__()
        self.model_dim = model_dim
        self.max_hits = max_hits
        # state_feedback: feed the predicted direction back into the next step's input. The 30k/70k
        # eval showed this DRIFTS (exposure bias: trained on true dir, generates from its own noisy
        # one) and net-hurts coherence, while the direction as AUXILIARY supervision helps. So
        # default-off now: keep the dir head (supervision) but don't feed it back.
        self.state_feedback = state_feedback

        # --- token embeddings (position, v1) ---
        self.layer_embedding = nn.Embedding(N_LAYERS + 1, model_dim // 4)
        self.r_embedding = nn.Embedding(N_BINS_SPATIAL, model_dim // 4)
        self.phi_embedding = nn.Embedding(N_BINS_SPATIAL, model_dim // 4)
        self.z_embedding = nn.Embedding(N_BINS_SPATIAL, model_dim // 8)
        self.time_embedding = nn.Embedding(N_BINS_TIME, model_dim // 8)
        # --- direction-state embeddings (NEW) ---
        self.dth_embedding = nn.Embedding(N_BINS_DIR, model_dim // 2)
        self.dal_embedding = nn.Embedding(N_BINS_DIR, model_dim // 2)

        self.pos_embedding = nn.Embedding(max_hits + 1, model_dim)          # +1 for the vertex seed
        # abspos feedback (reconstructed physical r, sin/cos phi, z)
        self.abspos_proj = nn.Linear(4, model_dim)
        self.start_token = nn.Parameter(torch.zeros(model_dim))
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))

        dl = nn.TransformerDecoderLayer(d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim * 4,
                                        dropout=dropout, batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(dl, num_layers=n_layers)

        # --- output heads ---
        self.layer_head = nn.Linear(model_dim, N_LAYERS)
        self.r_head = nn.Linear(model_dim, N_BINS_SPATIAL)
        self.phi_head = nn.Linear(model_dim, N_BINS_SPATIAL)
        self.z_head = nn.Linear(model_dim, N_BINS_SPATIAL)
        self.time_head = nn.Linear(model_dim, N_BINS_TIME)
        self.dth_head = nn.Linear(model_dim, N_BINS_DIR)          # NEW: direction state
        self.dal_head = nn.Linear(model_dim, N_BINS_DIR)          # NEW
        self.stop_head = nn.Linear(model_dim, 1)                  # NEW: END the track

        self.register_buffer("_layer_means", torch.tensor(LAYER_MEANS, dtype=torch.float32))
        self.register_buffer("_layer_stds", torch.tensor(LAYER_STDS, dtype=torch.float32))
        self.register_buffer("_spatial_centers", _make_bin_centers(N_BINS_SPATIAL, *SPATIAL_RANGE))
        self.register_buffer("_time_centers", _make_bin_centers(N_BINS_TIME, *TIME_RANGE))
        self.register_buffer("_dir_centers", _make_bin_centers(N_BINS_DIR, *DIR_RANGE))

    def _causal_mask(self, n, device):
        return torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)

    def _tokenize(self, continuous, direction):
        """continuous (B,N,4) per-layer resid; direction (B,N,2) angles -> all bin indices."""
        rb = _digitize(continuous[..., 0], N_BINS_SPATIAL, *SPATIAL_RANGE)
        pb = _digitize(continuous[..., 1], N_BINS_SPATIAL, *SPATIAL_RANGE)
        zb = _digitize(continuous[..., 2], N_BINS_SPATIAL, *SPATIAL_RANGE)
        tb = _digitize(continuous[..., 3], N_BINS_TIME, *TIME_RANGE)
        thb = _digitize(direction[..., 0], N_BINS_DIR, *DIR_RANGE)
        alb = _digitize(direction[..., 1], N_BINS_DIR, *DIR_RANGE)
        return rb, pb, zb, tb, thb, alb

    def _embed(self, layer, rb, pb, zb, tb, thb, alb):
        """Embed a hit: position tokens + abspos feedback + DIRECTION-STATE feedback."""
        tok = torch.cat([self.layer_embedding(layer), self.r_embedding(rb), self.phi_embedding(pb),
                         self.z_embedding(zb), self.time_embedding(tb)], dim=-1)
        lc = layer.clamp(0, N_LAYERS - 1)
        lm, ls = self._layer_means[lc], self._layer_stds[lc]
        r_phys = self._spatial_centers[rb] * ls[..., 0] + lm[..., 0]
        phi_phys = self._spatial_centers[pb] * ls[..., 1] + lm[..., 1]
        z_phys = self._spatial_centers[zb] * ls[..., 2] + lm[..., 2]
        abspos = torch.stack([r_phys / 1000.0, torch.sin(phi_phys), torch.cos(phi_phys), z_phys / 3000.0], dim=-1)
        h = tok + self.abspos_proj(abspos)
        if self.state_feedback:                                       # carried state (default off; drifts)
            h = h + torch.cat([self.dth_embedding(thb), self.dal_embedding(alb)], dim=-1)
        return h

    def _vertex_embed(self, vertex_pos):
        vr, vz = vertex_pos[:, 0], vertex_pos[:, 1]
        zero = torch.zeros_like(vr)
        abspos = torch.stack([vr / 1000.0, zero, zero, vz / 3000.0], dim=-1)
        return self.start_token.unsqueeze(0) + self.abspos_proj(abspos)

    def _run(self, h, cond, device):
        seq = h.shape[1]
        h = h + self.pos_embedding(torch.arange(seq, device=device)).unsqueeze(0)
        return self.decoder(tgt=h, memory=self.cond_proj(cond).unsqueeze(1), tgt_mask=self._causal_mask(seq, device))

    def loss(self, layer, continuous, direction, cond, n_hits, vertex_pos, stop_pos_weight=1.0):
        """Teacher-forced. Sequence = [vertex, hit_0..hit_{N-1}]; output i predicts hit i (+ stop)."""
        B, N = layer.shape
        device = layer.device
        rb, pb, zb, tb, thb, alb = self._tokenize(continuous, direction)
        hemb = self._embed(layer, rb, pb, zb, tb, thb, alb)
        v = self._vertex_embed(vertex_pos).unsqueeze(1)
        h = self._run(torch.cat([v, hemb], dim=1), cond, device)[:, :N]

        pm = torch.zeros(B, N, dtype=torch.bool, device=device)
        stop_t = torch.zeros(B, N, device=device)
        for b in range(B):
            nh = int(n_hits[b].item())
            pm[b, :nh] = True
            if nh > 0:
                stop_t[b, nh - 1] = 1.0                    # stop fires on the last real hit
        fm = pm.view(-1)

        def ce(head_out, tgt, nb):
            return F.cross_entropy(head_out.reshape(-1, nb)[fm], tgt.reshape(-1)[fm])
        layer_l = ce(self.layer_head(h), layer, N_LAYERS)
        pos_l = (ce(self.r_head(h), rb, N_BINS_SPATIAL) + ce(self.phi_head(h), pb, N_BINS_SPATIAL)
                 + ce(self.z_head(h), zb, N_BINS_SPATIAL) + ce(self.time_head(h), tb, N_BINS_TIME)) / 4
        dir_l = (ce(self.dth_head(h), thb, N_BINS_DIR) + ce(self.dal_head(h), alb, N_BINS_DIR)) / 2
        stop_l = F.binary_cross_entropy_with_logits(
            self.stop_head(h).squeeze(-1).reshape(-1)[fm], stop_t.reshape(-1)[fm],
            pos_weight=torch.tensor(stop_pos_weight, device=layer.device))   # positives (stop=1) are rare
        return {"layer": layer_l, "pos": pos_l, "dir": dir_l, "stop": stop_l,
                "total": layer_l + pos_l + dir_l + stop_l}

    @torch.no_grad()
    def generate(self, cond, vertex_pos, max_hits=None, cont_temp=1.0, ablate_state=False):
        """Autoregressive: stop when the STOP head fires (no truth n_hits). Returns
        (hits (B,M,4) physical r,phi,z,time, layers (B,M), n_gen (B,))."""
        B, device = cond.shape[0], cond.device
        M = max_hits or self.max_hits
        gl = torch.full((B, M), N_LAYERS, dtype=torch.long, device=device)
        gr = torch.zeros(B, M, dtype=torch.long, device=device); gp = gr.clone(); gz = gr.clone()
        gt = gr.clone(); gth = gr.clone(); gal = gr.clone()
        alive = torch.ones(B, dtype=torch.bool, device=device)
        n_gen = torch.zeros(B, dtype=torch.long, device=device)
        vtok = self._vertex_embed(vertex_pos).unsqueeze(1)
        for step in range(M):
            if step == 0:
                h = vtok
            else:
                emb = self._embed(gl[:, :step], gr[:, :step], gp[:, :step], gz[:, :step],
                                  gt[:, :step], gth[:, :step], gal[:, :step])
                if ablate_state:
                    emb = emb - torch.cat([self.dth_embedding(gth[:, :step]),
                                           self.dal_embedding(gal[:, :step])], dim=-1)
                h = torch.cat([vtok, emb], dim=1)
            last = self._run(h, cond, device)[:, -1]
            def samp(head, temp):
                return torch.distributions.Categorical(logits=head(last) / temp).sample()
            nl = samp(self.layer_head, 1.0); nr = samp(self.r_head, cont_temp); np_ = samp(self.phi_head, cont_temp)
            nz = samp(self.z_head, cont_temp); nt = samp(self.time_head, cont_temp)
            nth = samp(self.dth_head, 1.0); nal = samp(self.dal_head, 1.0)
            stop = torch.sigmoid(self.stop_head(last).squeeze(-1)) > torch.rand(B, device=device)
            for g, val in [(gl, nl), (gr, nr), (gp, np_), (gz, nz), (gt, nt), (gth, nth), (gal, nal)]:
                g[:, step] = torch.where(alive, val, g[:, step])
            n_gen = torch.where(alive, n_gen + 1, n_gen)
            alive = alive & (~stop)
            if not alive.any():
                break
        hit_mask = torch.arange(M, device=device).unsqueeze(0) < n_gen.unsqueeze(1)
        resid = torch.stack([self._spatial_centers[gr], self._spatial_centers[gp],
                             self._spatial_centers[gz], self._time_centers[gt]], dim=-1)
        li = gl.clamp(0, N_LAYERS - 1)
        phys = resid * self._layer_stds[li] + self._layer_means[li]
        phys = phys * hit_mask.unsqueeze(-1).float()
        return phys, gl, n_gen


class TrackerStateModel(nn.Module):
    """Shared conditioning + v2 state-carrying AR, with cont (un)standardisation buffers."""

    def __init__(self, norm, embed_dim=64, model_dim=128, n_layers=4, n_heads=4, max_hits=32,
                 use_pdg=True, state_feedback=True):
        super().__init__()
        from genpu.conditioning import ParticleConditioning
        self.cond = ParticleConditioning(embed_dim=embed_dim, use_pdg=use_pdg)
        self.tracker = TrackerStateARModel(cond_dim=embed_dim, model_dim=model_dim, n_layers=n_layers,
                                           n_heads=n_heads, max_hits=max_hits, state_feedback=state_feedback)
        for k, v in norm.items():
            self.register_buffer(k, torch.as_tensor(v, dtype=torch.float32))

    def std_cont(self, cont):
        return (cont - self.cont_mean) / self.cont_std

    def cond_embed(self, cont_std, pdg):
        return self.cond(cont_std, pdg)


if __name__ == "__main__":
    import numpy as np
    from genpu.conditioning import ParticleConditioning
    torch.manual_seed(0)
    m = TrackerStateARModel(cond_dim=64)
    cond = ParticleConditioning(embed_dim=64)
    B, N = 4, 6
    cont = torch.randn(B, 7); pdg = torch.randint(0, 17, (B,))
    ce = cond(cont, pdg)
    layer = torch.randint(0, 48, (B, N)); continuous = torch.randn(B, N, 4)
    direction = torch.rand(B, N, 2) * 6 - 3; nh = torch.full((B,), N); vtx = cont[:, [5, 6]]
    out = m.loss(layer, continuous, direction, ce, nh, vtx)
    print("loss", {k: round(float(v), 3) for k, v in out.items()})
    phys, gl, ng = m.generate(ce, vtx, max_hits=10)
    print("gen", tuple(phys.shape), "n_gen", ng.tolist())
