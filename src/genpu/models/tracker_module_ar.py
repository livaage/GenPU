"""Tracker v3: surface-local autoregressive hit generator (paper-style, arXiv:2512.24254).

Replaces v1's `layer_class(48) + per-layer-standardized (r,phi,z) residual` with a
MODULE-local representation:
  - discrete anchor: a detector MODULE = (volume,layer,surface), predicted HIERARCHICALLY
    as layer_class(48) then surface-slot-within-layer (masked to that layer's valid surfaces).
  - continuous part: per-module-standardized LOCAL Cartesian (x,y,z) + time residuals, which
    are small & bounded (<=~50mm) because a module is a small patch -> no room for the
    unbounded outward radial drift that v1's ±60mm layer residual suffers.

The module token pins global position; the continuous head only places the hit within the
module. Geometry (vocab, local frames, layer<->surface maps) comes from genpu.module_geometry.

Decoder / conditioning / n_hits-driven generation mirror tracker_ar.TrackerARModel verbatim.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from genpu.detector_geometry import N_LAYERS
from genpu.models.tracker_ar import (
    N_BINS_SPATIAL, SPATIAL_RANGE, _digitize, _make_bin_centers,
)

N_BINS_TIME = 64
TIME_RANGE = (-3.0, 3.0)   # module-local standardized time (was per-layer [-1,15] in v1)
NEG_INF = -1e9


class TrackerModuleARModel(nn.Module):
    """Causal transformer decoder with a hierarchical module head + local (x,y,z,time) heads."""

    def __init__(
        self,
        hierarchy: dict,          # from ModuleGeometry.build_hierarchy()
        module_means,             # (M,4) physical local-frame origin [x,y,z,time]
        module_stds,              # (M,4)
        cond_dim: int = 128,
        model_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_hits: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.max_hits = max_hits
        self.n_modules = int(module_means.shape[0])
        self.max_surf = int(hierarchy["max_surf"])

        # ---- geometry buffers ----
        self.register_buffer("_layer_of_module", torch.as_tensor(hierarchy["layer_of_module"], dtype=torch.long))
        self.register_buffer("_local_of_module", torch.as_tensor(hierarchy["local_of_module"], dtype=torch.long))
        self.register_buffer("_local_to_module", torch.as_tensor(hierarchy["local_to_module"], dtype=torch.long))
        # additive surface-logit bias: 0 where a (layer,slot) is a real module, -inf otherwise
        surf_bias = torch.where(torch.as_tensor(hierarchy["valid_mask"]),
                                torch.zeros(1), torch.full((1,), NEG_INF))
        self.register_buffer("_surf_bias", surf_bias)                       # (48, max_surf)
        self.register_buffer("_mod_mean", torch.as_tensor(module_means, dtype=torch.float32))  # (M,4)
        self.register_buffer("_mod_std", torch.as_tensor(module_stds, dtype=torch.float32))     # (M,4)
        self.register_buffer("_spatial_centers", _make_bin_centers(N_BINS_SPATIAL, *SPATIAL_RANGE))
        self.register_buffer("_time_centers", _make_bin_centers(N_BINS_TIME, *TIME_RANGE))

        # ---- input embeddings (concatenate to exactly model_dim) ----
        d8 = model_dim // 8
        self.layer_embedding = nn.Embedding(N_LAYERS + 1, d8)          # +1 BOS
        self.surf_embedding = nn.Embedding(self.max_surf + 1, model_dim - 5 * d8)   # +1 BOS
        self.x_embedding = nn.Embedding(N_BINS_SPATIAL, d8)
        self.y_embedding = nn.Embedding(N_BINS_SPATIAL, d8)
        self.z_embedding = nn.Embedding(N_BINS_SPATIAL, d8)
        self.time_embedding = nn.Embedding(N_BINS_TIME, d8)
        self.pos_embedding = nn.Embedding(max_hits + 1, model_dim)     # +1 BOS slot
        # absolute global position feed (reconstructed x,y,z), like v1's abspos_proj
        self.abspos_proj = nn.Linear(3, model_dim)
        self.bos = nn.Parameter(torch.zeros(model_dim))

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, model_dim), nn.SiLU(), nn.Linear(model_dim, model_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, nhead=n_heads, dim_feedforward=model_dim * 4,
            dropout=dropout, batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # ---- output heads ----
        self.layer_head = nn.Linear(model_dim, N_LAYERS)
        self.surface_head = nn.Linear(model_dim, self.max_surf)
        self.x_head = nn.Linear(model_dim, N_BINS_SPATIAL)
        self.y_head = nn.Linear(model_dim, N_BINS_SPATIAL)
        self.z_head = nn.Linear(model_dim, N_BINS_SPATIAL)
        self.time_head = nn.Linear(model_dim, N_BINS_TIME)

    # ---------- tokenization ----------
    def _tokenize(self, continuous: torch.Tensor):
        """(B,N,4) local-standardized [x,y,z,time] -> bin indices (each (B,N))."""
        xb = _digitize(continuous[..., 0], N_BINS_SPATIAL, *SPATIAL_RANGE)
        yb = _digitize(continuous[..., 1], N_BINS_SPATIAL, *SPATIAL_RANGE)
        zb = _digitize(continuous[..., 2], N_BINS_SPATIAL, *SPATIAL_RANGE)
        tb = _digitize(continuous[..., 3], N_BINS_TIME, *TIME_RANGE)
        return xb, yb, zb, tb

    def _embed_hits(self, module_idx, xb, yb, zb, tb):
        """Embed a hit sequence. module_idx (B,N) global module index; *b (B,N) local-coord bins."""
        lc = self._layer_of_module[module_idx]
        ls = self._local_of_module[module_idx]
        tok = torch.cat([
            self.layer_embedding(lc),
            self.surf_embedding(ls),
            self.x_embedding(xb), self.y_embedding(yb),
            self.z_embedding(zb), self.time_embedding(tb),
        ], dim=-1)
        # reconstruct global (x,y,z) from module frame + binned local residual (train==gen)
        resid = torch.stack([self._spatial_centers[xb], self._spatial_centers[yb],
                             self._spatial_centers[zb]], dim=-1)
        g = resid * self._mod_std[module_idx][..., :3] + self._mod_mean[module_idx][..., :3]
        abspos = torch.stack([g[..., 0] / 1000.0, g[..., 1] / 1000.0, g[..., 2] / 3000.0], dim=-1)
        return tok + self.abspos_proj(abspos)

    def _causal_mask(self, seq_len, device):
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def _masked_surface_logits(self, h, layer_idx):
        """surface logits with slots invalid for `layer_idx` set to -inf. layer_idx (B,N)."""
        return self.surface_head(h) + self._surf_bias[layer_idx]

    # ---------- decoder + heads ----------
    def _decode(self, module_idx, xb, yb, zb, tb, cond, mask):
        """Embed (module + coord bins), prepend BOS, run the causal decoder. Returns h (B,N,D);
        output position i predicts hit i from BOS..hit_{i-1}."""
        B, N = module_idx.shape
        device = module_idx.device
        h = self._embed_hits(module_idx, xb, yb, zb, tb)
        bos = (self.bos.unsqueeze(0) + self.abspos_proj(torch.zeros(B, 3, device=device))).unsqueeze(1)
        h = torch.cat([bos, h], dim=1)
        key_mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=device), mask], dim=1)
        seq = h.shape[1]
        h = h + self.pos_embedding(torch.arange(seq, device=device)).unsqueeze(0)
        memory = self.cond_proj(cond).unsqueeze(1)
        h = self.decoder(tgt=h, memory=memory, tgt_mask=self._causal_mask(seq, device),
                         tgt_key_padding_mask=~key_mask)
        return h[:, :N].contiguous()

    def _heads(self, h, mask_layer):
        """Apply all output heads; surface logits masked to `mask_layer`'s valid surfaces."""
        return {
            "layer_logits": self.layer_head(h),
            "surface_logits": self._masked_surface_logits(h, mask_layer),
            "x_logits": self.x_head(h), "y_logits": self.y_head(h),
            "z_logits": self.z_head(h), "time_logits": self.time_head(h),
        }

    def forward(self, module_idx, continuous, cond, mask, target_layer=None):
        """Teacher-forced pass. `target_layer` (B,N) masks the surface head to the TARGET hit's
        layer; when None it uses the input module's layer (identical under teacher forcing, but
        must be supplied when the input is scheduled-sampled and differs from the target)."""
        xb, yb, zb, tb = self._tokenize(continuous)
        h = self._decode(module_idx, xb, yb, zb, tb, cond, mask)
        mask_layer = target_layer if target_layer is not None else self._layer_of_module[module_idx]
        return self._heads(h, mask_layer)

    def loss(self, module_idx, continuous, cond, mask, n_hits, ss_prob=0.0):
        """AR loss. ss_prob>0 enables 2-pass SCHEDULED SAMPLING: pass 1 (no grad) samples the
        model's own next-hit predictions given the true history; a fraction `ss_prob` of input
        positions are replaced by those own predictions; pass 2 predicts the TRUE targets from
        that partly-self-generated history — teaching recovery from its own errors (the exposure
        bias confirmed by the teacher-forced test). Targets are always the true hits."""
        B, N = module_idx.shape
        device = module_idx.device
        tgt_layer = self._layer_of_module[module_idx]
        tgt_surf = self._local_of_module[module_idx]
        xb, yb, zb, tb = self._tokenize(continuous)

        if ss_prob > 0.0 and self.training:
            with torch.no_grad():
                h1 = self._decode(module_idx, xb, yb, zb, tb, cond, mask)
                pl = torch.distributions.Categorical(logits=self.layer_head(h1)).sample()
                ps = torch.distributions.Categorical(logits=self.surface_head(h1) + self._surf_bias[pl]).sample()
                pmod = self._local_to_module[pl, ps].clamp_min(0)
                pxb = torch.distributions.Categorical(logits=self.x_head(h1)).sample()
                pyb = torch.distributions.Categorical(logits=self.y_head(h1)).sample()
                pzb = torch.distributions.Categorical(logits=self.z_head(h1)).sample()
                ptb = torch.distributions.Categorical(logits=self.time_head(h1)).sample()
                sw = (torch.rand(B, N, device=device) < ss_prob) & mask   # replace only real hits
                in_mod = torch.where(sw, pmod, module_idx)
                in_xb = torch.where(sw, pxb, xb); in_yb = torch.where(sw, pyb, yb)
                in_zb = torch.where(sw, pzb, zb); in_tb = torch.where(sw, ptb, tb)
            h = self._decode(in_mod, in_xb, in_yb, in_zb, in_tb, cond, mask)
            out = self._heads(h, tgt_layer)          # surface mask on the TRUE target layer
        else:
            out = self._heads(self._decode(module_idx, xb, yb, zb, tb, cond, mask), tgt_layer)

        pm = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            pm[b, :int(n_hits[b].item())] = True
        fm = pm.view(-1)
        if not fm.any():
            z = torch.zeros((), device=device)
            return {"tracker_layer_loss": z, "tracker_surface_loss": z,
                    "tracker_continuous_loss": z, "tracker_ar_loss": z}

        def ce(logits, target, C):
            return F.cross_entropy(logits.reshape(-1, C)[fm], target.reshape(-1)[fm])

        layer_l = ce(out["layer_logits"], tgt_layer, N_LAYERS)
        surf_l = ce(out["surface_logits"], tgt_surf, self.max_surf)
        x_l = ce(out["x_logits"], xb, N_BINS_SPATIAL)
        y_l = ce(out["y_logits"], yb, N_BINS_SPATIAL)
        z_l = ce(out["z_logits"], zb, N_BINS_SPATIAL)
        t_l = ce(out["time_logits"], tb, N_BINS_TIME)
        cont_l = (x_l + y_l + z_l + t_l) / 4
        return {"tracker_layer_loss": layer_l, "tracker_surface_loss": surf_l,
                "tracker_continuous_loss": cont_l,
                "tracker_ar_loss": layer_l + surf_l + cont_l}

    # ---------- teacher-forced one-step prediction (exposure-bias diagnostic) ----------
    @torch.no_grad()
    def teacher_forced_modules(self, module_idx, continuous, cond, mask, layer_temp=1.0, surf_temp=1.0):
        """Sample the model's next-module prediction at each position GIVEN THE TRUE history
        (teacher forcing). Position i predicts hit i from true hits 0..i-1. Returns (B,N) module
        indices. Contrast its per-step drift with free-running generate(): if teacher-forced is
        flat and free-running climbs, the drift is exposure bias, not a mislearned conditional."""
        xb, yb, zb, tb = self._tokenize(continuous)
        h = self._decode(module_idx, xb, yb, zb, tb, cond, mask)
        layer = torch.distributions.Categorical(logits=self.layer_head(h) / layer_temp).sample()
        surf_logits = self.surface_head(h) + self._surf_bias[layer]
        surf = torch.distributions.Categorical(logits=surf_logits / surf_temp).sample()
        return self._local_to_module[layer, surf].clamp_min(0)

    # ---------- generation ----------
    @torch.no_grad()
    def generate(self, cond, n_hits, layer_temp=1.0, surf_temp=1.0, cont_temp=1.0):
        """Autoregressive, n_hits-driven (from the count head). Returns:
          hits (B,max_n,4) physical [x,y,z,time], modules (B,max_n) long."""
        B = cond.shape[0]; device = cond.device
        max_n = int(n_hits.max().item())
        if max_n == 0:
            return torch.zeros(B, 1, 4, device=device), torch.zeros(B, 1, dtype=torch.long, device=device)
        gen_mod = torch.zeros(B, max_n, dtype=torch.long, device=device)
        gx = torch.zeros(B, max_n, dtype=torch.long, device=device)
        gy = torch.zeros(B, max_n, dtype=torch.long, device=device)
        gz = torch.zeros(B, max_n, dtype=torch.long, device=device)
        gt = torch.zeros(B, max_n, dtype=torch.long, device=device)
        memory = self.cond_proj(cond).unsqueeze(1)
        bos = (self.bos.unsqueeze(0) + self.abspos_proj(torch.zeros(B, 3, device=device))).unsqueeze(1)
        arange_b = torch.arange(B, device=device)

        for step in range(max_n):
            active = n_hits > step
            if not active.any():
                break
            if step == 0:
                h = bos
            else:
                h = torch.cat([bos, self._embed_hits(
                    gen_mod[:, :step], gx[:, :step], gy[:, :step], gz[:, :step], gt[:, :step])], dim=1)
            seq_len = h.shape[1]
            h = h + self.pos_embedding(torch.arange(seq_len, device=device)).unsqueeze(0)
            h = self.decoder(tgt=h, memory=memory, tgt_mask=self._causal_mask(seq_len, device))
            last = h[:, -1]

            # 1) sample layer, 2) sample surface slot masked to that layer, 3) -> global module
            layer = torch.distributions.Categorical(
                logits=self.layer_head(last) / layer_temp).sample()                 # (B,)
            surf_logits = self.surface_head(last) + self._surf_bias[layer]          # (B,max_surf)
            surf = torch.distributions.Categorical(logits=surf_logits / surf_temp).sample()
            module = self._local_to_module[layer, surf].clamp_min(0)               # (B,)
            # 4) sample local coords
            nx = torch.distributions.Categorical(logits=self.x_head(last) / cont_temp).sample()
            ny = torch.distributions.Categorical(logits=self.y_head(last) / cont_temp).sample()
            nz = torch.distributions.Categorical(logits=self.z_head(last) / cont_temp).sample()
            nt = torch.distributions.Categorical(logits=self.time_head(last) / cont_temp).sample()

            gen_mod[:, step] = torch.where(active, module, gen_mod[:, step])
            gx[:, step] = torch.where(active, nx, gx[:, step])
            gy[:, step] = torch.where(active, ny, gy[:, step])
            gz[:, step] = torch.where(active, nz, gz[:, step])
            gt[:, step] = torch.where(active, nt, gt[:, step])

        hit_mask = torch.arange(max_n, device=device).unsqueeze(0) < n_hits.unsqueeze(1)
        resid = torch.stack([self._spatial_centers[gx], self._spatial_centers[gy],
                             self._spatial_centers[gz], self._time_centers[gt]], dim=-1)
        phys = resid * self._mod_std[gen_mod] + self._mod_mean[gen_mod]
        phys = phys * hit_mask.unsqueeze(-1).float()
        return phys, gen_mod
