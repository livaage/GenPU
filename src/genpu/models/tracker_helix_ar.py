"""Tracker v4: helix-anchored AR hit generator.

Each hit = helix_reference(layer) + small DEVIATION. The fixed-momentum helix (computed once from
initial pT/eta/phi/charge/vertex, B=3.07T; genpu.helix) pins the trajectory, so z-vs-r and phi-vs-r
coherence hold BY CONSTRUCTION and the model only predicts the bounded (dx,dy,dz) deviation (~mm).
No surface head, no module vocabulary — the helix subsumes them.

Per step: layer(48) -> evaluate the (precomputed, per-particle) helix reference at that layer ->
deviation heads (dx,dy,dz binned in mm) + time. Position = helix_ref[layer] + deviation. Decoder,
conditioning, count/stop, and scheduled sampling mirror v3.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from genpu.detector_geometry import N_LAYERS
from genpu.models.tracker_ar import N_BINS_SPATIAL, _digitize, _make_bin_centers

N_BINS_TIME = 64
DEV_XY = (-160.0, 160.0)   # mm deviation from helix (median ~2, heavy tail: low-pT curly outer hits)
DEV_Z = (-500.0, 500.0)    # mm z deviation (median ~4; tail clipped at the edge bin)
TIME_RANGE = (-3.0, 6.0)   # standardized time (buffers set from data)


class TrackerHelixARModel(nn.Module):
    def __init__(self, cond_dim=64, model_dim=128, n_layers=4, n_heads=4, max_hits=32, dropout=0.0):
        super().__init__()
        self.max_hits = max_hits
        d = model_dim
        self.layer_embedding = nn.Embedding(N_LAYERS + 1, d // 8)     # +1 BOS
        self.dx_embedding = nn.Embedding(N_BINS_SPATIAL, d // 4)
        self.dy_embedding = nn.Embedding(N_BINS_SPATIAL, d // 4)
        self.dz_embedding = nn.Embedding(N_BINS_SPATIAL, d // 4)
        self.time_embedding = nn.Embedding(N_BINS_TIME, d // 8)
        self.pos_embedding = nn.Embedding(max_hits + 1, d)
        self.abspos_proj = nn.Linear(3, d)                            # actual position = ref+dev
        self.bos = nn.Parameter(torch.zeros(d))
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, d), nn.SiLU(), nn.Linear(d, d))
        dl = nn.TransformerDecoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d * 4,
                                        dropout=dropout, batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(dl, num_layers=n_layers)
        self.layer_head = nn.Linear(d, N_LAYERS)
        self.dx_head = nn.Linear(d, N_BINS_SPATIAL)
        self.dy_head = nn.Linear(d, N_BINS_SPATIAL)
        self.dz_head = nn.Linear(d, N_BINS_SPATIAL)
        self.time_head = nn.Linear(d, N_BINS_TIME)
        self.register_buffer("_dxy_centers", _make_bin_centers(N_BINS_SPATIAL, *DEV_XY))
        self.register_buffer("_dz_centers", _make_bin_centers(N_BINS_SPATIAL, *DEV_Z))
        self.register_buffer("_time_centers", _make_bin_centers(N_BINS_TIME, *TIME_RANGE))
        self.register_buffer("time_mean", torch.zeros(1))
        self.register_buffer("time_std", torch.ones(1))

    # ---- tokenization ----
    def _tokenize(self, deviation, time_std):
        xb = _digitize(deviation[..., 0], N_BINS_SPATIAL, *DEV_XY)
        yb = _digitize(deviation[..., 1], N_BINS_SPATIAL, *DEV_XY)
        zb = _digitize(deviation[..., 2], N_BINS_SPATIAL, *DEV_Z)
        tb = _digitize(time_std, N_BINS_TIME, *TIME_RANGE)
        return xb, yb, zb, tb

    def _ref_of(self, helix_ref, layer):
        """helix_ref (B,48,3), layer (B,N) -> per-hit reference (B,N,3)."""
        B = layer.shape[0]
        return helix_ref[torch.arange(B, device=layer.device)[:, None], layer]

    def _embed_hits(self, layer, xb, yb, zb, tb, helix_ref):
        tok = torch.cat([self.layer_embedding(layer), self.dx_embedding(xb), self.dy_embedding(yb),
                         self.dz_embedding(zb), self.time_embedding(tb)], dim=-1)
        dev = torch.stack([self._dxy_centers[xb], self._dxy_centers[yb], self._dz_centers[zb]], dim=-1)
        pos = self._ref_of(helix_ref, layer) + dev                    # actual (x,y,z)
        abspos = torch.stack([pos[..., 0] / 1000.0, pos[..., 1] / 1000.0, pos[..., 2] / 3000.0], dim=-1)
        return tok + self.abspos_proj(abspos)

    def _causal_mask(self, n, device):
        return torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)

    def _decode(self, layer, xb, yb, zb, tb, cond, mask, helix_ref):
        B, N = layer.shape
        device = layer.device
        h = self._embed_hits(layer, xb, yb, zb, tb, helix_ref)
        bos = (self.bos.unsqueeze(0) + self.abspos_proj(torch.zeros(B, 3, device=device))).unsqueeze(1)
        h = torch.cat([bos, h], dim=1)
        key_mask = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=device), mask], dim=1)
        seq = h.shape[1]
        h = h + self.pos_embedding(torch.arange(seq, device=device)).unsqueeze(0)
        memory = self.cond_proj(cond).unsqueeze(1)
        h = self.decoder(tgt=h, memory=memory, tgt_mask=self._causal_mask(seq, device),
                         tgt_key_padding_mask=~key_mask)
        return h[:, :N].contiguous()

    def _heads(self, h):
        return {"layer_logits": self.layer_head(h), "dx_logits": self.dx_head(h),
                "dy_logits": self.dy_head(h), "dz_logits": self.dz_head(h),
                "time_logits": self.time_head(h)}

    def forward(self, layer, deviation, time_std, cond, mask, helix_ref):
        xb, yb, zb, tb = self._tokenize(deviation, time_std)
        return self._heads(self._decode(layer, xb, yb, zb, tb, cond, mask, helix_ref))

    def loss(self, layer, deviation, time_std, cond, mask, n_hits, helix_ref, ss_prob=0.0):
        B, N = layer.shape
        device = layer.device
        xb, yb, zb, tb = self._tokenize(deviation, time_std)
        if ss_prob > 0.0 and self.training:
            with torch.no_grad():
                h1 = self._decode(layer, xb, yb, zb, tb, cond, mask, helix_ref)
                o1 = self._heads(h1)
                pl = torch.distributions.Categorical(logits=o1["layer_logits"]).sample()
                pxb = torch.distributions.Categorical(logits=o1["dx_logits"]).sample()
                pyb = torch.distributions.Categorical(logits=o1["dy_logits"]).sample()
                pzb = torch.distributions.Categorical(logits=o1["dz_logits"]).sample()
                ptb = torch.distributions.Categorical(logits=o1["time_logits"]).sample()
                sw = (torch.rand(B, N, device=device) < ss_prob) & mask
                il = torch.where(sw, pl, layer)
                ix = torch.where(sw, pxb, xb); iy = torch.where(sw, pyb, yb)
                iz = torch.where(sw, pzb, zb); it = torch.where(sw, ptb, tb)
            out = self._heads(self._decode(il, ix, iy, iz, it, cond, mask, helix_ref))
        else:
            out = self._heads(self._decode(layer, xb, yb, zb, tb, cond, mask, helix_ref))

        pm = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            pm[b, :int(n_hits[b].item())] = True
        fm = pm.view(-1)
        if not fm.any():
            z = torch.zeros((), device=device)
            return {"tracker_layer_loss": z, "tracker_dev_loss": z, "tracker_time_loss": z, "tracker_ar_loss": z}

        def ce(lg, tg, C):
            return F.cross_entropy(lg.reshape(-1, C)[fm], tg.reshape(-1)[fm])
        layer_l = ce(out["layer_logits"], layer, N_LAYERS)
        dev_l = (ce(out["dx_logits"], xb, N_BINS_SPATIAL) + ce(out["dy_logits"], yb, N_BINS_SPATIAL)
                 + ce(out["dz_logits"], zb, N_BINS_SPATIAL)) / 3
        time_l = ce(out["time_logits"], tb, N_BINS_TIME)
        return {"tracker_layer_loss": layer_l, "tracker_dev_loss": dev_l, "tracker_time_loss": time_l,
                "tracker_ar_loss": layer_l + dev_l + time_l}

    @torch.no_grad()
    def generate(self, cond, n_hits, helix_ref, layer_temp=1.0, dev_temp=1.0):
        """Returns hits (B,max_n,4) physical [x,y,z,time_std], layers (B,max_n)."""
        B = cond.shape[0]; device = cond.device
        max_n = int(n_hits.max().item())
        if max_n == 0:
            return torch.zeros(B, 1, 4, device=device), torch.zeros(B, 1, dtype=torch.long, device=device)
        gl = torch.zeros(B, max_n, dtype=torch.long, device=device)
        gx = torch.zeros(B, max_n, dtype=torch.long, device=device)
        gy = torch.zeros(B, max_n, dtype=torch.long, device=device)
        gz = torch.zeros(B, max_n, dtype=torch.long, device=device)
        gt = torch.zeros(B, max_n, dtype=torch.long, device=device)
        memory = self.cond_proj(cond).unsqueeze(1)
        bos = (self.bos.unsqueeze(0) + self.abspos_proj(torch.zeros(B, 3, device=device))).unsqueeze(1)
        for step in range(max_n):
            active = n_hits > step
            if not active.any():
                break
            if step == 0:
                h = bos
            else:
                h = torch.cat([bos, self._embed_hits(gl[:, :step], gx[:, :step], gy[:, :step],
                                                     gz[:, :step], gt[:, :step], helix_ref)], dim=1)
            seq = h.shape[1]
            h = h + self.pos_embedding(torch.arange(seq, device=device)).unsqueeze(0)
            h = self.decoder(tgt=h, memory=memory, tgt_mask=self._causal_mask(seq, device))
            o = self._heads(h[:, -1:])
            def samp(key, t):
                return torch.distributions.Categorical(logits=o[key][:, 0] / t).sample()
            nl = samp("layer_logits", layer_temp)
            nx = samp("dx_logits", dev_temp); ny = samp("dy_logits", dev_temp)
            nz = samp("dz_logits", dev_temp); nt = samp("time_logits", dev_temp)
            gl[:, step] = torch.where(active, nl, gl[:, step]); gx[:, step] = torch.where(active, nx, gx[:, step])
            gy[:, step] = torch.where(active, ny, gy[:, step]); gz[:, step] = torch.where(active, nz, gz[:, step])
            gt[:, step] = torch.where(active, nt, gt[:, step])
        hit_mask = torch.arange(max_n, device=device).unsqueeze(0) < n_hits.unsqueeze(1)
        dev = torch.stack([self._dxy_centers[gx], self._dxy_centers[gy], self._dz_centers[gz]], dim=-1)
        pos = self._ref_of(helix_ref, gl) + dev
        tvals = self._time_centers[gt]
        out = torch.cat([pos, tvals.unsqueeze(-1)], dim=-1) * hit_mask.unsqueeze(-1).float()
        return out, gl
