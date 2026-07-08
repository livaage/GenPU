"""Autoregressive tracker hit generator with tokenized predictions.

Generates tracker hits one at a time, ordered by increasing r (inside-out).
At each step, the model predicts tokens:
1. Layer class (categorical over 48 detector layers)
2. Binned per-layer-standardized residuals for (r, phi, z, time)

Tokenization eliminates mean-regression: cross-entropy on discrete bins
learns the full distribution over positions, not just the mean. Per-layer
standardization ensures all residuals are ~N(0,1) regardless of layer type.

At generation time, bin centers are converted back to physical units:
  physical = bin_center * layer_std + layer_mean
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from genpu.detector_geometry import N_LAYERS, LAYER_MEANS, LAYER_STDS

# Bin configuration for tokenized continuous features
# r, phi, z: range [-3, 3] covers >99.9% of residuals
# time: range [-1, 15] covers the heavy right tail
N_BINS_SPATIAL = 128   # bins for r, phi, z
N_BINS_TIME = 64       # fewer bins for time (less critical)

SPATIAL_RANGE = (-3.0, 3.0)
TIME_RANGE = (-1.0, 15.0)


def _make_bin_edges(n_bins: int, lo: float, hi: float) -> torch.Tensor:
    """Create bin edges. Values outside range get clipped to first/last bin."""
    return torch.linspace(lo, hi, n_bins + 1)


def _make_bin_centers(n_bins: int, lo: float, hi: float) -> torch.Tensor:
    edges = torch.linspace(lo, hi, n_bins + 1)
    return (edges[:-1] + edges[1:]) / 2


def _digitize(values: torch.Tensor, n_bins: int, lo: float, hi: float) -> torch.Tensor:
    """Convert continuous values to bin indices (0 to n_bins-1)."""
    # Clamp to range, then scale to [0, n_bins-1]
    clamped = values.clamp(lo, hi)
    normalized = (clamped - lo) / (hi - lo)  # [0, 1]
    bins = (normalized * (n_bins - 1)).round().long()
    return bins.clamp(0, n_bins - 1)


class TrackerARModel(nn.Module):
    """Autoregressive tracker with tokenized layer + binned continuous features."""

    def __init__(
        self,
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

        # Token embeddings for each feature
        # Layer: 48 classes + 1 BOS token
        self.layer_embedding = nn.Embedding(N_LAYERS + 1, model_dim // 4)
        # Spatial bins: r, phi, z each get N_BINS_SPATIAL bins
        self.r_embedding = nn.Embedding(N_BINS_SPATIAL, model_dim // 4)
        self.phi_embedding = nn.Embedding(N_BINS_SPATIAL, model_dim // 4)
        # z and time share the remaining quarter
        self.z_embedding = nn.Embedding(N_BINS_SPATIAL, model_dim // 8)
        self.time_embedding = nn.Embedding(N_BINS_TIME, model_dim // 8)

        # Learned sequence-position embedding
        self.pos_embedding = nn.Embedding(max_hits, model_dim)

        # ABSOLUTE spatial-position input: per-layer-standardized residuals hide the
        # trajectory, so also feed each hit's physical (r, phi, z) — [r/1e3, sin phi,
        # cos phi, z/3e3] — letting causal attention reconstruct the track's helix.
        self.abspos_proj = nn.Linear(4, model_dim)

        # Particle conditioning
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # Causal transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output heads: one per token type
        self.layer_head = nn.Linear(model_dim, N_LAYERS)
        self.r_head = nn.Linear(model_dim, N_BINS_SPATIAL)
        self.phi_head = nn.Linear(model_dim, N_BINS_SPATIAL)
        self.z_head = nn.Linear(model_dim, N_BINS_SPATIAL)
        self.time_head = nn.Linear(model_dim, N_BINS_TIME)

        # Register geometry and bin centers as buffers
        self.register_buffer("_layer_means", torch.tensor(LAYER_MEANS, dtype=torch.float32))
        self.register_buffer("_layer_stds", torch.tensor(LAYER_STDS, dtype=torch.float32))
        self.register_buffer("_spatial_centers", _make_bin_centers(N_BINS_SPATIAL, *SPATIAL_RANGE))
        self.register_buffer("_time_centers", _make_bin_centers(N_BINS_TIME, *TIME_RANGE))

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def _tokenize_continuous(self, continuous: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Convert per-layer-standardized (r, phi, z, time) to bin indices.

        Args:
            continuous: (B, N, 4) float — per-layer-standardized residuals

        Returns:
            r_bins, phi_bins, z_bins, time_bins: each (B, N) long
        """
        r_bins = _digitize(continuous[:, :, 0], N_BINS_SPATIAL, *SPATIAL_RANGE)
        phi_bins = _digitize(continuous[:, :, 1], N_BINS_SPATIAL, *SPATIAL_RANGE)
        z_bins = _digitize(continuous[:, :, 2], N_BINS_SPATIAL, *SPATIAL_RANGE)
        time_bins = _digitize(continuous[:, :, 3], N_BINS_TIME, *TIME_RANGE)
        return r_bins, phi_bins, z_bins, time_bins

    def _embed_hits(
        self,
        layer_classes: torch.Tensor,
        r_bins: torch.Tensor,
        phi_bins: torch.Tensor,
        z_bins: torch.Tensor,
        time_bins: torch.Tensor,
    ) -> torch.Tensor:
        """Embed tokenized hit sequence + each hit's absolute spatial position."""
        tok = torch.cat([
            self.layer_embedding(layer_classes),
            self.r_embedding(r_bins),
            self.phi_embedding(phi_bins),
            self.z_embedding(z_bins),
            self.time_embedding(time_bins),
        ], dim=-1)
        # reconstruct physical (r, phi, z) from bins + per-layer stats (bin centers ->
        # residual -> * std + mean). Consistent at train (true) and generate (own) time.
        lc = layer_classes.clamp(0, N_LAYERS - 1)
        lm = self._layer_means[lc]; ls = self._layer_stds[lc]           # (B, N, 4): r,phi,z,time
        r_phys = self._spatial_centers[r_bins] * ls[..., 0] + lm[..., 0]
        phi_phys = self._spatial_centers[phi_bins] * ls[..., 1] + lm[..., 1]
        z_phys = self._spatial_centers[z_bins] * ls[..., 2] + lm[..., 2]
        abspos = torch.stack([r_phys / 1000.0, torch.sin(phi_phys),
                              torch.cos(phi_phys), z_phys / 3000.0], dim=-1)
        return tok + self.abspos_proj(abspos)

    def forward(
        self,
        layer_classes: torch.Tensor,
        continuous: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced forward pass.

        Args:
            layer_classes: (B, N) int — true layer class sequence
            continuous: (B, N, 4) float — per-layer-standardized (r, phi, z, time)
            cond: (B, cond_dim) particle embedding
            mask: (B, N) bool
        """
        B, N = layer_classes.shape
        device = layer_classes.device

        r_bins, phi_bins, z_bins, time_bins = self._tokenize_continuous(continuous)

        h = self._embed_hits(layer_classes, r_bins, phi_bins, z_bins, time_bins)
        pos = self.pos_embedding(torch.arange(N, device=device)).unsqueeze(0)
        h = h + pos

        memory = self.cond_proj(cond).unsqueeze(1)
        causal_mask = self._causal_mask(N, device)
        tgt_key_padding_mask = ~mask

        h = self.decoder(
            tgt=h, memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        return {
            "layer_logits": self.layer_head(h),
            "r_logits": self.r_head(h),
            "phi_logits": self.phi_head(h),
            "z_logits": self.z_head(h),
            "time_logits": self.time_head(h),
        }

    def loss(
        self,
        layer_classes: torch.Tensor,
        continuous: torch.Tensor,
        cond: torch.Tensor,
        mask: torch.Tensor,
        n_hits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute AR loss: CE for all token types.

        At position i, predict tokens for position i+1.
        """
        B, N = layer_classes.shape
        device = layer_classes.device

        out = self.forward(layer_classes, continuous, cond, mask)

        # Tokenize targets
        r_bins, phi_bins, z_bins, time_bins = self._tokenize_continuous(continuous)

        # Targets: shifted by 1
        target_layers = torch.zeros_like(layer_classes)
        target_layers[:, :-1] = layer_classes[:, 1:]
        target_r = torch.zeros_like(r_bins)
        target_r[:, :-1] = r_bins[:, 1:]
        target_phi = torch.zeros_like(phi_bins)
        target_phi[:, :-1] = phi_bins[:, 1:]
        target_z = torch.zeros_like(z_bins)
        target_z[:, :-1] = z_bins[:, 1:]
        target_time = torch.zeros_like(time_bins)
        target_time[:, :-1] = time_bins[:, 1:]

        # Valid prediction mask: positions 0..n_hits-2
        pred_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            nh = n_hits[b].item()
            if nh > 1:
                pred_mask[b, :nh - 1] = True

        if not pred_mask.any():
            zero = torch.tensor(0.0, device=device)
            return {
                "tracker_layer_loss": zero,
                "tracker_continuous_loss": zero,
                "tracker_ar_loss": zero,
            }

        flat_mask = pred_mask.view(-1)

        layer_loss = F.cross_entropy(
            out["layer_logits"].view(-1, N_LAYERS)[flat_mask],
            target_layers.view(-1)[flat_mask],
        )
        r_loss = F.cross_entropy(
            out["r_logits"].view(-1, N_BINS_SPATIAL)[flat_mask],
            target_r.view(-1)[flat_mask],
        )
        phi_loss = F.cross_entropy(
            out["phi_logits"].view(-1, N_BINS_SPATIAL)[flat_mask],
            target_phi.view(-1)[flat_mask],
        )
        z_loss = F.cross_entropy(
            out["z_logits"].view(-1, N_BINS_SPATIAL)[flat_mask],
            target_z.view(-1)[flat_mask],
        )
        time_loss = F.cross_entropy(
            out["time_logits"].view(-1, N_BINS_TIME)[flat_mask],
            target_time.view(-1)[flat_mask],
        )

        continuous_loss = (r_loss + phi_loss + z_loss + time_loss) / 4

        return {
            "tracker_layer_loss": layer_loss,
            "tracker_continuous_loss": continuous_loss,
            "tracker_ar_loss": layer_loss + continuous_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        cond: torch.Tensor,
        n_hits: torch.Tensor,
        layer_temp: float = 1.0,
        cont_temp: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate tracker hits in physical units.

        layer_temp / cont_temp: sampling temperatures (>1 broadens) for the layer
        head and the continuous (r,phi,z,time) heads respectively. Free-running AR
        under-disperses (event gate: narrowed residuals + occupancy drift); temp>1
        counteracts the peakedness.

        Returns:
            hits: (B, N_max, 4) — physical (r, phi, z, time)
            layer_classes: (B, N_max) — predicted layer class per hit
        """
        B = cond.shape[0]
        device = cond.device
        max_n = n_hits.max().item()
        if max_n == 0:
            return (torch.zeros(B, 1, 4, device=device),
                    torch.zeros(B, 1, dtype=torch.long, device=device))

        gen_layers = torch.full((B, max_n), N_LAYERS, dtype=torch.long, device=device)
        gen_r = torch.zeros(B, max_n, dtype=torch.long, device=device)
        gen_phi = torch.zeros(B, max_n, dtype=torch.long, device=device)
        gen_z = torch.zeros(B, max_n, dtype=torch.long, device=device)
        gen_time = torch.zeros(B, max_n, dtype=torch.long, device=device)

        memory = self.cond_proj(cond).unsqueeze(1)

        for step in range(max_n):
            active = n_hits > step
            if not active.any():
                break

            seq_len = step + 1
            h = self._embed_hits(
                gen_layers[:, :seq_len],
                gen_r[:, :seq_len],
                gen_phi[:, :seq_len],
                gen_z[:, :seq_len],
                gen_time[:, :seq_len],
            )
            pos = self.pos_embedding(torch.arange(seq_len, device=device)).unsqueeze(0)
            h = h + pos

            causal_mask = self._causal_mask(seq_len, device)
            h = self.decoder(tgt=h, memory=memory, tgt_mask=causal_mask)

            last_h = h[:, -1]

            # Sample all tokens (logits / temperature broadens the distribution)
            next_layer = torch.distributions.Categorical(logits=self.layer_head(last_h) / layer_temp).sample()
            next_r = torch.distributions.Categorical(logits=self.r_head(last_h) / cont_temp).sample()
            next_phi = torch.distributions.Categorical(logits=self.phi_head(last_h) / cont_temp).sample()
            next_z = torch.distributions.Categorical(logits=self.z_head(last_h) / cont_temp).sample()
            next_time = torch.distributions.Categorical(logits=self.time_head(last_h) / cont_temp).sample()

            gen_layers[:, step] = torch.where(active, next_layer, gen_layers[:, step])
            gen_r[:, step] = torch.where(active, next_r, gen_r[:, step])
            gen_phi[:, step] = torch.where(active, next_phi, gen_phi[:, step])
            gen_z[:, step] = torch.where(active, next_z, gen_z[:, step])
            gen_time[:, step] = torch.where(active, next_time, gen_time[:, step])

        hit_mask = torch.arange(max_n, device=device).unsqueeze(0) < n_hits.unsqueeze(1)

        # Convert bin indices to residuals (bin centers)
        r_resid = self._spatial_centers[gen_r]
        phi_resid = self._spatial_centers[gen_phi]
        z_resid = self._spatial_centers[gen_z]
        time_resid = self._time_centers[gen_time]

        residuals = torch.stack([r_resid, phi_resid, z_resid, time_resid], dim=-1)

        # Denormalize: physical = residual * layer_std + layer_mean
        layer_idx = gen_layers.clamp(0, N_LAYERS - 1)
        layer_means = self._layer_means[layer_idx]
        layer_stds = self._layer_stds[layer_idx]
        hits_physical = residuals * layer_stds + layer_means
        hits_physical = hits_physical * hit_mask.unsqueeze(-1).float()

        return hits_physical, gen_layers
