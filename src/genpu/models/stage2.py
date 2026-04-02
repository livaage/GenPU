"""Stage 2: Per-particle detector response model.

Given a particle's kinematics, generates:
1. Number of tracker/calo hits (categorical)
2. Tracker hit features (diffusion)
3. Calo hit features (diffusion)

Training: jointly optimize hit count CE loss + diffusion MSE loss.
Generation: predict counts, then run diffusion conditioned on particle.
"""

import torch
import torch.nn as nn

from genpu.models.particle_encoder import ParticleEncoder
from genpu.models.hit_count_head import HitCountHead
from genpu.models.hit_denoiser import HitDenoiser
from genpu.models.diffusion import DiffusionSchedule


class Stage2Model(nn.Module):
    """Full Stage 2 model combining encoder, hit counts, and diffusion."""

    def __init__(
        self,
        embed_dim: int = 128,
        tracker_hit_dim: int = 6,   # r, phi, z, time, volume_id, layer_id
        calo_hit_dim: int = 5,      # eta, phi, log_energy, contrib_frac, detector
        max_tracker_hits: int = 32,
        max_calo_hits: int = 64,
        n_denoiser_layers: int = 4,
        n_heads: int = 4,
        diffusion_timesteps: int = 1000,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = ParticleEncoder(embed_dim=embed_dim)
        self.hit_count = HitCountHead(
            embed_dim=embed_dim,
            max_tracker_hits=max_tracker_hits,
            max_calo_hits=max_calo_hits,
        )

        self.tracker_denoiser = HitDenoiser(
            hit_dim=tracker_hit_dim,
            cond_dim=embed_dim,
            model_dim=embed_dim,
            n_layers=n_denoiser_layers,
            n_heads=n_heads,
            timesteps=diffusion_timesteps,
            dropout=dropout,
        )

        self.calo_denoiser = HitDenoiser(
            hit_dim=calo_hit_dim,
            cond_dim=embed_dim,
            model_dim=embed_dim,
            n_layers=n_denoiser_layers,
            n_heads=n_heads,
            timesteps=diffusion_timesteps,
            dropout=dropout,
        )

        self.schedule = DiffusionSchedule(timesteps=diffusion_timesteps)

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute all losses for one training batch.

        Args:
            batch: from ParticleHitDataset.collate_fn, containing:
                particle_features: (B, 6)
                tracker_hits: (B, N_trk, 6)
                tracker_mask: (B, N_trk)
                calo_hits: (B, N_cal, 5)
                calo_mask: (B, N_cal)
                n_tracker: (B,)
                n_calo: (B,)

        Returns:
            dict of loss components + total loss
        """
        pf = batch["particle_features"]
        B = pf.shape[0]
        device = pf.device

        # Encode particles
        emb = self.encoder(pf)

        # Hit count loss
        count_losses = self.hit_count.loss(emb, batch["n_tracker"], batch["n_calo"])

        # Diffusion losses — only on particles that have hits
        losses = dict(count_losses)

        # Tracker diffusion
        trk_has_hits = batch["n_tracker"] > 0
        if trk_has_hits.any():
            trk_emb = emb[trk_has_hits]
            trk_x0 = batch["tracker_hits"][trk_has_hits]
            trk_mask = batch["tracker_mask"][trk_has_hits]

            t = torch.randint(0, self.schedule.timesteps, (trk_emb.shape[0],), device=device)
            noise = torch.randn_like(trk_x0)
            trk_xt = self.schedule.q_sample(trk_x0, t, noise)
            trk_xt = trk_xt * trk_mask.unsqueeze(-1).float()

            noise_pred = self.tracker_denoiser(trk_xt, t, trk_emb, trk_mask)

            # MSE loss only on valid (unmasked) positions
            trk_diff_loss = ((noise_pred - noise) ** 2 * trk_mask.unsqueeze(-1).float()).sum()
            trk_diff_loss = trk_diff_loss / trk_mask.sum().clamp(min=1)
            losses["tracker_diffusion_loss"] = trk_diff_loss
        else:
            losses["tracker_diffusion_loss"] = torch.tensor(0.0, device=device)

        # Calo diffusion
        cal_has_hits = batch["n_calo"] > 0
        if cal_has_hits.any():
            cal_emb = emb[cal_has_hits]
            cal_x0 = batch["calo_hits"][cal_has_hits]
            cal_mask = batch["calo_mask"][cal_has_hits]

            t = torch.randint(0, self.schedule.timesteps, (cal_emb.shape[0],), device=device)
            noise = torch.randn_like(cal_x0)
            cal_xt = self.schedule.q_sample(cal_x0, t, noise)
            cal_xt = cal_xt * cal_mask.unsqueeze(-1).float()

            noise_pred = self.calo_denoiser(cal_xt, t, cal_emb, cal_mask)

            cal_diff_loss = ((noise_pred - noise) ** 2 * cal_mask.unsqueeze(-1).float()).sum()
            cal_diff_loss = cal_diff_loss / cal_mask.sum().clamp(min=1)
            losses["calo_diffusion_loss"] = cal_diff_loss
        else:
            losses["calo_diffusion_loss"] = torch.tensor(0.0, device=device)

        losses["total_loss"] = (
            losses["total_count_loss"]
            + losses["tracker_diffusion_loss"]
            + losses["calo_diffusion_loss"]
        )
        return losses

    @torch.no_grad()
    def generate(
        self,
        particle_features: torch.Tensor,
        n_steps: int = 50,
        eta: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Generate detector hits for a batch of particles.

        Args:
            particle_features: (B, 6)
            n_steps: DDIM sampling steps (50 is fast, 1000 is full DDPM)
            eta: DDIM stochasticity

        Returns:
            dict with generated tracker_hits, calo_hits, counts
        """
        B = particle_features.shape[0]
        device = particle_features.device

        emb = self.encoder(particle_features)

        # Sample hit counts
        n_trk, n_cal = self.hit_count.sample(emb)

        # Generate tracker hits
        max_trk = n_trk.max().item() if n_trk.max() > 0 else 1
        trk_mask = torch.arange(max_trk, device=device).unsqueeze(0) < n_trk.unsqueeze(1)
        trk_hits = self.schedule.sample_loop(
            self.tracker_denoiser,
            (B, max_trk, 6),
            emb, trk_mask,
            n_steps=n_steps, eta=eta,
        )

        # Generate calo hits
        max_cal = n_cal.max().item() if n_cal.max() > 0 else 1
        cal_mask = torch.arange(max_cal, device=device).unsqueeze(0) < n_cal.unsqueeze(1)
        cal_hits = self.schedule.sample_loop(
            self.calo_denoiser,
            (B, max_cal, 5),
            emb, cal_mask,
            n_steps=n_steps, eta=eta,
        )

        return {
            "tracker_hits": trk_hits,
            "tracker_mask": trk_mask,
            "calo_hits": cal_hits,
            "calo_mask": cal_mask,
            "n_tracker": n_trk,
            "n_calo": n_cal,
        }
