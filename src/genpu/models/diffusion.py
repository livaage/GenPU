"""Diffusion utilities: noise schedules, training losses, sampling.

Implements DDPM with cosine schedule, plus DDIM for fast sampling.
"""

import math

import torch
import torch.nn as nn


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule from 'Improved DDPM' (Nichol & Dhariwal 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)


class DiffusionSchedule(nn.Module):
    """Precomputed diffusion schedule parameters."""

    def __init__(self, timesteps: int = 1000):
        super().__init__()
        self.timesteps = timesteps

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # Posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward diffusion: add noise to x_0 at timestep t.

        Args:
            x_0: (B, N, D) clean data
            t: (B,) timestep indices
            noise: (B, N, D) Gaussian noise

        Returns:
            x_t: (B, N, D) noised data
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def predict_x0_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Recover x_0 from x_t and predicted noise."""
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t][:, None, None]
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t][:, None, None]
        return sqrt_recip * x_t - sqrt_recipm1 * noise

    def q_posterior_mean(self, x_0: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Posterior mean of q(x_{t-1} | x_t, x_0)."""
        coef1 = self.posterior_mean_coef1[t][:, None, None]
        coef2 = self.posterior_mean_coef2[t][:, None, None]
        return coef1 * x_0 + coef2 * x_t

    @torch.no_grad()
    def p_sample(self, model, x_t, t, cond, mask):
        """One DDPM reverse step: sample x_{t-1} from p(x_{t-1} | x_t)."""
        noise_pred = model(x_t, t, cond, mask)
        x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)
        mean = self.q_posterior_mean(x_0_pred, x_t, t)

        # No noise at t=0
        noise = torch.randn_like(x_t)
        nonzero_mask = (t > 0).float()[:, None, None]
        log_var = self.posterior_log_variance_clipped[t][:, None, None]
        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def ddim_sample(self, model, x_t, t, t_prev, cond, mask, eta=0.0):
        """One DDIM reverse step (deterministic when eta=0)."""
        noise_pred = model(x_t, t, cond, mask)
        x_0_pred = self.predict_x0_from_noise(x_t, t, noise_pred)

        alpha_t = self.alphas_cumprod[t][:, None, None]
        alpha_prev = self.alphas_cumprod[t_prev][:, None, None]

        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        dir_xt = torch.sqrt(1 - alpha_prev - sigma**2) * noise_pred
        noise = torch.randn_like(x_t) if eta > 0 else 0
        return torch.sqrt(alpha_prev) * x_0_pred + dir_xt + sigma * noise

    @torch.no_grad()
    def sample_loop(self, model, shape, cond, mask, n_steps=None, eta=0.0):
        """Full sampling loop (DDPM if n_steps=None, DDIM otherwise).

        Args:
            model: denoiser network
            shape: (B, N, D) output shape
            cond: (B, embed_dim) particle conditioning
            mask: (B, N) bool mask for valid positions
            n_steps: if set, use DDIM with this many steps
            eta: DDIM stochasticity (0=deterministic)

        Returns:
            x_0: (B, N, D) generated samples
        """
        device = cond.device
        x = torch.randn(shape, device=device)

        if n_steps is None or n_steps >= self.timesteps:
            # Full DDPM
            for t_val in reversed(range(self.timesteps)):
                t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
                x = self.p_sample(model, x, t, cond, mask)
        else:
            # DDIM with fewer steps
            step_size = self.timesteps // n_steps
            timesteps = list(range(0, self.timesteps, step_size))
            for i in reversed(range(1, len(timesteps))):
                t = torch.full((shape[0],), timesteps[i], device=device, dtype=torch.long)
                t_prev = torch.full((shape[0],), timesteps[i - 1], device=device, dtype=torch.long)
                x = self.ddim_sample(model, x, t, t_prev, cond, mask, eta=eta)

        # Zero out masked positions
        x = x * mask.unsqueeze(-1).float()
        return x
