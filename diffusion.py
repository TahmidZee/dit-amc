import math
from typing import Optional

import torch


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


def _extract(values: torch.Tensor, t: torch.Tensor, shape) -> torch.Tensor:
    out = values.gather(0, t)
    while len(out.shape) < len(shape):
        out = out.unsqueeze(-1)
    return out


class DiffusionSchedule:
    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: Optional[torch.device] = None,
    ) -> None:
        self.timesteps = timesteps
        betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        if device is not None:
            betas = betas.to(device)
            alpha_bars = alpha_bars.to(device)
        self.betas = betas
        self.alpha_bars = alpha_bars
        self.alpha_bars_rev = torch.flip(alpha_bars, dims=[0])

    def to(self, device: torch.device) -> "DiffusionSchedule":
        self.betas = self.betas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.alpha_bars_rev = self.alpha_bars_rev.to(device)
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = _extract(self.alpha_bars, t, x0.shape)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise

    def snr_to_t(self, snr_db: torch.Tensor) -> torch.Tensor:
        snr_lin = torch.pow(10.0, snr_db / 10.0)
        sigma2 = 1.0 / (snr_lin + 1e-8)
        target = 1.0 / (1.0 + sigma2)
        t_rev = torch.bucketize(target, self.alpha_bars_rev, right=True)
        t = (self.timesteps - 1) - t_rev
        return torch.clamp(t, 0, self.timesteps - 1)
