import math
from typing import Callable, Optional

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
        self.timesteps = int(timesteps)
        betas = linear_beta_schedule(self.timesteps, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1, dtype=alpha_bars.dtype), alpha_bars[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.alpha_bars_prev = alpha_bars_prev
        self.alpha_bars_rev = torch.flip(alpha_bars, dims=[0])
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(torch.clamp(1.0 - alpha_bars, min=0.0))
        self.sqrt_recip_alpha_bars = torch.sqrt(torch.clamp(1.0 / alpha_bars, min=1e-12))
        self.sqrt_recipm1_alpha_bars = torch.sqrt(torch.clamp(1.0 / alpha_bars - 1.0, min=0.0))
        self.posterior_variance = (
            betas * torch.clamp(1.0 - alpha_bars_prev, min=0.0) / torch.clamp(1.0 - alpha_bars, min=1e-12)
        )
        self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))

        if device is not None:
            self.to(device)

    def to(self, device: torch.device) -> "DiffusionSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.alpha_bars_prev = self.alpha_bars_prev.to(device)
        self.alpha_bars_rev = self.alpha_bars_rev.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        self.sqrt_recip_alpha_bars = self.sqrt_recip_alpha_bars.to(device)
        self.sqrt_recipm1_alpha_bars = self.sqrt_recipm1_alpha_bars.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = _extract(self.alpha_bars, t, x0.shape)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(torch.clamp(1.0 - alpha_bar, min=0.0)) * noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        sqrt_recip_ab = _extract(self.sqrt_recip_alpha_bars, t, x_t.shape)
        sqrt_recipm1_ab = _extract(self.sqrt_recipm1_alpha_bars, t, x_t.shape)
        return sqrt_recip_ab * x_t - sqrt_recipm1_ab * eps

    def predict_x0_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        sqrt_ab = _extract(self.sqrt_alpha_bars, t, x_t.shape)
        sqrt_1mab = _extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)
        return sqrt_ab * x_t - sqrt_1mab * v

    def predict_eps_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        sqrt_ab = _extract(self.sqrt_alpha_bars, t, x_t.shape)
        sqrt_1mab = _extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)
        return sqrt_1mab * x_t + sqrt_ab * v

    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        model_pred: torch.Tensor,
        target: str = "v",
        eta: float = 0.0,
        clip_x0: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tgt = str(target).strip().lower()
        if tgt not in {"v", "eps"}:
            raise ValueError("target must be one of: v | eps.")

        if tgt == "v":
            x0_pred = self.predict_x0_from_v(x_t, t, model_pred)
            eps_pred = self.predict_eps_from_v(x_t, t, model_pred)
        else:
            eps_pred = model_pred
            x0_pred = self.predict_x0_from_eps(x_t, t, eps_pred)

        if clip_x0:
            x0_pred = torch.clamp(x0_pred, min=-1.0, max=1.0)

        alpha_bar_t = _extract(self.alpha_bars, t, x_t.shape)
        alpha_bar_prev = _extract(self.alpha_bars, t_prev, x_t.shape)
        eta_f = float(max(0.0, eta))
        sigma = eta_f * torch.sqrt(
            torch.clamp((1.0 - alpha_bar_prev) / torch.clamp(1.0 - alpha_bar_t, min=1e-12), min=0.0)
            * torch.clamp(1.0 - alpha_bar_t / torch.clamp(alpha_bar_prev, min=1e-12), min=0.0)
        )
        noise = torch.randn_like(x_t) if eta_f > 0.0 else torch.zeros_like(x_t)
        dir_term = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma * sigma, min=0.0)) * eps_pred
        x_prev = torch.sqrt(torch.clamp(alpha_bar_prev, min=0.0)) * x0_pred + dir_term + sigma * noise

        same_step = (t_prev == t).view(-1, *([1] * (x_t.ndim - 1)))
        if bool(torch.any(same_step)):
            x_prev = torch.where(same_step, x_t, x_prev)
        return x_prev, x0_pred

    def ddim_sample_loop(
        self,
        denoise_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_start: torch.Tensor,
        t_start: torch.Tensor,
        steps: int = 8,
        target: str = "v",
        eta: float = 0.0,
        clip_x0: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_steps = max(1, int(steps))
        t0 = torch.clamp(t_start.long(), min=0, max=self.timesteps - 1)
        x_t = x_start
        t_cur = t0
        last_x0 = x_start

        for step_idx in range(n_steps):
            frac_next = float(step_idx + 1) / float(n_steps)
            t_next = torch.floor((1.0 - frac_next) * t0.float()).long()
            pred = denoise_fn(x_t, t_cur)
            x_t, last_x0 = self.ddim_step(
                x_t=x_t,
                t=t_cur,
                t_prev=t_next,
                model_pred=pred,
                target=target,
                eta=eta,
                clip_x0=clip_x0,
            )
            t_cur = t_next

        return x_t, last_x0

    def snr_to_t(self, snr_db: torch.Tensor) -> torch.Tensor:
        snr_lin = torch.pow(10.0, snr_db / 10.0)
        sigma2 = 1.0 / (snr_lin + 1e-8)
        target = 1.0 / (1.0 + sigma2)
        t_rev = torch.bucketize(target, self.alpha_bars_rev, right=True)
        t = (self.timesteps - 1) - t_rev
        return torch.clamp(t, 0, self.timesteps - 1)
