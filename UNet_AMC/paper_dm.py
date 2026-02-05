"""
Paper-style one-step diffusion denoiser (SDE formulation) for RF signals.

Based on:
  "Erasing Noise in Signal Detection with Diffusion Model: From Theory to Application"
  Xiucheng Wang, Peilin Zheng, Nan Cheng

Key equations used:
  - Remark 2 / Eq.(21): SNR (via noise power σ^2) ↔ timestep t
      t = (2σ^2 + 1 - sqrt(1 + 4σ^2)) / (4σ^2)

  - Theorem 2 / Eq.(22): scaling α to align OOD received signal r to diffusion variable x_t
      α = sqrt(((1-t)^2 E||Hs||^2 + t) / (E||Hs||^2 + σ^2))

  - Eq.(23)-(24): one-step denoise
      (ĥ_t, ε̂_t) = ε_θ(α r, t)
      x̂0 ≈ α r - t ĥ_t - ε̂_t

For AMC we treat H = I and x0 as a clean-ish signal sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def snr_db_to_linear(snr_db: torch.Tensor) -> torch.Tensor:
    return torch.pow(10.0, snr_db.float() / 10.0)


def sigma2_from_snr_db(
    snr_db: torch.Tensor,
    signal_power: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """
    Convert SNR(dB) to noise power σ^2 assuming SNR = P_signal / σ^2.
    """
    snr_lin = snr_db_to_linear(snr_db)
    if not torch.is_tensor(signal_power):
        signal_power = torch.tensor(float(signal_power), device=snr_lin.device)
    return signal_power / torch.clamp(snr_lin, min=1e-12)


def t_from_sigma2_paper(sigma2: torch.Tensor) -> torch.Tensor:
    """
    Eq.(21) in the paper (Remark 2):

      t = (2σ^2 + 1 - sqrt(1 + 4σ^2)) / (4σ^2)

    Returns t in (0, 1]. Numerically stable for small σ^2.
    """
    sigma2 = torch.clamp(sigma2.float(), min=1e-12)
    num = 2.0 * sigma2 + 1.0 - torch.sqrt(1.0 + 4.0 * sigma2)
    den = 4.0 * sigma2
    t = num / den
    return torch.clamp(t, min=1e-6, max=1.0)


def alpha_from_sigma2_t_paper(
    sigma2: torch.Tensor,
    t: torch.Tensor,
    signal_power: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """
    Eq.(22) in the paper (Theorem 2):

      α = sqrt(((1-t)^2 E||Hs||^2 + t) / (E||Hs||^2 + σ^2))

    For AMC we use signal_power ≈ E||x0||^2 (after normalization usually ~1).
    """
    if not torch.is_tensor(signal_power):
        signal_power = torch.tensor(float(signal_power), device=sigma2.device)
    sigma2 = sigma2.float()
    t = t.float()

    num = (1.0 - t) ** 2 * signal_power + t
    den = signal_power + sigma2
    alpha = torch.sqrt(torch.clamp(num / torch.clamp(den, min=1e-12), min=1e-12))
    return alpha


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) float tensor (we use continuous t in (0,1])
        """
        device = t.device
        half_dim = self.dim // 2
        if half_dim <= 0:
            raise ValueError("time embedding dim too small")
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvBlock(nn.Module):
    """
    Residual Conv block with time conditioning:
      Conv -> BN -> SiLU -> (add time) -> Conv -> BN -> SiLU + residual
    """

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        h = h + self.time_mlp(t_emb)[:, :, None]
        h = self.act(self.bn2(self.conv2(h)))
        return h + self.residual(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch, time_dim)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # skip is taken AFTER block (standard U-Net) – channels match out_ch
        x = self.block(x, t_emb)
        skip = x
        x = self.pool(x)
        return x, skip


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 2, stride=2)
        self.block = ConvBlock(out_ch + skip_ch, out_ch, time_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            x = F.pad(x, (diff // 2, diff - diff // 2))
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t_emb)


class PaperOneStepDenoiserNet(nn.Module):
    """
    U-Net that predicts (h_t, eps_t_scaled) from (x_t, t).

    Output channels = 2*C:
      - first C channels: h_pred
      - next  C channels: eps_pred (this represents the scaled noise term, consistent with Eq.(24))
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 64, depth: int = 4, time_dim: int = 256):
        super().__init__()
        self.in_channels = in_channels

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.inc = ConvBlock(in_channels, base_channels, time_dim)

        self.downs = nn.ModuleList()
        ch = base_channels
        skips = []
        for _ in range(depth):
            self.downs.append(Down(ch, ch * 2, time_dim))
            skips.append(ch * 2)  # skip channels after block
            ch *= 2

        self.mid = ConvBlock(ch, ch, time_dim)

        self.ups = nn.ModuleList()
        for skip_ch in reversed(skips):
            out_ch = ch // 2
            self.ups.append(Up(ch, skip_ch, out_ch, time_dim))
            ch = out_ch

        self.out = nn.Conv1d(base_channels, 2 * in_channels, 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x_t: (B, C, L)
          t:   (B,) continuous in (0,1]
        """
        t_emb = self.time_mlp(t.float())

        x = self.inc(x_t, t_emb)
        skips = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)

        x = self.mid(x, t_emb)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, t_emb)

        out = self.out(x)
        h_pred, eps_pred = out[:, : self.in_channels, :], out[:, self.in_channels :, :]
        return h_pred, eps_pred


@dataclass
class PaperDenoiseParams:
    """
    Convenience container for per-sample denoising params.
    """

    sigma2: torch.Tensor  # (B,)
    t: torch.Tensor       # (B,)
    alpha: torch.Tensor   # (B,)


class PaperOneStepDenoiser(nn.Module):
    """
    Wraps PaperOneStepDenoiserNet with the paper's SNR->t and scaling α.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        time_dim: int = 256,
        signal_power: float = 1.0,
    ):
        super().__init__()
        self.net = PaperOneStepDenoiserNet(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            time_dim=time_dim,
        )
        self.signal_power = float(signal_power)

    def compute_params(self, snr_db: torch.Tensor) -> PaperDenoiseParams:
        sigma2 = sigma2_from_snr_db(snr_db, signal_power=self.signal_power)
        t = t_from_sigma2_paper(sigma2)
        alpha = alpha_from_sigma2_t_paper(sigma2, t, signal_power=self.signal_power)
        return PaperDenoiseParams(sigma2=sigma2, t=t, alpha=alpha)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.net(x_t, t)

    @torch.no_grad()
    def denoise_observation(self, r: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        """
        Apply paper-style one-step denoising to an observed received signal r.

        Args:
          r: (B, C, L)
          snr_db: (B,) SNR in dB
        Returns:
          x0_hat: (B, C, L)
        """
        params = self.compute_params(snr_db)
        x_t = params.alpha[:, None, None] * r
        h_pred, eps_pred = self.net(x_t, params.t)
        x0_hat = x_t - params.t[:, None, None] * h_pred - eps_pred
        return x0_hat


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

