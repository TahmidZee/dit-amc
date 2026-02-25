import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion import DiffusionSchedule


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, device=timesteps.device).float() / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def snr_db_to_eta(
    snr_db: torch.Tensor,
    rho_min: float = 1e-4,
    rho_max: float = 1.0 - 1e-4,
    eta_min: float = -8.0,
    eta_max: float = 5.5,
) -> torch.Tensor:
    """
    Convert SNR in dB to eta = logit(rho), where rho=noise_fraction=P_noise/P_total.
    """
    snr_lin = torch.pow(10.0, snr_db.float() / 10.0)
    rho = 1.0 / (1.0 + snr_lin)
    rho = torch.clamp(rho, min=float(rho_min), max=float(rho_max))
    eta = torch.log(rho) - torch.log1p(-rho)
    eta = torch.clamp(eta, min=float(eta_min), max=float(eta_max))
    return eta


def _inv_softplus(x: float) -> float:
    x = float(max(x, 1e-6))
    return math.log(math.expm1(x))


class AnalyticNoiseProxy(nn.Module):
    """
    Scale-invariant analytic proxy for noise fraction.

    e_proxy = E[|x[n]-x[n-1]|^2] / (E[|x[n]|^2] + eps)
    rho0 = sigmoid(a * e_proxy + b), with a>=0 to keep monotonicity.
    """

    def __init__(
        self,
        rho_min: float = 1e-4,
        rho_max: float = 1.0 - 1e-4,
        init_scale: float = 1.0,
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if float(rho_min) <= 0.0 or float(rho_max) >= 1.0:
            raise ValueError("AnalyticNoiseProxy requires 0 < rho_min < rho_max < 1.")
        if float(rho_min) >= float(rho_max):
            raise ValueError("AnalyticNoiseProxy requires rho_min < rho_max.")
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.scale_raw = nn.Parameter(torch.tensor(_inv_softplus(float(init_scale)), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(float(init_bias), dtype=torch.float32))

    @staticmethod
    def compute_proxy_energy(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        x_f = x.float()
        d = x_f[:, :, 1:] - x_f[:, :, :-1]
        d_energy = (d * d).sum(dim=1).mean(dim=1)
        p_total = (x_f * x_f).sum(dim=1).mean(dim=1)
        return d_energy / (p_total + float(eps))

    def set_calibration(self, scale: float, bias: float) -> None:
        with torch.no_grad():
            self.scale_raw.copy_(
                torch.tensor(_inv_softplus(float(max(scale, 1e-6))), dtype=self.scale_raw.dtype)
            )
            self.bias.copy_(torch.tensor(float(bias), dtype=self.bias.dtype))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e_proxy = self.compute_proxy_energy(x)
        scale = F.softplus(self.scale_raw) + 1e-6
        eta0 = scale * e_proxy + self.bias
        rho0 = torch.sigmoid(eta0)
        rho0 = torch.clamp(rho0, min=self.rho_min, max=self.rho_max)
        eta0 = torch.log(rho0) - torch.log1p(-rho0)
        return rho0, eta0, e_proxy


class NoiseFractionNet(nn.Module):
    """
    Hybrid eta predictor:
      eta_hat = clamp(logit(rho0) + delta_eta_nn, eta_min, eta_max)
    where rho0 is from AnalyticNoiseProxy.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 32,
        rho_min: float = 1e-4,
        rho_max: float = 1.0 - 1e-4,
        eta_min: float = -8.0,
        eta_max: float = 5.5,
    ) -> None:
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError("NoiseFractionNet requires in_channels > 0.")
        if float(eta_min) >= float(eta_max):
            raise ValueError("NoiseFractionNet requires eta_min < eta_max.")
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.proxy = AnalyticNoiseProxy(rho_min=rho_min, rho_max=rho_max)
        h = int(hidden_channels)
        self.feature = nn.Sequential(
            nn.Conv1d(int(in_channels), h, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(h, h, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(h + 1, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )
        self._last_aux: Optional[dict] = None

    def set_proxy_calibration(self, scale: float, bias: float) -> None:
        self.proxy.set_calibration(scale=scale, bias=bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        rho0, eta0, e_proxy = self.proxy(x)
        h = self.feature(x.float()).mean(dim=-1)
        h = torch.cat([h, e_proxy.unsqueeze(-1)], dim=1)
        delta_eta = self.delta_head(h).squeeze(-1)
        eta_hat = torch.clamp(eta0 + delta_eta, min=self.eta_min, max=self.eta_max)
        aux = {
            "rho0": rho0,
            "eta0": eta0,
            "delta_eta": delta_eta,
            "e_proxy": e_proxy,
        }
        self._last_aux = aux
        return eta_hat, aux


def _group_norm_groups(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class FiLMResBlock1D(nn.Module):
    """Conditional residual block used in the 1D denoiser."""

    def __init__(self, channels: int, cond_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_norm_groups(int(channels)), int(channels))
        self.norm2 = nn.GroupNorm(_group_norm_groups(int(channels)), int(channels))
        self.conv1 = nn.Conv1d(int(channels), int(channels), kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(int(channels), int(channels), kernel_size=3, padding=1)
        self.cond_proj = nn.Linear(int(cond_dim), 2 * int(channels))
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.cond_proj(cond).chunk(2, dim=1)
        h = self.norm1(x)
        h = h * (1.0 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.conv2(h)
        return x + h


class DenoiserUNet1D(nn.Module):
    """Lightweight conditional residual U-Net denoiser for IQ inputs."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        cond_dim: int = 32,
        dropout: float = 0.0,
        soft_high_snr_blend: bool = True,
        eta_hi: float = -math.log(10.0),
        tau_eta: float = 0.46,
    ) -> None:
        super().__init__()
        c = int(base_channels)
        self.soft_high_snr_blend = bool(soft_high_snr_blend)
        self.eta_hi = float(eta_hi)
        self.tau_eta = float(tau_eta)

        self.eta_embed = nn.Sequential(
            nn.Linear(1, int(cond_dim)),
            nn.GELU(),
            nn.Linear(int(cond_dim), int(cond_dim)),
            nn.GELU(),
        )

        self.in_proj = nn.Conv1d(int(in_channels), c, kernel_size=3, padding=1)
        self.enc1 = FiLMResBlock1D(c, int(cond_dim), dropout=dropout)
        self.down1 = nn.Conv1d(c, 2 * c, kernel_size=4, stride=2, padding=1)
        self.enc2 = FiLMResBlock1D(2 * c, int(cond_dim), dropout=dropout)
        self.down2 = nn.Conv1d(2 * c, 4 * c, kernel_size=4, stride=2, padding=1)
        self.mid = FiLMResBlock1D(4 * c, int(cond_dim), dropout=dropout)

        self.up2 = nn.Conv1d(4 * c, 2 * c, kernel_size=3, padding=1)
        self.dec2 = FiLMResBlock1D(4 * c, int(cond_dim), dropout=dropout)
        self.up1 = nn.Conv1d(4 * c, c, kernel_size=3, padding=1)
        self.dec1 = FiLMResBlock1D(2 * c, int(cond_dim), dropout=dropout)
        self.out = nn.Conv1d(2 * c, int(in_channels), kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _residual_scale(self, eta: torch.Tensor) -> torch.Tensor:
        if not self.soft_high_snr_blend:
            return torch.ones_like(eta)
        tau = max(1e-6, self.tau_eta)
        m_hi = torch.sigmoid((self.eta_hi - eta) / tau)
        return 1.0 - m_hi

    def forward(self, x: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        if eta.ndim == 2 and eta.shape[1] == 1:
            eta = eta.squeeze(1)
        cond = self.eta_embed(eta.float().view(-1, 1))

        x_f = x.float()
        e1 = self.enc1(self.in_proj(x_f), cond)
        e2 = self.enc2(self.down1(e1), cond)
        m = self.mid(self.down2(e2), cond)

        u2 = F.interpolate(m, size=e2.shape[-1], mode="linear", align_corners=False)
        u2 = self.up2(u2)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2(u2, cond)

        u1 = F.interpolate(u2, size=e1.shape[-1], mode="linear", align_corners=False)
        u1 = self.up1(u1)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1(u1, cond)

        delta = self.out(u1).to(dtype=x.dtype)
        scale = self._residual_scale(eta.float()).to(dtype=delta.dtype).view(-1, 1, 1)
        return x + scale * delta


class WaveformDiffusionDenoiser1D(nn.Module):
    """Conditional waveform denoiser that predicts eps/v for diffusion training."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 48,
        cond_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c = int(base_channels)
        cd = int(cond_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(cd, cd),
            nn.GELU(),
            nn.Linear(cd, cd),
            nn.GELU(),
        )
        self.eta_mlp = nn.Sequential(
            nn.Linear(1, cd),
            nn.GELU(),
            nn.Linear(cd, cd),
            nn.GELU(),
        )
        self.cond_fuse = nn.Sequential(
            nn.Linear(2 * cd, cd),
            nn.GELU(),
        )
        self.cond_proj = nn.Sequential(
            nn.Conv1d(int(in_channels), c, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(c, c, kernel_size=3, padding=1),
        )
        self.in_proj = nn.Conv1d(int(in_channels) * 2, c, kernel_size=3, padding=1)
        self.enc1 = FiLMResBlock1D(c, cd, dropout=dropout)
        self.down1 = nn.Conv1d(c, 2 * c, kernel_size=4, stride=2, padding=1)
        self.enc2 = FiLMResBlock1D(2 * c, cd, dropout=dropout)
        self.down2 = nn.Conv1d(2 * c, 4 * c, kernel_size=4, stride=2, padding=1)
        self.mid = FiLMResBlock1D(4 * c, cd, dropout=dropout)
        self.up2 = nn.Conv1d(4 * c, 2 * c, kernel_size=3, padding=1)
        self.dec2 = FiLMResBlock1D(4 * c, cd, dropout=dropout)
        self.up1 = nn.Conv1d(4 * c, c, kernel_size=3, padding=1)
        self.dec1 = FiLMResBlock1D(2 * c, cd, dropout=dropout)
        self.out = nn.Conv1d(2 * c, int(in_channels), kernel_size=3, padding=1)

    def forward(
        self,
        x_t: torch.Tensor,
        x_cond: torch.Tensor,
        t: torch.Tensor,
        eta_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if eta_cond is None:
            eta_cond = torch.zeros((x_t.shape[0],), device=x_t.device, dtype=x_t.dtype)
        if eta_cond.ndim == 2 and eta_cond.shape[1] == 1:
            eta_cond = eta_cond.squeeze(1)
        t_emb = timestep_embedding(t.long(), self.time_mlp[0].in_features)
        t_cond = self.time_mlp(t_emb.to(dtype=x_t.dtype))
        e_cond = self.eta_mlp(eta_cond.float().view(-1, 1).to(dtype=x_t.dtype))
        cond = self.cond_fuse(torch.cat([t_cond, e_cond], dim=1))

        x_in = torch.cat([x_t.float(), x_cond.float()], dim=1)
        c_in = self.cond_proj(x_cond.float())
        e1 = self.enc1(self.in_proj(x_in) + c_in, cond)
        e2 = self.enc2(self.down1(e1), cond)
        m = self.mid(self.down2(e2), cond)
        u2 = F.interpolate(m, size=e2.shape[-1], mode="linear", align_corners=False)
        u2 = self.up2(u2)
        u2 = torch.cat([u2, e2], dim=1)
        u2 = self.dec2(u2, cond)
        u1 = F.interpolate(u2, size=e1.shape[-1], mode="linear", align_corners=False)
        u1 = self.up1(u1)
        u1 = torch.cat([u1, e1], dim=1)
        u1 = self.dec1(u1, cond)
        return self.out(u1).to(dtype=x_t.dtype)


class ResBlock1d(nn.Module):
    """Residual block for 1D signals with optional channel projection."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x + residual


class DiTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
        self.cond1 = nn.Linear(dim, 2 * dim)
        self.cond2 = nn.Linear(dim, 2 * dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale1, shift1 = self.cond1(cond).chunk(2, dim=-1)
        x_norm = self.norm1(x) * (1.0 + scale1[:, None, :]) + shift1[:, None, :]
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = x + attn_out

        scale2, shift2 = self.cond2(cond).chunk(2, dim=-1)
        x_norm = self.norm2(x) * (1.0 + scale2[:, None, :]) + shift2[:, None, :]
        x = x + self.mlp(x_norm)
        return x


class DiffusionAMC(nn.Module):
    def __init__(
        self,
        num_classes: int = 11,
        in_channels: int = 2,
        seq_len: int = 128,
        patch_size: int = 8,
        dim: int = 192,
        depth: int = 10,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        snr_scale: float = 20.0,
        stem_channels: int = 0,
        stem_layers: int = 2,
        group_pool: str = "mean",
    ) -> None:
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError("seq_len must be divisible by patch_size.")
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.dim = dim
        self.snr_scale = snr_scale
        if group_pool not in {"mean", "attn"}:
            raise ValueError(f"group_pool must be mean|attn, got {group_pool}")
        self.group_pool = group_pool

        if stem_channels and stem_channels > 0:
            # Deeper residual stem: project to stem_channels, then stack ResBlocks.
            layers = [
                nn.Conv1d(in_channels, stem_channels, kernel_size=7, padding=3),
                nn.BatchNorm1d(stem_channels),
                nn.GELU(),
            ]
            for _ in range(int(stem_layers)):
                layers.append(ResBlock1d(stem_channels, kernel_size=3, dropout=dropout * 0.5))
            self.stem = nn.Sequential(*layers)
            patch_in = stem_channels
        else:
            self.stem = None
            patch_in = in_channels

        self.patch_embed = nn.Conv1d(
            in_channels=patch_in,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, dim))

        self.t_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.snr_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.cls_head = nn.Linear(dim, num_classes)

        # Diffusion-style denoising head: predict clean patch embeddings z0 from noisy zt.
        self.x0_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )
        self.snr_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1),
        )
        # Learned pooling across group-k windows (attention over per-window representations).
        self.group_attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, max(32, dim // 2)),
            nn.GELU(),
            nn.Linear(max(32, dim // 2), 1),
        )

    def forward_tokens(
        self,
        tokens_in: torch.Tensor,
        t: torch.Tensor,
        snr: Optional[torch.Tensor] = None,
        snr_mode: str = "predict",
        group_size: Optional[int] = None,
        group_mask: Optional[torch.Tensor] = None,
    ):
        """
        tokens_in: (B, N, D) patch embeddings (without positional embedding).
        Returns:
          logits: (B, C)
          x0_pred: (B, N, D) predicted clean patch embeddings
          snr_pred: (B,)
        """
        if snr_mode not in {"predict", "known", "none"}:
            raise ValueError(f"snr_mode must be one of predict|known|none, got {snr_mode}")
        tokens = tokens_in + self.pos_embed

        pooled0 = tokens.mean(dim=1)
        snr_pred = self.snr_head(pooled0).squeeze(-1)  # (B*,)

        if snr_mode == "known" and snr is not None:
            snr_used = snr
        elif snr_mode == "predict":
            snr_used = snr_pred
        else:
            snr_used = torch.zeros_like(snr_pred)

        if snr_used.ndim == 2 and snr_used.shape[-1] == 1:
            snr_used = snr_used.squeeze(-1)
        snr_used = snr_used.clamp(min=-self.snr_scale, max=self.snr_scale)
        snr_used = snr_used.unsqueeze(-1) / self.snr_scale
        cond = self.t_mlp(timestep_embedding(t, self.dim)) + self.snr_mlp(snr_used)

        for block in self.blocks:
            tokens = block(tokens, cond)

        tokens = self.norm(tokens)
        x0_pred = self.x0_head(tokens)

        pooled = x0_pred.mean(dim=1)  # (B*, D)
        if group_size is not None and group_size > 1:
            if pooled.shape[0] % group_size != 0:
                raise ValueError("Batch size must be divisible by group_size.")
            b = pooled.shape[0] // group_size
            pooled_g = pooled.view(b, group_size, -1)  # (B,G,D)
            snr_g = snr_pred.view(b, group_size)  # (B,G)
            if group_mask is None:
                mask = torch.ones((b, group_size), device=pooled_g.device, dtype=torch.float32)
            else:
                if group_mask.ndim != 2 or group_mask.shape[0] != b or group_mask.shape[1] != group_size:
                    raise ValueError("group_mask must have shape (B, group_size)")
                mask = group_mask.to(device=pooled_g.device, dtype=torch.float32)

            if self.group_pool == "mean":
                denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)  # (B,1)
                pooled = torch.sum(pooled_g * mask.unsqueeze(-1), dim=1) / denom
                snr_pred_out = torch.sum(snr_g * mask, dim=1) / denom.squeeze(-1)
            else:
                scores = self.group_attn(pooled_g).squeeze(-1)  # (B,G)
                scores = scores.masked_fill(mask <= 0, float("-inf"))
                w = torch.softmax(scores, dim=1)  # (B,G)
                # If all masked (shouldn't happen), softmax -> NaNs; guard.
                w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
                pooled = torch.sum(w.unsqueeze(-1) * pooled_g, dim=1)
                snr_pred_out = torch.sum(w * snr_g, dim=1)
        else:
            snr_pred_out = snr_pred
        logits = self.cls_head(pooled)
        return logits, x0_pred, snr_pred_out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 2, L)
        returns tokens0: (B, N, D) patch embeddings before positional embedding.
        """
        if self.stem is not None:
            x = self.stem(x)
        return self.patch_embed(x).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        snr: Optional[torch.Tensor] = None,
        snr_mode: str = "predict",
        group_mask: Optional[torch.Tensor] = None,
    ):
        # Accept (B,2,L) or (B,K,2,L)
        group_size = None
        if x.ndim == 4:
            group_size = x.shape[1]
            x = x.reshape(-1, x.shape[2], x.shape[3])
            # If t (and/or snr) is provided per-group, expand to per-window.
            if t.ndim != 1:
                raise ValueError("t must be 1D (batch,) tensor.")
            if t.shape[0] * group_size == x.shape[0]:
                t = t.repeat_interleave(group_size)
            if snr is not None and snr_mode == "known":
                if snr.ndim != 1:
                    raise ValueError("snr must be 1D (batch,) tensor when provided.")
                if snr.shape[0] * group_size == x.shape[0]:
                    snr = snr.repeat_interleave(group_size)
        tokens0 = self.encode(x)
        return self.forward_tokens(
            tokens0,
            t,
            snr=snr,
            snr_mode=snr_mode,
            group_size=group_size,
            group_mask=group_mask,
        )


class CausalConv1d(nn.Module):
    """Conv1d with left padding to emulate 'causal' padding (Keras-style)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.left_pad = (self.kernel_size - 1) * self.dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class TemporalAttentionPool(nn.Module):
    """Simple attention pooling over time for (B, T, D) sequences."""

    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, hidden)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        scores = self.v(torch.tanh(self.proj(x))).squeeze(-1)  # (B, T)
        w = torch.softmax(scores, dim=1)  # (B, T)
        pooled = torch.sum(w.unsqueeze(-1) * x, dim=1)  # (B, D)
        return pooled, w


class TemporalResidualBlock1D(nn.Module):
    """Residual 1D block with optional channel projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        k = int(kernel_size)
        d = int(dilation)
        if k <= 0 or (k % 2) == 0:
            raise ValueError("TemporalResidualBlock1D requires odd kernel_size > 0.")
        if d <= 0:
            raise ValueError("TemporalResidualBlock1D requires dilation > 0.")
        pad = (k // 2) * d
        self.conv1 = nn.Conv1d(int(in_channels), int(out_channels), kernel_size=k, padding=pad, dilation=d)
        self.conv2 = nn.Conv1d(int(out_channels), int(out_channels), kernel_size=k, padding=pad, dilation=d)
        self.bn1 = nn.BatchNorm1d(int(out_channels))
        self.bn2 = nn.BatchNorm1d(int(out_channels))
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        if int(in_channels) != int(out_channels):
            self.skip = nn.Conv1d(int(in_channels), int(out_channels), kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.act(y)
        return y + residual


class TemporalBackboneTCN(nn.Module):
    """Dilated temporal ConvNet that maps (B,C,T) to (B,T,D)."""

    def __init__(
        self,
        in_channels: int,
        channels: int = 128,
        levels: int = 6,
        kernel_size: int = 3,
        dilation_base: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if int(levels) <= 0:
            raise ValueError("TemporalBackboneTCN requires levels > 0.")
        if int(channels) <= 0:
            raise ValueError("TemporalBackboneTCN requires channels > 0.")
        if int(dilation_base) <= 0:
            raise ValueError("TemporalBackboneTCN requires dilation_base > 0.")
        blocks = []
        c_in = int(in_channels)
        c_out = int(channels)
        base = int(dilation_base)
        for i in range(int(levels)):
            d = int(base ** i)
            blocks.append(
                TemporalResidualBlock1D(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=int(kernel_size),
                    dilation=d,
                    dropout=float(dropout),
                )
            )
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.output_dim = c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(x)
        return y.transpose(1, 2)


class TemporalBackboneResNet1D(nn.Module):
    """Residual 1D stack with cyclical dilation; maps (B,C,T) to (B,T,D)."""

    def __init__(
        self,
        in_channels: int,
        channels: int = 128,
        blocks: int = 8,
        kernel_size: int = 5,
        dilation_cycle: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if int(blocks) <= 0:
            raise ValueError("TemporalBackboneResNet1D requires blocks > 0.")
        if int(channels) <= 0:
            raise ValueError("TemporalBackboneResNet1D requires channels > 0.")
        if int(dilation_cycle) <= 0:
            raise ValueError("TemporalBackboneResNet1D requires dilation_cycle > 0.")
        c_in = int(in_channels)
        c_out = int(channels)
        cyc = int(dilation_cycle)
        layers = []
        for i in range(int(blocks)):
            d = int(2 ** (i % cyc))
            layers.append(
                TemporalResidualBlock1D(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=int(kernel_size),
                    dilation=d,
                    dropout=float(dropout),
                )
            )
            c_in = c_out
        self.blocks = nn.Sequential(*layers)
        self.output_dim = c_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.blocks(x)
        return y.transpose(1, 2)


class MoEClassifierHead(nn.Module):
    """
    Mixture-of-Experts classifier head for CLDNN features.

    - Experts: parallel MLP classifier heads.
    - Router: eta-based (interpretable) or learned MLP.
    - Hard routing: top-1 expert one-hot selection (inference-style sparsity).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        n_experts: int = 2,
        gate_type: str = "eta-sigmoid",
        gate_center: float = 0.5,
        gate_tau: float = 0.3,
        gate_use_feat: bool = False,
    ) -> None:
        super().__init__()
        self.n_experts = int(n_experts)
        self.gate_type = str(gate_type).strip().lower()
        self.gate_center = float(gate_center)
        self.gate_tau = float(gate_tau)
        self.gate_use_feat = bool(gate_use_feat)
        if self.n_experts < 2:
            raise ValueError("MoEClassifierHead requires n_experts >= 2.")
        if self.gate_tau <= 0.0:
            raise ValueError("MoE gate_tau must be > 0.")
        if self.gate_type not in {"eta-sigmoid", "learned", "hard"}:
            raise ValueError("MoE gate_type must be one of: eta-sigmoid|learned|hard.")

        in_dim = int(input_dim)
        hid = int(hidden_dim)
        cls = int(num_classes)
        self.fc1 = nn.ModuleList([nn.Linear(in_dim, hid) for _ in range(self.n_experts)])
        self.fc2 = nn.ModuleList([nn.Linear(hid, hid) for _ in range(self.n_experts)])
        self.fc_out = nn.ModuleList([nn.Linear(hid, cls) for _ in range(self.n_experts)])
        self.act = nn.SELU()
        self.drop = nn.Dropout(float(dropout))

        if self.gate_type == "learned":
            router_in = 1 + (in_dim if self.gate_use_feat else 0)
            router_hidden = max(32, min(256, router_in // 2))
            self.router_mlp = nn.Sequential(
                nn.Linear(router_in, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, self.n_experts),
            )
            self.register_buffer("eta_centers", torch.zeros(1), persistent=False)
        else:
            self.router_mlp = None
            if self.n_experts > 2:
                centers = torch.linspace(-1.0, 1.0, self.n_experts, dtype=torch.float32)
                self.register_buffer("eta_centers", centers, persistent=False)
            else:
                self.register_buffer("eta_centers", torch.zeros(1), persistent=False)

    def _eta_gate_probs(
        self,
        feat: torch.Tensor,
        eta_pred: Optional[torch.Tensor],
        tau_scale: float = 1.0,
    ) -> torch.Tensor:
        b = int(feat.shape[0])
        if eta_pred is None:
            eta = torch.zeros((b,), device=feat.device, dtype=feat.dtype)
        else:
            eta_flat = eta_pred.float().view(-1)
            if eta_flat.numel() != b:
                eta = torch.zeros((b,), device=feat.device, dtype=feat.dtype)
            else:
                eta = eta_flat.to(device=feat.device, dtype=feat.dtype)
        tau_eff = max(1e-6, float(self.gate_tau) * max(1e-6, float(tau_scale)))
        z = (eta - float(self.gate_center)) / tau_eff

        if self.n_experts == 2:
            g_low = torch.sigmoid(z)
            gate = torch.stack([g_low, 1.0 - g_low], dim=1)
            return gate

        # For >2 experts, use soft assignment over evenly-spaced eta anchors.
        logits = -(z.unsqueeze(1) - self.eta_centers.unsqueeze(0).to(z.dtype)) ** 2
        return torch.softmax(logits, dim=1)

    def _learned_gate_probs(self, feat: torch.Tensor, eta_pred: Optional[torch.Tensor]) -> torch.Tensor:
        b = int(feat.shape[0])
        eta_in = torch.zeros((b, 1), device=feat.device, dtype=feat.dtype)
        if eta_pred is not None:
            eta_flat = eta_pred.detach().float().view(-1)
            if eta_flat.numel() == b:
                eta_in = eta_flat.view(-1, 1).to(device=feat.device, dtype=feat.dtype)
        if self.gate_use_feat:
            # Prevent router objectives from perturbing trunk/noise calibration.
            router_in = torch.cat([eta_in, feat.detach()], dim=1)
        else:
            router_in = eta_in
        gate_logits = self.router_mlp(router_in)
        return torch.softmax(gate_logits, dim=1)

    def forward(
        self,
        feat: torch.Tensor,
        eta_pred: Optional[torch.Tensor],
        tau_scale: float = 1.0,
        gate_override: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if gate_override is not None:
            gate = gate_override.to(device=feat.device, dtype=feat.dtype).detach()
        elif self.gate_type == "learned":
            gate = self._learned_gate_probs(feat, eta_pred)
        else:
            gate = self._eta_gate_probs(feat, eta_pred, tau_scale=tau_scale)

        if self.gate_type == "hard":
            top_idx = torch.argmax(gate, dim=1)
            gate = F.one_hot(top_idx, num_classes=self.n_experts).to(dtype=feat.dtype)

        logits_e = self.expert_logits(feat)  # (B, E, C)
        logits = torch.sum(gate.unsqueeze(-1) * logits_e, dim=1)  # (B, C)
        return logits, gate, logits_e

    def expert_logits(self, feat: torch.Tensor) -> torch.Tensor:
        expert_logits = []
        for i in range(self.n_experts):
            z = self.act(self.fc1[i](feat))
            z = self.drop(z)
            z = self.act(self.fc2[i](z))
            z = self.drop(z)
            expert_logits.append(self.fc_out[i](z))
        return torch.stack(expert_logits, dim=1)  # (B, E, C)


class ExpertFeatureExtractor(nn.Module):
    """
    Computes domain-specific features from raw I/Q signals for improved low-SNR classification.

    Features computed (v1):
      - Conjugate products x[n]x*[n-k] for k=1,2,3 (phase-invariant, CFO-robust)
      - Magnitude |x|
      - Phase differences as (sin, cos) for continuity (computed without atan2)

    Features computed (v2):
      - Normalized local short-time correlations r_k[n] / (eps + local_power[n])
      - For each k=1,2,3: Re, Im, |r_k|
      - Stable phase-diff (sin/cos) via dot/cross formulation (no atan2)

    These are processed by a small 1D CNN to produce feature maps that merge
    with the main CLDNN branch before the LSTM.
    """

    EXPERT_INPUT_CHANNELS_V1 = 9   # 2*3 (conj products) + 1 (mag) + 2 (phase diff sin/cos)
    EXPERT_INPUT_CHANNELS_V2 = 11  # 3*3 (Re,Im,|r_k| for k=1,2,3) + 2 (phase diff sin/cos)

    def __init__(
        self,
        out_channels: int = 64,
        time_out: int = 124,
        dropout: float = 0.3,
        stacf_window: int = 0,
        v2_mode: bool = False,
        corr_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.use_v2 = bool(v2_mode)
        self.corr_norm_eps = max(1e-9, float(corr_norm_eps))
        n_in = self.EXPERT_INPUT_CHANNELS_V2 if self.use_v2 else self.EXPERT_INPUT_CHANNELS_V1
        self.conv1 = nn.Conv1d(n_in, out_channels, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # Reduce time from L → L-4 to match main branch after conv_merge(kernel=5, valid)
        self.conv_reduce = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=0)
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.stacf_window = max(0, int(stacf_window))
        # Use odd window for centered smoothing without temporal bias.
        if self.stacf_window > 1 and (self.stacf_window % 2 == 0):
            self.stacf_window += 1

    def _stable_phase_diff(self, I: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stable phase-difference representation without atan2:
          cos(dphi) = dot / (|x_n||x_{n-1}| + eps)
          sin(dphi) = cross / (|x_n||x_{n-1}| + eps)
        """
        I_curr = I[:, 1:]
        Q_curr = Q[:, 1:]
        I_prev = I[:, :-1]
        Q_prev = Q[:, :-1]

        dot = I_curr * I_prev + Q_curr * Q_prev
        cross = Q_curr * I_prev - I_curr * Q_prev
        mag_curr = torch.sqrt(torch.clamp(I_curr * I_curr + Q_curr * Q_curr, min=0.0) + self.corr_norm_eps)
        mag_prev = torch.sqrt(torch.clamp(I_prev * I_prev + Q_prev * Q_prev, min=0.0) + self.corr_norm_eps)
        denom = mag_curr * mag_prev + self.corr_norm_eps

        pd_cos = F.pad(dot / denom, (0, 1))
        pd_sin = F.pad(cross / denom, (0, 1))
        return pd_sin, pd_cos

    def _normalized_local_corr(
        self,
        x_complex: torch.Tensor,
        lag: int,
        local_power: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rk = x_complex[:, lag:] * x_complex[:, :-lag].conj()
        rk = F.pad(rk, (0, lag))
        rk_re = self._smooth_1d(rk.real)
        rk_im = self._smooth_1d(rk.imag)
        denom = local_power + self.corr_norm_eps
        rk_re = rk_re / denom
        rk_im = rk_im / denom
        rk_abs = torch.sqrt(torch.clamp(rk_re * rk_re + rk_im * rk_im, min=0.0) + self.corr_norm_eps)
        return rk_re, rk_im, rk_abs

    def _smooth_1d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Deterministic short-time expectation approximation for low-SNR robustness.

        Args:
            x: (B, L) real-valued sequence.
        Returns:
            (B, L) smoothed sequence (or identity when stacf_window <= 1).
        """
        if self.stacf_window <= 1:
            return x
        w = int(self.stacf_window)
        pad = w // 2
        y = F.avg_pool1d(x.unsqueeze(1), kernel_size=w, stride=1, padding=pad)
        if y.shape[-1] > x.shape[-1]:
            y = y[..., : x.shape[-1]]
        return y.squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, L) raw I/Q signal
        Returns:
            h: (B, out_channels, L-4) expert feature maps (Conv1d kernel=5 valid reduces time by 4)
        """
        I = x[:, 0, :]  # (B, L)
        Q = x[:, 1, :]  # (B, L)

        # Build complex signal
        x_complex = torch.complex(I, Q)  # (B, L)

        pd_sin, pd_cos = self._stable_phase_diff(I, Q)
        pd_sin = self._smooth_1d(pd_sin)
        pd_cos = self._smooth_1d(pd_cos)

        if self.use_v2:
            # Local power normalization approximates short-time expectation and
            # suppresses amplitude-driven variance at low SNR.
            pow2 = I * I + Q * Q
            local_power = self._smooth_1d(pow2)
            r1_re, r1_im, r1_abs = self._normalized_local_corr(x_complex, lag=1, local_power=local_power)
            r2_re, r2_im, r2_abs = self._normalized_local_corr(x_complex, lag=2, local_power=local_power)
            r3_re, r3_im, r3_abs = self._normalized_local_corr(x_complex, lag=3, local_power=local_power)
            feats = torch.stack([
                r1_re, r1_im, r1_abs,
                r2_re, r2_im, r2_abs,
                r3_re, r3_im, r3_abs,
                pd_sin, pd_cos,
            ], dim=1)
        else:
            # Conjugate products: x[n] * conj(x[n-k]) — phase-invariant, CFO-robust
            conj_1 = x_complex[:, 1:] * x_complex[:, :-1].conj()
            conj_2 = x_complex[:, 2:] * x_complex[:, :-2].conj()
            conj_3 = x_complex[:, 3:] * x_complex[:, :-3].conj()
            conj_1 = F.pad(conj_1, (0, 1))
            conj_2 = F.pad(conj_2, (0, 2))
            conj_3 = F.pad(conj_3, (0, 3))
            mag = self._smooth_1d(torch.abs(x_complex))
            conj_1_real = self._smooth_1d(conj_1.real)
            conj_1_imag = self._smooth_1d(conj_1.imag)
            conj_2_real = self._smooth_1d(conj_2.real)
            conj_2_imag = self._smooth_1d(conj_2.imag)
            conj_3_real = self._smooth_1d(conj_3.real)
            conj_3_imag = self._smooth_1d(conj_3.imag)
            feats = torch.stack([
                conj_1_real, conj_1_imag,
                conj_2_real, conj_2_imag,
                conj_3_real, conj_3_imag,
                mag,
                pd_sin, pd_cos,
            ], dim=1)

        # Process with small CNN
        h = self.drop(self.act(self.bn1(self.conv1(feats))))  # (B, out_ch, L)
        h = self.drop(self.act(self.bn2(self.conv2(h))))      # (B, out_ch, L)
        h = self.drop(self.act(self.bn3(self.conv_reduce(h)))) # (B, out_ch, L-4)
        return h


class CyclostationaryStats(nn.Module):
    """
    Computes global cyclostationary statistics from raw I/Q signals.

    These scalar features are robust to noise and complement the CNN features:
      - Short-lag autocorrelation magnitudes |R[k]|
      - Autocorrelation phases angle(R[k])
      - Normalized 4th-order cumulant proxy (kurtosis)

    Lags are adaptive: short lags for any seq_len, long lags only when L is large enough.
    """

    # Base lags for any seq_len (L >= 16)
    MAG_LAGS_SHORT = (1, 2, 4, 8)
    PHASE_LAGS_SHORT = (1, 2)
    # Extended lags when seq_len >= 128
    MAG_LAGS_LONG = (16, 32, 64)
    PHASE_LAGS_LONG = (4, 8)

    def __init__(self, seq_len: int = 128) -> None:
        super().__init__()
        self.mag_lags = list(self.MAG_LAGS_SHORT)
        self.phase_lags = list(self.PHASE_LAGS_SHORT)
        if seq_len >= 256:
            self.mag_lags += [k for k in self.MAG_LAGS_LONG if k < seq_len // 2]
            self.phase_lags += [k for k in self.PHASE_LAGS_LONG if k < seq_len // 2]
        # Phase lags produce 2 features each (cos + sin) for gradient stability
        self.n_stats = len(self.mag_lags) + 2 * len(self.phase_lags) + 1  # +1 for kurtosis

    @property
    def N_STATS(self) -> int:
        return self.n_stats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, L) raw I/Q signal
        Returns:
            stats: (B, N_STATS)
        """
        I = x[:, 0, :]  # (B, L)
        Q = x[:, 1, :]  # (B, L)
        xc = torch.complex(I, Q)  # (B, L)

        stats = []
        # Short-lag autocorrelation R[k] = (1/N) * sum x[n] * conj(x[n-k])
        for k in self.mag_lags:
            rk = (xc[:, k:] * xc[:, :-k].conj()).mean(dim=1)  # (B,) complex
            stats.append(rk.abs())  # |R[k]|

        # Autocorrelation phases (carry modulation info)
        # Use cos/sin decomposition instead of torch.angle for gradient stability:
        # torch.angle has gradient 1/|z|² which explodes as |z|→0 (common at low SNR).
        # Normalized real/imag parts preserve phase info continuously.
        for k in self.phase_lags:
            rk = (xc[:, k:] * xc[:, :-k].conj()).mean(dim=1)
            denom = rk.abs() + 1e-9
            stats.append(rk.real / denom)  # cos(angle(rk))
            stats.append(rk.imag / denom)  # sin(angle(rk))

        # Normalized 4th-order cumulant proxy (kurtosis)
        # C4 = E[|x|^4] / E[|x|^2]^2 - 2
        pow2 = (xc.real ** 2 + xc.imag ** 2)  # |x|^2, (B, L)
        pow4 = pow2 ** 2  # |x|^4, (B, L)
        e2 = pow2.mean(dim=1)  # E[|x|^2], (B,)
        e4 = pow4.mean(dim=1)  # E[|x|^4], (B,)
        kurtosis = e4 / (e2 ** 2 + 1e-8) - 2.0  # (B,)
        stats.append(kurtosis)

        return torch.stack(stats, dim=1)  # (B, N_STATS)


class CLDNNAMC(nn.Module):
    """
    K=1-first AMC model inspired by MCLDNN / CLDNN family (CNN + temporal backbone), implemented in PyTorch.

    Design goals:
      - Strong inductive bias for short IQ sequences (e.g., L=128) and longer windows (e.g., L=1024)
      - Small parameter count (<< DiT-AMC) to reduce overfitting
      - Optional attention pooling over LSTM outputs (better than "last hidden" at low SNR)
      - Optional SNR conditioning via FiLM (Feature-wise Linear Modulation)
      - Optional expert features (conjugate products, cyclostationary stats) for low-SNR

    Forward signature matches DiffusionAMC so we can reuse evaluate()/dynamic-K code.
    """

    def __init__(
        self,
        num_classes: int = 11,
        seq_len: int = 128,
        conv_channels: int = 50,
        merge_channels: int = 100,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        bidirectional: bool = False,
        cldnn_backbone: str = "lstm",  # lstm|tcn|resnet1d
        cldnn_tcn_levels: int = 6,
        cldnn_tcn_channels: int = 128,
        cldnn_tcn_kernel: int = 3,
        cldnn_tcn_dilation_base: int = 2,
        cldnn_tcn_dropout: float = 0.15,
        cldnn_resnet_blocks: int = 8,
        cldnn_resnet_channels: int = 128,
        cldnn_resnet_kernel: int = 5,
        cldnn_resnet_dilation_cycle: int = 4,
        cldnn_resnet_dropout: float = 0.15,
        dropout: float = 0.5,
        pool: str = "attn",  # attn|last|mean
        snr_cond: bool = False,  # Enable SNR conditioning via FiLM
        noise_cond: bool = False,  # Enable noise-fraction conditioning via FiLM (eta=logit(rho))
        snr_embed_dim: int = 64,  # SNR embedding dimension
        snr_loss_detach_backbone: bool = False,  # Head-only SNR supervision (do not backprop SNR loss into backbone)
        snr_min_db: float = -20.0,
        snr_max_db: float = 18.0,
        noise_eta_min: float = -8.0,
        noise_eta_max: float = 5.5,
        denoiser: bool = False,  # Enable residual U-Net denoiser preprocessor
        denoiser_dual_path: bool = False,  # Feed classifier [x_raw, x_dn] (4ch) instead of x_dn only
        denoiser_base_channels: int = 32,
        denoiser_dropout: float = 0.0,
        denoiser_soft_high_snr_blend: bool = True,
        denoiser_eta_hi: float = -math.log(10.0),
        denoiser_tau_eta: float = 0.46,
        dn_diff_enable: bool = False,
        dn_diff_target: str = "v",  # v|eps
        dn_diff_train_timesteps: int = 100,
        dn_diff_eval_mode: str = "ddim",  # ddim|onestep
        dn_diff_eval_steps: int = 8,
        dn_diff_ddim_eta: float = 0.0,
        dn_diff_multisample: int = 1,
        dn_diff_eval_t_start_source: str = "snr_pred",  # snr_pred|snr_true|fixed
        dn_diff_fixed_t_start: int = 30,
        dn_diff_snr2t_scale: float = 1.0,
        dn_diff_snr2t_bias: float = 0.0,
        dn_diff_low_snr_thresh: float = -6.0,
        dn_diff_high_snr_margin: float = 2.0,
        dn_diff_hard_bypass_high_snr: bool = True,
        dn_diff_cond_diagnostic: str = "none",  # none|zero|shuffle
        dn_diff_force_deterministic_multisample: bool = False,
        noise_head_hidden: int = 32,
        expert_features: bool = False,  # Enable expert feature branch
        expert_channels: int = 64,  # Channels for expert feature CNN
        expert_stacf_window: int = 0,  # Window size for short-time ACF smoothing (0/1=off)
        expert_v2: bool = False,  # Use normalized local-correlation expert extractor (v2)
        expert_corr_norm_eps: float = 1e-6,  # Epsilon for normalized local-correlation expert features
        expert_eta_gate: bool = False,  # Gate expert branch by predicted eta to avoid high-SNR harm
        expert_eta_gate_center: float = 0.8,  # eta value where expert gate is ~0.5
        expert_eta_gate_tau: float = 0.7,  # softness for expert eta gating sigmoid
        expert_eta_gate_min: float = 0.0,  # minimum expert gate scale
        expert_eta_gate_max: float = 1.0,  # maximum expert gate scale
        expert_use_cyclo_stats: bool = True,  # Append global cyclostationary stats to classifier feature
        raw_low_snr_drop_prob: float = 0.0,  # Probability of attenuating raw branch in dual-path mode
        raw_low_snr_drop_gate: str = "auto",  # Gate source for raw attenuation: auto|eta|snr
        raw_low_snr_drop_eta_thresh: float = 1.0,  # Apply raw-branch attenuation when eta >= threshold
        raw_low_snr_drop_snr_thresh: float = -6.0,  # Fallback gate when eta unavailable: apply when snr <= threshold
        raw_low_snr_drop_min_scale: float = 0.0,  # Min multiplicative scale for gated raw branch
        raw_low_snr_drop_max_scale: float = 0.0,  # Max multiplicative scale for gated raw branch
        raw_low_snr_drop_prob_lo: float = -1.0,  # Optional SNR-shaped schedule prob for very low SNR
        raw_low_snr_drop_prob_mid: float = -1.0,  # Optional SNR-shaped schedule prob for transition SNR
        raw_low_snr_drop_prob_hi: float = -1.0,  # Optional SNR-shaped schedule prob for high SNR
        raw_low_snr_drop_snr_lo: float = -10.0,  # lower SNR split for schedule
        raw_low_snr_drop_snr_mid: float = -6.0,  # middle SNR split for schedule
        cls_hidden: int = 0,  # Classifier hidden dim (0 = auto: max(128, num_classes*8))
        moe_n_experts: int = 1,
        moe_gate_type: str = "eta-sigmoid",
        moe_gate_center: float = 0.5,
        moe_gate_tau: float = 0.3,
        moe_gate_use_feat: bool = False,
        supcon_proj_dim: int = 0,  # >0 enables a SupCon MLP projection head
    ) -> None:
        super().__init__()
        if pool not in {"attn", "last", "mean"}:
            raise ValueError(f"pool must be one of attn|last|mean, got {pool}")
        self.seq_len = int(seq_len)
        if float(snr_min_db) >= float(snr_max_db):
            raise ValueError(f"snr_min_db must be < snr_max_db (got {snr_min_db} vs {snr_max_db})")
        self.snr_min_db = float(snr_min_db)
        self.snr_max_db = float(snr_max_db)
        self._snr_center_db = 0.5 * (self.snr_min_db + self.snr_max_db)
        self._snr_half_range_db = 0.5 * (self.snr_max_db - self.snr_min_db)
        self.time_out = max(1, self.seq_len - 4)  # after conv_merge / expert conv_reduce (kernel=5 valid)
        self.pool = pool
        self.cldnn_backbone = str(cldnn_backbone).strip().lower()
        if self.cldnn_backbone not in {"lstm", "tcn", "resnet1d"}:
            raise ValueError("cldnn_backbone must be one of: lstm|tcn|resnet1d.")
        self.cldnn_tcn_levels = int(cldnn_tcn_levels)
        self.cldnn_tcn_channels = int(cldnn_tcn_channels)
        self.cldnn_tcn_kernel = int(cldnn_tcn_kernel)
        self.cldnn_tcn_dilation_base = int(cldnn_tcn_dilation_base)
        self.cldnn_tcn_dropout = float(cldnn_tcn_dropout)
        self.cldnn_resnet_blocks = int(cldnn_resnet_blocks)
        self.cldnn_resnet_channels = int(cldnn_resnet_channels)
        self.cldnn_resnet_kernel = int(cldnn_resnet_kernel)
        self.cldnn_resnet_dilation_cycle = int(cldnn_resnet_dilation_cycle)
        self.cldnn_resnet_dropout = float(cldnn_resnet_dropout)
        self.snr_cond = bool(snr_cond)
        self.noise_cond = bool(noise_cond)
        self.snr_loss_detach_backbone = bool(snr_loss_detach_backbone)
        self.use_expert = bool(expert_features)
        self.denoiser_enabled = bool(denoiser)
        self.dn_diff_enabled = bool(dn_diff_enable)
        self.dn_diff_target = str(dn_diff_target).strip().lower()
        self.dn_diff_train_timesteps = int(dn_diff_train_timesteps)
        self.dn_diff_eval_mode = str(dn_diff_eval_mode).strip().lower()
        self.dn_diff_eval_steps = int(dn_diff_eval_steps)
        self.dn_diff_ddim_eta = float(dn_diff_ddim_eta)
        self.dn_diff_ddim_eta_used = float(dn_diff_ddim_eta)
        self.dn_diff_multisample = int(dn_diff_multisample)
        self.dn_diff_eval_t_start_source = str(dn_diff_eval_t_start_source).strip().lower()
        self.dn_diff_fixed_t_start = int(dn_diff_fixed_t_start)
        self.dn_diff_snr2t_scale = float(dn_diff_snr2t_scale)
        self.dn_diff_snr2t_bias = float(dn_diff_snr2t_bias)
        self.dn_diff_low_snr_thresh = float(dn_diff_low_snr_thresh)
        self.dn_diff_high_snr_margin = float(dn_diff_high_snr_margin)
        self.dn_diff_hard_bypass_high_snr = bool(dn_diff_hard_bypass_high_snr)
        self.dn_diff_cond_diagnostic = str(dn_diff_cond_diagnostic).strip().lower()
        self.dn_diff_force_deterministic_multisample = bool(dn_diff_force_deterministic_multisample)
        self._dn_diff_t_start: Optional[torch.Tensor] = None
        self._dn_diff_active_mask: Optional[torch.Tensor] = None
        if self.dn_diff_target not in {"v", "eps"}:
            raise ValueError("dn_diff_target must be one of: v | eps.")
        if self.dn_diff_eval_mode not in {"ddim", "onestep"}:
            raise ValueError("dn_diff_eval_mode must be one of: ddim | onestep.")
        if self.dn_diff_eval_t_start_source not in {"snr_pred", "snr_true", "fixed"}:
            raise ValueError("dn_diff_eval_t_start_source must be one of: snr_pred|snr_true|fixed.")
        if self.dn_diff_cond_diagnostic not in {"none", "zero", "shuffle"}:
            raise ValueError("dn_diff_cond_diagnostic must be one of: none|zero|shuffle.")
        if self.dn_diff_train_timesteps <= 1:
            raise ValueError("dn_diff_train_timesteps must be > 1.")
        if self.dn_diff_eval_steps <= 0:
            raise ValueError("dn_diff_eval_steps must be > 0.")
        if self.dn_diff_multisample <= 0:
            raise ValueError("dn_diff_multisample must be > 0.")
        if bool(denoiser_dual_path) and not (self.denoiser_enabled or self.dn_diff_enabled):
            raise ValueError("denoiser_dual_path requires denoiser=True or dn_diff_enable=True.")
        self.denoiser_dual_path = bool(denoiser_dual_path) if (self.denoiser_enabled or self.dn_diff_enabled) else False
        self.classifier_paths = 2 if self.denoiser_dual_path else 1
        self.force_denoiser_bypass = False
        if self.snr_cond and self.noise_cond:
            raise ValueError("snr_cond and noise_cond are mutually exclusive; enable only one conditioning path.")
        if float(noise_eta_min) >= float(noise_eta_max):
            raise ValueError(
                f"noise_eta_min must be < noise_eta_max (got {noise_eta_min} vs {noise_eta_max})"
            )
        self.noise_eta_min = float(noise_eta_min)
        self.noise_eta_max = float(noise_eta_max)
        self._noise_eta_center = 0.5 * (self.noise_eta_min + self.noise_eta_max)
        self._noise_eta_half_range = 0.5 * (self.noise_eta_max - self.noise_eta_min)
        self.expert_stacf_window = max(0, int(expert_stacf_window))
        self.expert_v2 = bool(expert_v2)
        self.expert_corr_norm_eps = max(1e-9, float(expert_corr_norm_eps))
        self.expert_eta_gate = bool(expert_eta_gate)
        self.expert_eta_gate_center = float(expert_eta_gate_center)
        self.expert_eta_gate_tau = float(expert_eta_gate_tau)
        self.expert_eta_gate_min = float(expert_eta_gate_min)
        self.expert_eta_gate_max = float(expert_eta_gate_max)
        if self.expert_eta_gate_tau <= 0.0:
            raise ValueError("expert_eta_gate_tau must be > 0.")
        if (
            self.expert_eta_gate_min < 0.0
            or self.expert_eta_gate_max < 0.0
            or self.expert_eta_gate_min > 1.0
            or self.expert_eta_gate_max > 1.0
            or self.expert_eta_gate_max < self.expert_eta_gate_min
        ):
            raise ValueError("expert_eta_gate_min/max must be in [0,1] and max >= min.")
        self.expert_use_cyclo_stats = bool(expert_use_cyclo_stats)
        self.raw_low_snr_drop_prob = float(raw_low_snr_drop_prob)
        if self.raw_low_snr_drop_prob < 0.0 or self.raw_low_snr_drop_prob > 1.0:
            raise ValueError("raw_low_snr_drop_prob must be in [0,1].")
        self.raw_low_snr_drop_gate = str(raw_low_snr_drop_gate).strip().lower()
        if self.raw_low_snr_drop_gate not in {"auto", "eta", "snr"}:
            raise ValueError("raw_low_snr_drop_gate must be one of: auto|eta|snr.")
        self.raw_low_snr_drop_eta_thresh = float(raw_low_snr_drop_eta_thresh)
        self.raw_low_snr_drop_snr_thresh = float(raw_low_snr_drop_snr_thresh)
        self.raw_low_snr_drop_min_scale = float(raw_low_snr_drop_min_scale)
        self.raw_low_snr_drop_max_scale = float(raw_low_snr_drop_max_scale)
        if self.raw_low_snr_drop_min_scale < 0.0 or self.raw_low_snr_drop_max_scale < 0.0:
            raise ValueError("raw_low_snr_drop_min_scale/max_scale must be >= 0.")
        if self.raw_low_snr_drop_min_scale > 1.0 or self.raw_low_snr_drop_max_scale > 1.0:
            raise ValueError("raw_low_snr_drop_min_scale/max_scale must be <= 1.")
        if self.raw_low_snr_drop_max_scale < self.raw_low_snr_drop_min_scale:
            raise ValueError("raw_low_snr_drop_max_scale must be >= raw_low_snr_drop_min_scale.")
        self.raw_low_snr_drop_prob_lo = float(raw_low_snr_drop_prob_lo)
        self.raw_low_snr_drop_prob_mid = float(raw_low_snr_drop_prob_mid)
        self.raw_low_snr_drop_prob_hi = float(raw_low_snr_drop_prob_hi)
        self.raw_low_snr_drop_snr_lo = float(raw_low_snr_drop_snr_lo)
        self.raw_low_snr_drop_snr_mid = float(raw_low_snr_drop_snr_mid)
        self.raw_low_snr_drop_schedule = (
            self.raw_low_snr_drop_prob_lo >= 0.0
            and self.raw_low_snr_drop_prob_mid >= 0.0
            and self.raw_low_snr_drop_prob_hi >= 0.0
        )
        if self.raw_low_snr_drop_schedule:
            for p in (
                self.raw_low_snr_drop_prob_lo,
                self.raw_low_snr_drop_prob_mid,
                self.raw_low_snr_drop_prob_hi,
            ):
                if p < 0.0 or p > 1.0:
                    raise ValueError("raw_low_snr_drop_prob_lo/mid/hi must be in [0,1] when schedule is enabled.")
            if self.raw_low_snr_drop_snr_lo > self.raw_low_snr_drop_snr_mid:
                raise ValueError("raw_low_snr_drop_snr_lo must be <= raw_low_snr_drop_snr_mid.")
        self.moe_n_experts = int(moe_n_experts)
        self.moe_gate_type = str(moe_gate_type).strip().lower()
        self.moe_gate_center = float(moe_gate_center)
        self.moe_gate_tau = float(moe_gate_tau)
        self.moe_gate_use_feat = bool(moe_gate_use_feat)
        if self.moe_n_experts < 1:
            raise ValueError("moe_n_experts must be >= 1.")
        if self.moe_gate_tau <= 0.0:
            raise ValueError("moe_gate_tau must be > 0.")
        if self.moe_gate_type not in {"eta-sigmoid", "learned", "hard"}:
            raise ValueError("moe_gate_type must be one of: eta-sigmoid|learned|hard.")
        if self.moe_n_experts == 1:
            # Keep compatibility: MoE disabled in single-expert mode regardless of gate settings.
            self.moe_gate_type = "eta-sigmoid"
        if self.cldnn_tcn_levels <= 0:
            raise ValueError("cldnn_tcn_levels must be > 0.")
        if self.cldnn_tcn_channels <= 0:
            raise ValueError("cldnn_tcn_channels must be > 0.")
        if self.cldnn_tcn_kernel <= 0 or (self.cldnn_tcn_kernel % 2) == 0:
            raise ValueError("cldnn_tcn_kernel must be odd and > 0.")
        if self.cldnn_tcn_dilation_base <= 0:
            raise ValueError("cldnn_tcn_dilation_base must be > 0.")
        if self.cldnn_tcn_dropout < 0.0:
            raise ValueError("cldnn_tcn_dropout must be >= 0.")
        if self.cldnn_resnet_blocks <= 0:
            raise ValueError("cldnn_resnet_blocks must be > 0.")
        if self.cldnn_resnet_channels <= 0:
            raise ValueError("cldnn_resnet_channels must be > 0.")
        if self.cldnn_resnet_kernel <= 0 or (self.cldnn_resnet_kernel % 2) == 0:
            raise ValueError("cldnn_resnet_kernel must be odd and > 0.")
        if self.cldnn_resnet_dilation_cycle <= 0:
            raise ValueError("cldnn_resnet_dilation_cycle must be > 0.")
        if self.cldnn_resnet_dropout < 0.0:
            raise ValueError("cldnn_resnet_dropout must be >= 0.")

        # Branch 1: IQ joint Conv2D over (2 x L) per path.
        self.conv_iq = nn.Conv2d(
            in_channels=self.classifier_paths,
            out_channels=conv_channels,
            kernel_size=(2, 8),
            padding="same",
        )

        # Branch 2/3: I-only and Q-only causal Conv1D
        self.conv_i = CausalConv1d(self.classifier_paths, conv_channels, kernel_size=8)
        self.conv_q = CausalConv1d(self.classifier_paths, conv_channels, kernel_size=8)

        # Fuse I/Q branches with a Conv2D over time only (kernel 1x8)
        self.conv_fuse = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=conv_channels,
            kernel_size=(1, 8),
            padding="same",
        )

        # Merge IQ + fused(I,Q) then reduce with Conv2D (kernel 2x5, valid)
        self.conv_merge = nn.Conv2d(
            in_channels=2 * conv_channels,
            out_channels=merge_channels,
            kernel_size=(2, 5),
            padding=0,
        )

        self.act = nn.ReLU()
        self.drop = nn.Dropout(float(dropout))

        # Expert feature branch (conjugate products, magnitude, phase diffs)
        if self.use_expert:
            self.expert_extractor = ExpertFeatureExtractor(
                out_channels=expert_channels,
                time_out=self.time_out,
                dropout=float(dropout),
                stacf_window=self.expert_stacf_window,
                v2_mode=self.expert_v2,
                corr_norm_eps=self.expert_corr_norm_eps,
            )
            self.cyclo_stats = CyclostationaryStats(seq_len=self.seq_len) if self.expert_use_cyclo_stats else None
            reduce_in = merge_channels + expert_channels
        else:
            self.expert_extractor = None
            self.cyclo_stats = None
            reduce_in = merge_channels

        # -----------------------------------------------------------------
        # Temporal reduction: strided convolutions to reduce T for the
        # temporal backbone (LSTM/TCN/ResNet1D).
        # -----------------------------------------------------------------
        self._target_lstm_len = 128  # historical target for temporal sequence length
        reduce_layers: list = []
        t_len = self.time_out
        while t_len > self._target_lstm_len * 1.5:
            # Each block: Conv1d(stride=2) + BN + ReLU + Dropout
            out_ch = merge_channels  # keep channel count consistent
            k = 5 if t_len > 256 else 3
            pad = k // 2
            reduce_layers.append(nn.Conv1d(reduce_in, out_ch, kernel_size=k, stride=2, padding=pad))
            reduce_layers.append(nn.BatchNorm1d(out_ch))
            reduce_layers.append(nn.ReLU())
            reduce_layers.append(nn.Dropout(float(dropout)))
            reduce_in = out_ch
            t_len = (t_len + 2 * pad - k) // 2 + 1  # conv output formula

        if reduce_layers:
            self.temporal_reduce = nn.Sequential(*reduce_layers)
        else:
            self.temporal_reduce = None
        temporal_input_size = reduce_in
        self._lstm_time_len = t_len  # actual length after reduction

        self.tcn_backbone: Optional[TemporalBackboneTCN] = None
        self.resnet_backbone: Optional[TemporalBackboneResNet1D] = None
        if self.cldnn_backbone == "lstm":
            # Preserve legacy default path exactly.
            self.lstm = nn.LSTM(
                input_size=temporal_input_size,
                hidden_size=lstm_hidden,
                num_layers=int(lstm_layers),
                batch_first=True,
                bidirectional=bool(bidirectional),
                dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
            )
            temporal_out_dim = lstm_hidden * (2 if bidirectional else 1)
        elif self.cldnn_backbone == "tcn":
            self.lstm = None
            self.tcn_backbone = TemporalBackboneTCN(
                in_channels=temporal_input_size,
                channels=self.cldnn_tcn_channels,
                levels=self.cldnn_tcn_levels,
                kernel_size=self.cldnn_tcn_kernel,
                dilation_base=self.cldnn_tcn_dilation_base,
                dropout=self.cldnn_tcn_dropout,
            )
            temporal_out_dim = int(self.tcn_backbone.output_dim)
        else:
            self.lstm = None
            self.resnet_backbone = TemporalBackboneResNet1D(
                in_channels=temporal_input_size,
                channels=self.cldnn_resnet_channels,
                blocks=self.cldnn_resnet_blocks,
                kernel_size=self.cldnn_resnet_kernel,
                dilation_cycle=self.cldnn_resnet_dilation_cycle,
                dropout=self.cldnn_resnet_dropout,
            )
            temporal_out_dim = int(self.resnet_backbone.output_dim)
        self.temporal_out_dim = temporal_out_dim
        self.lstm_out_dim = temporal_out_dim
        self.attn_pool = TemporalAttentionPool(temporal_out_dim, hidden=max(64, temporal_out_dim // 2))

        # Conditioning path: either SNR-FiLM or noise-fraction-FiLM (eta).
        if self.snr_cond:
            self.snr_embed = nn.Sequential(
                nn.Linear(1, snr_embed_dim),
                nn.GELU(),
                nn.Linear(snr_embed_dim, snr_embed_dim),
                nn.GELU(),
            )
            # FiLM: gamma (scale) and beta (shift) for the pooled feature
            self.film_gamma = nn.Linear(snr_embed_dim, temporal_out_dim)
            self.film_beta = nn.Linear(snr_embed_dim, temporal_out_dim)
            # Stabilize training: start FiLM close to identity (gamma≈1, beta≈0)
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.ones_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
            self.noise_embed = None
            self.noise_film_gamma = None
            self.noise_film_beta = None
            self.noise_delta_head = None
            self.noise_proxy_scale_raw = None
            self.noise_proxy_bias = None
        elif self.noise_cond:
            self.snr_embed = None
            self.film_gamma = None
            self.film_beta = None
            # Learn a residual correction on top of an analytic monotonic proxy.
            self.noise_proxy_scale_raw = nn.Parameter(torch.tensor(1.0))
            self.noise_proxy_bias = nn.Parameter(torch.tensor(0.0))
            self.noise_delta_head = nn.Linear(temporal_out_dim, 1)
            self.noise_embed = nn.Sequential(
                nn.Linear(1, snr_embed_dim),
                nn.GELU(),
                nn.Linear(snr_embed_dim, snr_embed_dim),
                nn.GELU(),
            )
            self.noise_film_gamma = nn.Linear(snr_embed_dim, temporal_out_dim)
            self.noise_film_beta = nn.Linear(snr_embed_dim, temporal_out_dim)
            nn.init.zeros_(self.noise_film_gamma.weight)
            nn.init.ones_(self.noise_film_gamma.bias)
            nn.init.zeros_(self.noise_film_beta.weight)
            nn.init.zeros_(self.noise_film_beta.bias)
        else:
            self.snr_embed = None
            self.film_gamma = None
            self.film_beta = None
            self.noise_embed = None
            self.noise_film_gamma = None
            self.noise_film_beta = None
            self.noise_delta_head = None
            self.noise_proxy_scale_raw = None
            self.noise_proxy_bias = None

        # Optional denoiser/noise-head paths.
        if self.denoiser_enabled or self.dn_diff_enabled:
            self.noise_fraction_net = NoiseFractionNet(
                in_channels=2,
                hidden_channels=int(noise_head_hidden),
                eta_min=self.noise_eta_min,
                eta_max=self.noise_eta_max,
            )
        else:
            self.noise_fraction_net = None

        if self.denoiser_enabled:
            self.denoiser = DenoiserUNet1D(
                in_channels=2,
                base_channels=int(denoiser_base_channels),
                cond_dim=max(16, int(noise_head_hidden)),
                dropout=float(denoiser_dropout),
                soft_high_snr_blend=bool(denoiser_soft_high_snr_blend),
                eta_hi=float(denoiser_eta_hi),
                tau_eta=float(denoiser_tau_eta),
            )
        else:
            self.denoiser = None

        if self.dn_diff_enabled:
            self.dn_diff_model = WaveformDiffusionDenoiser1D(
                in_channels=2,
                base_channels=max(32, int(denoiser_base_channels)),
                cond_dim=max(16, int(noise_head_hidden)),
                dropout=float(denoiser_dropout),
            )
            self.dn_diff_schedule = DiffusionSchedule(timesteps=self.dn_diff_train_timesteps)
        else:
            self.dn_diff_model = None
            self.dn_diff_schedule = None

        # Classifier head — auto-scale hidden size to num_classes
        n_cyclo = self.cyclo_stats.N_STATS if self.cyclo_stats is not None else 0
        cls_input_dim = temporal_out_dim + n_cyclo
        if cls_hidden > 0:
            _cls_h = int(cls_hidden)
        else:
            _cls_h = max(128, num_classes * 8)  # e.g., 128 for 11 cls, 192 for 24 cls
        self.fc1 = nn.Linear(cls_input_dim, _cls_h)
        self.fc2 = nn.Linear(_cls_h, _cls_h)
        self.fc_out = nn.Linear(_cls_h, num_classes)
        self.fc_act = nn.SELU()
        self.moe_head: Optional[MoEClassifierHead] = None
        if self.moe_n_experts > 1:
            self.moe_head = MoEClassifierHead(
                input_dim=cls_input_dim,
                hidden_dim=_cls_h,
                num_classes=num_classes,
                dropout=float(dropout),
                n_experts=self.moe_n_experts,
                gate_type=self.moe_gate_type,
                gate_center=self.moe_gate_center,
                gate_tau=self.moe_gate_tau,
                gate_use_feat=self.moe_gate_use_feat,
            )

        # SNR head for auxiliary supervision and predict-mode FiLM
        self.snr_head = nn.Linear(temporal_out_dim, 1)

        # SupCon projection head (optional).  Maps pre-FiLM pooled features
        # to a low-dim L2-normalised space for supervised contrastive loss.
        # 2-layer MLP: Linear → BN → ReLU → Linear  (following Khosla et al.)
        self.supcon_proj_dim = int(supcon_proj_dim)
        if self.supcon_proj_dim > 0:
            self.proj_head = nn.Sequential(
                nn.Linear(temporal_out_dim, temporal_out_dim),
                nn.BatchNorm1d(temporal_out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(temporal_out_dim, self.supcon_proj_dim),
            )
        else:
            self.proj_head = None

        # Will hold the pre-FiLM pooled features for SupCon projection
        # (set during _forward_windows, consumed externally after forward())
        self._pooled_pre_film: Optional[torch.Tensor] = None
        # Optional auxiliary eta predictions (noise-fraction logit), set during forward.
        self._eta_pred_flat: Optional[torch.Tensor] = None
        self._eta_pred: Optional[torch.Tensor] = None
        self._expert_gate_flat: Optional[torch.Tensor] = None
        self._expert_gate: Optional[torch.Tensor] = None
        self._x_dn_flat: Optional[torch.Tensor] = None
        self._x_dn: Optional[torch.Tensor] = None
        self._moe_gate: Optional[torch.Tensor] = None
        self._moe_expert_load: Optional[torch.Tensor] = None
        self._moe_logits_experts: Optional[torch.Tensor] = None
        self._moe_logits_experts_aux: Optional[torch.Tensor] = None
        # Frozen early-feature encoder used by perceptual/feature-preservation loss.
        # Stored outside nn.Module registration to keep checkpoint compatibility.
        self.__dict__["_feat_encoder"] = None

    def set_noise_proxy_calibration(self, scale: float, bias: float) -> None:
        if self.noise_fraction_net is not None:
            self.noise_fraction_net.set_proxy_calibration(scale=scale, bias=bias)

    def _eta_from_snr(self, snr: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if snr is None:
            return None
        return snr_db_to_eta(
            snr,
            eta_min=self.noise_eta_min,
            eta_max=self.noise_eta_max,
        )

    @staticmethod
    def _snr_from_eta(eta: torch.Tensor) -> torch.Tensor:
        rho = torch.clamp(torch.sigmoid(eta.float()), min=1e-6, max=1.0 - 1e-6)
        snr_lin = torch.clamp((1.0 - rho) / rho, min=1e-8)
        return 10.0 * torch.log10(snr_lin)

    def _dn_diff_apply_cond_diagnostic(self, x_cond: torch.Tensor, mode: str) -> torch.Tensor:
        m = str(mode).strip().lower()
        if m == "none":
            return x_cond
        if m == "zero":
            return torch.zeros_like(x_cond)
        if m == "shuffle":
            if x_cond.shape[0] <= 1:
                return x_cond
            perm = torch.randperm(x_cond.shape[0], device=x_cond.device)
            return x_cond[perm]
        raise ValueError("dn_diff_cond_diagnostic must be one of: none|zero|shuffle.")

    def dn_diff_predict(
        self,
        x_t: torch.Tensor,
        x_cond: torch.Tensor,
        t: torch.Tensor,
        snr: Optional[torch.Tensor] = None,
        snr_mode: str = "predict",
        cond_diagnostic: Optional[str] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        if self.dn_diff_model is None:
            raise RuntimeError("dn_diff_predict requires dn_diff_enable=True.")
        eta_pred = None
        if self.noise_fraction_net is not None:
            eta_pred, _aux = self.noise_fraction_net(x_cond)
        eta_cond = None
        if snr_mode == "known" and snr is not None:
            eta_cond = self._eta_from_snr(snr)
        elif snr_mode == "predict" and eta_pred is not None:
            eta_cond = eta_pred.detach()
        if eta_cond is None:
            eta_cond = torch.zeros((x_t.shape[0],), device=x_t.device, dtype=x_t.dtype)
        diag_mode = self.dn_diff_cond_diagnostic if cond_diagnostic is None else cond_diagnostic
        x_cond_in = self._dn_diff_apply_cond_diagnostic(x_cond, diag_mode)
        pred = self.dn_diff_model(x_t=x_t, x_cond=x_cond_in, t=t, eta_cond=eta_cond)
        return pred, eta_pred, eta_cond

    def _dn_diff_map_snr_to_t(self, snr_db: torch.Tensor) -> torch.Tensor:
        if self.dn_diff_schedule is None:
            raise RuntimeError("dn_diff_schedule is not initialized.")
        sched = self.dn_diff_schedule.to(snr_db.device)
        t_base = sched.snr_to_t(snr_db.float())
        t_scaled = self.dn_diff_snr2t_scale * t_base.float() + self.dn_diff_snr2t_bias
        return torch.clamp(torch.round(t_scaled).long(), min=0, max=sched.timesteps - 1)

    def _dn_diff_eval_t_start(
        self,
        x_raw: torch.Tensor,
        snr_flat: Optional[torch.Tensor],
        snr_mode: str,
    ) -> torch.Tensor:
        if self.dn_diff_schedule is None:
            raise RuntimeError("dn_diff_schedule is not initialized.")
        src = str(self.dn_diff_eval_t_start_source).strip().lower()
        if src == "fixed":
            t = torch.full(
                (x_raw.shape[0],),
                int(self.dn_diff_fixed_t_start),
                device=x_raw.device,
                dtype=torch.long,
            )
            return torch.clamp(t, min=0, max=self.dn_diff_schedule.timesteps - 1)
        if src == "snr_true" and snr_flat is not None:
            return self._dn_diff_map_snr_to_t(snr_flat)

        snr_est = None
        if self.noise_fraction_net is not None:
            eta_pred, _aux = self.noise_fraction_net(x_raw)
            snr_est = self._snr_from_eta(eta_pred.detach())
        if snr_est is None:
            if snr_flat is not None:
                snr_est = snr_flat.float()
            else:
                snr_est = torch.zeros((x_raw.shape[0],), device=x_raw.device, dtype=x_raw.dtype)
        return self._dn_diff_map_snr_to_t(snr_est)

    def _dn_diff_eval_reconstruct(
        self,
        x_raw: torch.Tensor,
        snr_flat: Optional[torch.Tensor],
        snr_mode: str,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.dn_diff_schedule is None:
            raise RuntimeError("dn_diff_schedule is not initialized.")
        sched = self.dn_diff_schedule.to(x_raw.device)
        t_start = self._dn_diff_eval_t_start(x_raw=x_raw, snr_flat=snr_flat, snr_mode=snr_mode)
        self._dn_diff_t_start = t_start.detach()

        n_samples = max(1, int(self.dn_diff_multisample))
        eta_eff = float(self.dn_diff_ddim_eta)
        if (
            n_samples > 1
            and abs(eta_eff) < 1e-12
            and not bool(self.dn_diff_force_deterministic_multisample)
        ):
            eta_eff = 0.2
        self.dn_diff_ddim_eta_used = float(eta_eff)

        x_samples = []
        eta_pred_last = None
        eta_cond_last = None
        for _ in range(n_samples):
            x_noise = torch.randn_like(x_raw)
            x_start = sched.q_sample(x_raw, t_start, x_noise)
            if self.dn_diff_eval_mode == "onestep":
                pred, eta_pred, eta_cond = self.dn_diff_predict(
                    x_t=x_start,
                    x_cond=x_raw,
                    t=t_start,
                    snr=snr_flat,
                    snr_mode=snr_mode,
                )
                if self.dn_diff_target == "v":
                    x0_pred = sched.predict_x0_from_v(x_start, t_start, pred)
                else:
                    x0_pred = sched.predict_x0_from_eps(x_start, t_start, pred)
            else:
                eta_pred = None
                eta_cond = None

                def _denoise_fn(x_cur: torch.Tensor, t_cur: torch.Tensor) -> torch.Tensor:
                    pred_cur, _eta_pred_cur, _eta_cond_cur = self.dn_diff_predict(
                        x_t=x_cur,
                        x_cond=x_raw,
                        t=t_cur,
                        snr=snr_flat,
                        snr_mode=snr_mode,
                    )
                    return pred_cur

                _x_final, x0_pred = sched.ddim_sample_loop(
                    denoise_fn=_denoise_fn,
                    x_start=x_start,
                    t_start=t_start,
                    steps=int(self.dn_diff_eval_steps),
                    target=self.dn_diff_target,
                    eta=float(eta_eff),
                    clip_x0=False,
                )
            x_samples.append(x0_pred)
            eta_pred_last = eta_pred
            eta_cond_last = eta_cond

        x_dn = torch.stack(x_samples, dim=0).mean(dim=0)
        snr_for_bypass = snr_flat
        if snr_for_bypass is None and self.noise_fraction_net is not None:
            eta_bypass, _aux_bypass = self.noise_fraction_net(x_raw)
            snr_for_bypass = self._snr_from_eta(eta_bypass.detach())
        if bool(self.dn_diff_hard_bypass_high_snr) and snr_for_bypass is not None:
            cutoff = float(self.dn_diff_low_snr_thresh + self.dn_diff_high_snr_margin)
            bypass_mask = snr_for_bypass.float() > cutoff
            self._dn_diff_active_mask = (~bypass_mask).float()
            x_dn = torch.where(bypass_mask.view(-1, 1, 1), x_raw, x_dn)
        else:
            self._dn_diff_active_mask = torch.ones((x_raw.shape[0],), device=x_raw.device, dtype=x_raw.dtype)
        return x_dn, eta_pred_last, eta_cond_last

    def set_dn_diff_train_freeze(self, freeze_classifier: bool) -> None:
        if not self.dn_diff_enabled:
            return
        if not bool(freeze_classifier):
            for p in self.parameters():
                p.requires_grad_(True)
            return
        for p in self.parameters():
            p.requires_grad_(False)
        if self.dn_diff_model is not None:
            for p in self.dn_diff_model.parameters():
                p.requires_grad_(True)
        if self.noise_fraction_net is not None:
            for p in self.noise_fraction_net.parameters():
                p.requires_grad_(True)

    @staticmethod
    def _scale_grad(x: torch.Tensor, scale: float) -> torch.Tensor:
        s = float(scale)
        if s <= 0.0:
            return x.detach()
        if s >= 1.0:
            return x
        return x.detach() + s * (x - x.detach())

    def _maybe_low_snr_raw_dropout(
        self,
        x_raw: torch.Tensor,
        eta_cond: Optional[torch.Tensor] = None,
        snr_flat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        In dual-path mode, attenuate raw branch for a subset of low-SNR samples so
        the classifier cannot ignore the denoised path.
        """
        if (
            (not self.training)
            or (not self.denoiser_dual_path)
            or (self.raw_low_snr_drop_prob <= 0.0 and not self.raw_low_snr_drop_schedule)
        ):
            return x_raw

        b = int(x_raw.shape[0])
        gate = None
        snr_f = None
        gate_mode = str(self.raw_low_snr_drop_gate)
        if gate_mode == "eta":
            if eta_cond is not None:
                eta_f = eta_cond.float().view(-1)
                if eta_f.numel() == b:
                    gate = eta_f >= float(self.raw_low_snr_drop_eta_thresh)
        elif gate_mode == "snr":
            if snr_flat is not None:
                snr_f = snr_flat.float().view(-1)
                if snr_f.numel() == b:
                    gate = snr_f <= float(self.raw_low_snr_drop_snr_thresh)
        else:
            if eta_cond is not None:
                eta_f = eta_cond.float().view(-1)
                if eta_f.numel() == b:
                    gate = eta_f >= float(self.raw_low_snr_drop_eta_thresh)
            if gate is None and snr_flat is not None:
                snr_f = snr_flat.float().view(-1)
                if snr_f.numel() == b:
                    gate = snr_f <= float(self.raw_low_snr_drop_snr_thresh)
        if gate is None:
            return x_raw
        if not bool(torch.any(gate)):
            return x_raw

        # Optional SNR-shaped probability schedule:
        #   snr <= split_lo  -> prob_lo
        #   split_lo < snr <= split_mid -> prob_mid
        #   snr > split_mid -> prob_hi
        # Falls back to scalar raw_low_snr_drop_prob when schedule is disabled
        # or SNR labels are unavailable.
        if self.raw_low_snr_drop_schedule and snr_f is not None and snr_f.numel() == b:
            p_lo = float(self.raw_low_snr_drop_prob_lo)
            p_mid = float(self.raw_low_snr_drop_prob_mid)
            p_hi = float(self.raw_low_snr_drop_prob_hi)
            s_lo = float(self.raw_low_snr_drop_snr_lo)
            s_mid = float(self.raw_low_snr_drop_snr_mid)
            p_vec = torch.full((b,), p_hi, device=x_raw.device, dtype=x_raw.dtype)
            p_vec = torch.where(snr_f <= s_mid, torch.full_like(p_vec, p_mid), p_vec)
            p_vec = torch.where(snr_f <= s_lo, torch.full_like(p_vec, p_lo), p_vec)
            p_vec = torch.clamp(p_vec, min=0.0, max=1.0)
        else:
            p_vec = torch.full(
                (b,),
                float(self.raw_low_snr_drop_prob),
                device=x_raw.device,
                dtype=x_raw.dtype,
            )

        apply_mask = gate & (torch.rand((b,), device=x_raw.device) < p_vec)
        if not bool(torch.any(apply_mask)):
            return x_raw

        lo = float(self.raw_low_snr_drop_min_scale)
        hi = float(self.raw_low_snr_drop_max_scale)
        if hi <= lo + 1e-12:
            scales = torch.full((b,), lo, device=x_raw.device, dtype=x_raw.dtype)
        else:
            scales = lo + (hi - lo) * torch.rand((b,), device=x_raw.device, dtype=x_raw.dtype)
        scales = torch.where(apply_mask, scales, torch.ones_like(scales))
        return x_raw * scales.view(-1, 1, 1)

    def denoise_only(
        self,
        x: torch.Tensor,
        snr: Optional[torch.Tensor] = None,
        snr_mode: str = "predict",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Run only the denoiser/noise-head path.
        Returns: (x_dn, eta_pred, eta_cond_used)
        """
        if x.ndim != 3:
            raise ValueError(f"denoise_only expects (B,2,L), got {tuple(x.shape)}")
        if x.shape[1] != 2:
            raise ValueError(f"denoise_only expects channel dim 2, got {x.shape[1]}")

        eta_pred = None
        if self.noise_fraction_net is not None:
            eta_pred, _aux = self.noise_fraction_net(x)

        eta_cond = None
        if snr_mode == "known" and snr is not None:
            eta_cond = self._eta_from_snr(snr)
        elif snr_mode == "predict" and eta_pred is not None:
            eta_cond = eta_pred.detach()

        if eta_cond is None:
            if eta_pred is not None:
                eta_cond = eta_pred.detach()
            else:
                eta_cond = torch.zeros((x.shape[0],), device=x.device, dtype=x.dtype)

        if self.denoiser is None:
            return x, eta_pred, eta_cond
        x_dn = self.denoiser(x, eta_cond)
        return x_dn, eta_pred, eta_cond

    @property
    def has_feat_encoder(self) -> bool:
        return self.__dict__.get("_feat_encoder", None) is not None

    def set_feat_encoder(self, feat_encoder: nn.Module) -> None:
        """
        Set a frozen early-feature encoder snapshot used for feature-preservation loss.
        """
        feat_enc = copy.deepcopy(feat_encoder)
        feat_enc = feat_enc.to(device=next(self.parameters()).device, dtype=torch.float32)
        feat_enc.eval()
        for p in feat_enc.parameters():
            p.requires_grad_(False)
        self.__dict__["_feat_encoder"] = feat_enc

    def build_feat_encoder(self) -> None:
        """
        Snapshot early CLDNN convolutions as a fixed feature encoder.
        """
        feat_enc = nn.ModuleDict(
            {
                "conv_iq": copy.deepcopy(self.conv_iq),
                "conv_i": copy.deepcopy(self.conv_i),
                "conv_q": copy.deepcopy(self.conv_q),
                "conv_fuse": copy.deepcopy(self.conv_fuse),
                "conv_merge": copy.deepcopy(self.conv_merge),
            }
        )
        self.set_feat_encoder(feat_enc)

    def early_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frozen early features in float32 while preserving gradients w.r.t input x.

        Args:
            x: (B, C, L) where C == 2 * classifier_paths (2 for single-path, 4 for dual-path)
               or (B, K, C, L), which will be flattened to (B*K, C, L).
        Returns:
            (B, D) pooled early features.
        """
        feat_enc = self.__dict__.get("_feat_encoder", None)
        if feat_enc is None:
            raise RuntimeError("Feature encoder is not initialized. Call build_feat_encoder() first.")

        if x.ndim == 4:
            x = x.reshape(-1, x.shape[2], x.shape[3])
        if x.ndim != 3:
            raise ValueError(f"early_features expects (B,C,L) or (B,K,C,L), got {tuple(x.shape)}")

        expected_c = 2 * int(self.classifier_paths)
        if x.shape[1] != expected_c:
            raise ValueError(f"early_features expects C={expected_c}, got C={x.shape[1]}")

        x_f = x.float()
        b = x_f.shape[0]
        L = x_f.shape[-1]
        x_paths = x_f.view(b, self.classifier_paths, 2, L)

        # Mirror the early CLDNN stack but intentionally omit dropout for deterministic targets.
        h_iq = F.relu(feat_enc["conv_iq"](x_paths))
        i = x_paths[:, :, 0, :]
        q = x_paths[:, :, 1, :]
        h_i = F.relu(feat_enc["conv_i"](i)).unsqueeze(2)
        h_q = F.relu(feat_enc["conv_q"](q)).unsqueeze(2)
        h_iq_sep = torch.cat([h_i, h_q], dim=2)
        h_sep = F.relu(feat_enc["conv_fuse"](h_iq_sep))
        h = torch.cat([h_iq, h_sep], dim=1)
        h = F.relu(feat_enc["conv_merge"](h)).squeeze(2)  # (B, merge_ch, T)

        # Time-average to keep memory/compute bounded and stable across sequence lengths.
        return h.mean(dim=-1)

    def _forward_windows(
        self,
        x_flat: torch.Tensor,
        snr_flat: Optional[torch.Tensor] = None,
        snr_mode: str = "none",
        eta_pred_external: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_flat: (B*, 2, L) or (B*, 4, L) when dual-path is enabled
        snr_flat: (B*,) optional SNR values for conditioning
        snr_mode: "known" | "predict" | "none"
        returns:
          feat: (B*, D_cls) where D_cls = temporal_out_dim + n_global_stats (if expert)
          snr_pred: (B*,)
        """
        b = x_flat.shape[0]
        # IQ joint branch
        L = x_flat.shape[-1]  # seq_len (e.g. 128 or 1024)
        expected_c = 2 * self.classifier_paths
        if x_flat.shape[1] != expected_c:
            raise ValueError(
                f"Expected input channels={expected_c} for classifier_paths={self.classifier_paths}, "
                f"got {x_flat.shape[1]}"
            )
        x_paths = x_flat.view(b, self.classifier_paths, 2, L)
        h_iq = self.drop(self.act(self.conv_iq(x_paths)))  # (B,conv,1,L)

        # I/Q separate branches (causal Conv1D)
        i = x_paths[:, :, 0, :]  # (B,paths,L)
        q = x_paths[:, :, 1, :]  # (B,paths,L)
        h_i = self.drop(self.act(self.conv_i(i)))  # (B,conv,L)
        h_q = self.drop(self.act(self.conv_q(q)))  # (B,conv,L)
        h_i = h_i.unsqueeze(2)  # (B,conv,1,L)
        h_q = h_q.unsqueeze(2)  # (B,conv,1,L)
        h_iq_sep = torch.cat([h_i, h_q], dim=2)  # (B,conv,2,L)
        h_sep = self.drop(self.act(self.conv_fuse(h_iq_sep)))  # (B,conv,2,L)

        # Merge the two streams along channels
        h = torch.cat([h_iq, h_sep], dim=1)  # (B,2*conv,2,L)
        h = self.drop(self.act(self.conv_merge(h)))  # (B,merge,1,L-4)

        # (B, merge, 1, L-4) → (B, merge, L-4)
        h = h.squeeze(2)  # (B, merge, L-4)  — channels-first for temporal_reduce

        # Expert feature branch: concatenate to main branch before temporal reduction
        self._expert_gate_flat = None
        if self.use_expert and self.expert_extractor is not None:
            x_raw = x_flat[:, :2, :]
            h_expert = self.expert_extractor(x_raw)  # (B, expert_ch, L-4)
            if self.expert_eta_gate:
                eta_gate_in = None
                if eta_pred_external is not None:
                    eta_f = eta_pred_external.float().view(-1)
                    if eta_f.numel() == b:
                        eta_gate_in = eta_f
                if eta_gate_in is None and snr_flat is not None:
                    snr_f = snr_flat.float().view(-1)
                    if snr_f.numel() == b:
                        eta_gate_in = snr_db_to_eta(
                            snr_f,
                            eta_min=self.noise_eta_min,
                            eta_max=self.noise_eta_max,
                        )
                if eta_gate_in is not None:
                    tau = max(1e-6, float(self.expert_eta_gate_tau))
                    gate = torch.sigmoid((eta_gate_in - float(self.expert_eta_gate_center)) / tau)
                    gmin = float(self.expert_eta_gate_min)
                    gmax = float(self.expert_eta_gate_max)
                    gate = gmin + (gmax - gmin) * gate
                    self._expert_gate_flat = gate
                    h_expert = h_expert * gate.view(-1, 1, 1)
            h = torch.cat([h, h_expert], dim=1)       # (B, merge+expert_ch, L-4)

        # Temporal reduction (strided convs, channels-first)
        if self.temporal_reduce is not None:
            h = self.temporal_reduce(h)  # (B, C, T_reduced)

        # Temporal backbone (LSTM, TCN, or ResNet1D) -> (B, T, D)
        if self.cldnn_backbone == "lstm":
            h_seq = h.transpose(1, 2)
            temporal_out, _ = self.lstm(h_seq)  # type: ignore[arg-type]
        elif self.cldnn_backbone == "tcn":
            if self.tcn_backbone is None:
                raise RuntimeError("TCN backbone selected but not initialized.")
            temporal_out = self.tcn_backbone(h)
        else:
            if self.resnet_backbone is None:
                raise RuntimeError("ResNet1D backbone selected but not initialized.")
            temporal_out = self.resnet_backbone(h)

        if self.pool == "attn":
            feat, _w = self.attn_pool(temporal_out)
        elif self.pool == "mean":
            feat = temporal_out.mean(dim=1)
        else:  # last
            feat = temporal_out[:, -1, :]

        # Predict SNR from features (before FiLM, so it's not circular)
        feat_for_snr = feat.detach() if self.snr_loss_detach_backbone else feat
        snr_pred = self.snr_head(feat_for_snr).squeeze(-1)  # (B,)
        eta_pred = eta_pred_external

        # Predict eta=logit(rho) using a hybrid analytic+learned head when enabled.
        if self.noise_cond and self.noise_delta_head is not None and eta_pred is None:
            # Analytic proxy: local difference energy ratio (scale-invariant under RMS normalization).
            x_f = x_flat[:, :2, :].float()
            d = x_f[:, :, 1:] - x_f[:, :, :-1]  # (B,2,L-1)
            d_energy = (d * d).sum(dim=1).mean(dim=1)  # (B,)
            p_total = (x_f * x_f).sum(dim=1).mean(dim=1)  # (B,)
            e_proxy = d_energy / (p_total + 1e-8)
            proxy_scale = F.softplus(self.noise_proxy_scale_raw) + 1e-6
            eta_proxy = proxy_scale * e_proxy + self.noise_proxy_bias

            feat_for_eta = feat.detach() if self.snr_loss_detach_backbone else feat
            eta_resid = self.noise_delta_head(feat_for_eta).squeeze(-1)
            eta_pred = torch.clamp(
                eta_proxy + eta_resid,
                min=self.noise_eta_min,
                max=self.noise_eta_max,
            )
        self._eta_pred_flat = eta_pred

        # Save pre-FiLM features for SupCon projection head.
        # This must happen BEFORE FiLM so that SupCon learns SNR-invariant
        # representations (FiLM injects SNR information which we want the
        # contrastive objective to learn *without*).
        self._pooled_pre_film = feat  # (B*, D_lstm) — no clone needed, FiLM creates new tensor

        # Apply FiLM conditioning if enabled.
        if self.snr_cond:
            if snr_mode == "known" and snr_flat is not None:
                # Use ground-truth SNR
                snr_for_film = snr_flat.view(-1, 1).float()
            elif snr_mode == "predict":
                # Use predicted SNR (realistic — no oracle)
                snr_for_film = snr_pred.detach().view(-1, 1)
            else:
                snr_for_film = None

            if snr_for_film is not None:
                # Prevent out-of-range predicted SNR from destabilizing FiLM.
                snr_for_film = torch.clamp(snr_for_film, min=self.snr_min_db, max=self.snr_max_db)
                # Normalize SNR to [-1, 1] using dataset min/max (e.g., RML2016.10a: [-20,18], RML2018.01A: [-20,30]).
                snr_norm = (snr_for_film - self._snr_center_db) / max(1e-6, self._snr_half_range_db)
                snr_emb = self.snr_embed(snr_norm)  # (B, snr_embed_dim)
                gamma = self.film_gamma(snr_emb)  # (B, D)
                beta = self.film_beta(snr_emb)  # (B, D)
                feat = gamma * feat + beta  # FiLM modulation
        elif self.noise_cond:
            eta_for_film = None
            if snr_mode == "known" and snr_flat is not None:
                # Oracle eta from labels (upper-bound mode).
                eta_for_film = snr_db_to_eta(
                    snr_flat,
                    eta_min=self.noise_eta_min,
                    eta_max=self.noise_eta_max,
                )
            elif snr_mode == "predict" and eta_pred is not None:
                eta_for_film = eta_pred.detach()

            if eta_for_film is not None and self.noise_embed is not None:
                eta_for_film = torch.clamp(
                    eta_for_film.float(),
                    min=self.noise_eta_min,
                    max=self.noise_eta_max,
                ).view(-1, 1)
                eta_norm = (eta_for_film - self._noise_eta_center) / max(1e-6, self._noise_eta_half_range)
                eta_emb = self.noise_embed(eta_norm)
                gamma = self.noise_film_gamma(eta_emb)
                beta = self.noise_film_beta(eta_emb)
                feat = gamma * feat + beta

        # Append global cyclostationary stats to feature vector
        if self.use_expert and self.cyclo_stats is not None:
            global_stats = self.cyclo_stats(x_flat[:, :2, :])  # (B, 7)
            if self._expert_gate_flat is not None:
                global_stats = global_stats * self._expert_gate_flat.view(-1, 1)
            feat = torch.cat([feat, global_stats], dim=1)  # (B, D + 7)

        return feat, snr_pred

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # unused, for API compatibility
        snr: Optional[torch.Tensor] = None,  # used for SNR conditioning if snr_mode == "known"
        snr_mode: str = "predict",
        group_mask: Optional[torch.Tensor] = None,
        cls_to_denoiser_scale: float = 1.0,
        denoiser_bypass: bool = False,
        moe_gate_tau_scale: float = 1.0,
        moe_use_oracle_gate: bool = False,
        moe_oracle_snr: Optional[torch.Tensor] = None,
        moe_head_ce_detach_trunk: bool = False,
    ):
        # Accept (B,2,L) or (B,K,2,L)
        group_size = None
        if x.ndim == 4:
            group_size = x.shape[1]
            x_flat = x.reshape(-1, x.shape[2], x.shape[3])
        else:
            x_flat = x

        # Prepare SNR for conditioning
        snr_flat = None
        if (self.snr_cond or self.noise_cond or self.denoiser_enabled or self.dn_diff_enabled) and snr is not None:
            if group_size is not None:
                snr_flat = snr.repeat_interleave(group_size)
            else:
                snr_flat = snr

        # Optional denoiser preprocessor path (A2/A3/A4).
        x_for_clf = x_flat
        eta_pred_external = None
        self._x_dn_flat = None
        self._dn_diff_active_mask = None
        self._dn_diff_t_start = None
        if self.dn_diff_enabled:
            x_raw = x_flat[:, :2, :]
            if bool(self.force_denoiser_bypass or denoiser_bypass):
                x_dn = x_raw
                eta_pred_dn = None
                if self.noise_fraction_net is not None:
                    eta_pred_dn, _aux = self.noise_fraction_net(x_raw)
                eta_cond = eta_pred_dn.detach() if eta_pred_dn is not None else None
            elif self.training:
                x_dn = x_raw
                eta_pred_dn = None
                if self.noise_fraction_net is not None:
                    eta_pred_dn, _aux = self.noise_fraction_net(x_raw)
                eta_cond = eta_pred_dn.detach() if eta_pred_dn is not None else None
            else:
                x_dn, eta_pred_dn, eta_cond = self._dn_diff_eval_reconstruct(
                    x_raw=x_raw,
                    snr_flat=snr_flat,
                    snr_mode=snr_mode,
                )
            self._x_dn_flat = x_dn
            eta_pred_external = eta_pred_dn
            if self.denoiser_dual_path:
                x_raw_for_clf = self._maybe_low_snr_raw_dropout(
                    x_raw,
                    eta_cond=eta_cond,
                    snr_flat=snr_flat,
                )
                x_for_clf = torch.cat([x_raw_for_clf, x_dn], dim=1)
            else:
                x_for_clf = x_dn
        elif self.denoiser_enabled:
            x_raw = x_flat[:, :2, :]
            x_dn, eta_pred_dn, _eta_cond = self.denoise_only(x_raw, snr=snr_flat, snr_mode=snr_mode)
            if bool(self.force_denoiser_bypass or denoiser_bypass):
                x_dn = x_raw
            self._x_dn_flat = x_dn
            eta_pred_external = eta_pred_dn
            x_dn_for_clf = self._scale_grad(x_dn, cls_to_denoiser_scale)
            if self.denoiser_dual_path:
                x_raw_for_clf = self._maybe_low_snr_raw_dropout(
                    x_raw,
                    eta_cond=_eta_cond,
                    snr_flat=snr_flat,
                )
                x_for_clf = torch.cat([x_raw_for_clf, x_dn_for_clf], dim=1)
            else:
                x_for_clf = x_dn_for_clf

        feat_flat, snr_pred_flat = self._forward_windows(
            x_for_clf,
            snr_flat=snr_flat,
            snr_mode=snr_mode,
            eta_pred_external=eta_pred_external,
        )  # (B*,D_cls), (B*,)

        if group_size is not None and group_size > 1:
            b = feat_flat.shape[0] // group_size
            feat_g = feat_flat.view(b, group_size, -1)  # (B,G,D)
            snr_g = snr_pred_flat.view(b, group_size)  # (B,G)
            if group_mask is None:
                mask = torch.ones((b, group_size), device=feat_g.device, dtype=torch.float32)
            else:
                mask = group_mask.to(device=feat_g.device, dtype=torch.float32)
            denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)  # (B,1)
            feat = torch.sum(feat_g * mask.unsqueeze(-1), dim=1) / denom
            snr_pred = torch.sum(snr_g * mask, dim=1) / denom.squeeze(-1)
            if self._eta_pred_flat is not None:
                eta_g = self._eta_pred_flat.view(b, group_size)
                self._eta_pred = torch.sum(eta_g * mask, dim=1) / denom.squeeze(-1)
            else:
                self._eta_pred = None
            if self._expert_gate_flat is not None:
                eg_g = self._expert_gate_flat.view(b, group_size)
                self._expert_gate = torch.sum(eg_g * mask, dim=1) / denom.squeeze(-1)
            else:
                self._expert_gate = None
            if self._x_dn_flat is not None:
                xdn_g = self._x_dn_flat.view(
                    b, group_size, self._x_dn_flat.shape[1], self._x_dn_flat.shape[2]
                )
                self._x_dn = (
                    torch.sum(xdn_g * mask.unsqueeze(-1).unsqueeze(-1), dim=1)
                    / denom.unsqueeze(-1)
                )
            else:
                self._x_dn = None
        else:
            feat = feat_flat
            snr_pred = snr_pred_flat
            self._eta_pred = self._eta_pred_flat
            self._expert_gate = self._expert_gate_flat
            self._x_dn = self._x_dn_flat

        # Classifier head
        self._moe_gate = None
        self._moe_expert_load = None
        self._moe_logits_experts = None
        self._moe_logits_experts_aux = None
        if self.moe_head is not None:
            eta_router = None
            if bool(moe_use_oracle_gate):
                snr_for_gate = moe_oracle_snr if moe_oracle_snr is not None else snr
                eta_router = self._eta_from_snr(snr_for_gate)
            if eta_router is None:
                eta_router = self._eta_pred
            if eta_router is None and snr is not None and snr_mode == "known":
                eta_router = self._eta_from_snr(snr)
            if eta_router is None:
                eta_router = snr_db_to_eta(
                    snr_pred.detach(),
                    eta_min=self.noise_eta_min,
                    eta_max=self.noise_eta_max,
                )
            logits, gate, logits_e = self.moe_head(
                feat,
                eta_router,
                tau_scale=float(moe_gate_tau_scale),
            )
            self._moe_gate = gate
            self._moe_expert_load = gate.mean(dim=0)
            self._moe_logits_experts = logits_e
            if bool(moe_head_ce_detach_trunk):
                self._moe_logits_experts_aux = self.moe_head.expert_logits(feat.detach())
        else:
            z = self.fc_act(self.fc1(feat))
            z = self.drop(z)
            z = self.fc_act(self.fc2(z))
            z = self.drop(z)
            logits = self.fc_out(z)

        # Return (logits, x0_pred_dummy, snr_pred) to match evaluate() expectations.
        return logits, None, snr_pred

    def forward_features(
        self,
        x: torch.Tensor,
        snr: Optional[torch.Tensor] = None,
        snr_mode: str = "predict",
    ) -> torch.Tensor:
        """
        Forward pass that returns only the pooled features (before classifier head).
        Used for contrastive pre-training.

        Args:
            x: (B, 2, L) or (B, K, 2, L)
            snr: optional SNR values for conditioning
            snr_mode: "known" or "predict"

        Returns:
            feat: (B, D) or (B*K, D) if not pooling K windows
        """
        # Flatten if grouped
        if x.ndim == 4:
            x_flat = x.reshape(-1, x.shape[2], x.shape[3])
            group_size = x.shape[1]
        else:
            x_flat = x
            group_size = None

        # Prepare SNR for conditioning
        snr_flat = None
        if (self.snr_cond or self.noise_cond or self.denoiser_enabled or self.dn_diff_enabled) and snr is not None:
            if group_size is not None:
                snr_flat = snr.repeat_interleave(group_size)
            else:
                snr_flat = snr

        x_for_clf = x_flat
        eta_pred_external = None
        if self.denoiser_enabled:
            x_raw = x_flat[:, :2, :]
            x_dn, eta_pred_dn, _eta_cond = self.denoise_only(x_raw, snr=snr_flat, snr_mode=snr_mode)
            self._x_dn_flat = x_dn
            eta_pred_external = eta_pred_dn
            if self.denoiser_dual_path:
                x_for_clf = torch.cat([x_raw, x_dn], dim=1)
            else:
                x_for_clf = x_dn

        feat_flat, _ = self._forward_windows(
            x_for_clf,
            snr_flat=snr_flat,
            snr_mode=snr_mode,
            eta_pred_external=eta_pred_external,
        )

        # For contrastive learning, return per-window features (don't pool)
        return feat_flat


# =========================================================================
# Multi-View CLDNN-AMC: STFT Branch + Cross-View Attention Fusion
# =========================================================================

class STFTBranch(nn.Module):
    """
    Spectral feature extractor using Short-Time Fourier Transform.

    Input:  (B, 2, L) raw IQ
    Output: (B, out_channels, T_stft)

    Computes the STFT of the complex baseband signal, takes the log-magnitude
    spectrogram, and processes it with a Conv2D stack that progressively
    collapses the frequency dimension while preserving temporal resolution.
    """

    def __init__(
        self,
        n_fft: int = 64,
        hop_length: int = 8,
        out_channels: int = 64,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

        # Conv2D stack: reduce frequency, preserve time
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, stride=(2, 1), padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # collapse remaining freq bins
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2, L) raw I/Q signal
        Returns:
            h: (B, out_channels, T_stft) spectral temporal features
        """
        # STFT in float32 for FFT precision
        x_f32 = x.float()
        xc = torch.complex(x_f32[:, 0, :], x_f32[:, 1, :])  # (B, L)
        X = torch.stft(
            xc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )  # (B, n_freq, T_stft)
        # Log-magnitude spectrogram (good dynamic range, stable numerics)
        mag = torch.log1p(X.abs()).unsqueeze(1)  # (B, 1, n_freq, T_stft)
        # Conv2D processing (progressively reduce freq dimension)
        h = self.drop(self.act(self.bn1(self.conv1(mag))))   # (B, 32, F, T)
        h = self.drop(self.act(self.bn2(self.conv2(h))))     # (B, 64, F/2, T)
        h = self.drop(self.act(self.bn3(self.conv3(h))))     # (B, C, F/4, T)
        h = self.freq_pool(h).squeeze(2)                     # (B, C, T)
        return h


class CrossViewFusion(nn.Module):
    """
    Cross-view attention fusion with SNR-conditioned per-feature gating.

    The primary view (IQ) attends to the secondary view (STFT) via
    multi-head cross-attention.  An SNR-conditioned gate modulates the
    cross-attention output per feature dimension, learning to:
      - Trust raw IQ at high SNR (gate → 0, ignore STFT)
      - Trust spectral features at low SNR (gate → 1, pull in STFT info)

    This is the core novelty: view-dependent, SNR-adaptive feature fusion.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        snr_gate: bool = True,
        snr_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        # Post-attention feed-forward (pre-norm residual)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

        # SNR-conditioned per-feature gate
        self.use_snr_gate = snr_gate
        if snr_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(1, snr_embed_dim),
                nn.GELU(),
                nn.Linear(snr_embed_dim, dim),
                nn.Sigmoid(),
            )
            # Initialize output to ~0 so sigmoid → 0.5 (balanced initial gate)
            nn.init.zeros_(self.gate_net[-2].weight)
            nn.init.zeros_(self.gate_net[-2].bias)

    def forward(
        self,
        feat_primary: torch.Tensor,
        feat_secondary: torch.Tensor,
        snr_norm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            feat_primary:   (B, T, D) — primary view features (IQ)
            feat_secondary: (B, T', D) — secondary view features (STFT)
            snr_norm:       (B, 1) — normalized SNR in [-1, 1], or None
        Returns:
            fused: (B, T, D) — fused features at primary temporal resolution
        """
        q = self.norm_q(feat_primary)
        kv = self.norm_kv(feat_secondary)
        attn_out, _ = self.cross_attn(q, kv, kv)

        # SNR-conditioned gating
        if self.use_snr_gate and snr_norm is not None:
            gate = self.gate_net(snr_norm).unsqueeze(1)  # (B, 1, D)
            attn_out = gate * attn_out

        # Residual connections
        fused = feat_primary + attn_out
        fused = fused + self.ff(fused)
        return fused


class MultiViewCLDNNAMC(nn.Module):
    """
    Multi-View CNN+LSTM for Automatic Modulation Classification.

    Extends CLDNNAMC with an STFT spectral branch and cross-view attention
    fusion.  The IQ-CNN branch is identical to CLDNNAMC.  The STFT branch
    extracts frequency-domain features that complement the time-domain IQ
    features, especially at low SNR where spectral shape is more robust.

    Architecture:
      IQ-CNN branch  ─────→ temporal reduce → (B, T, D) ─┐
                                                          ├─ CrossViewFusion → BiLSTM → pool → classifier
      STFT-CNN branch → project to D → (B, T', D) ───────┘

    Forward signature matches CLDNNAMC / DiffusionAMC for evaluate() compatibility.
    """

    def __init__(
        self,
        num_classes: int = 24,
        seq_len: int = 1024,
        conv_channels: int = 128,
        merge_channels: int = 256,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        pool: str = "attn",
        snr_cond: bool = True,
        snr_embed_dim: int = 64,
        snr_loss_detach_backbone: bool = False,
        snr_min_db: float = -20.0,
        snr_max_db: float = 30.0,
        cls_hidden: int = 0,
        # STFT branch
        stft_nfft: int = 64,
        stft_hop: int = 8,
        stft_channels: int = 64,
        # Cross-view fusion
        cross_view_heads: int = 4,
        snr_gate: bool = True,
    ) -> None:
        super().__init__()
        if pool not in {"attn", "last", "mean"}:
            raise ValueError(f"pool must be one of attn|last|mean, got {pool}")

        self.seq_len = int(seq_len)
        if float(snr_min_db) >= float(snr_max_db):
            raise ValueError(f"snr_min_db must be < snr_max_db (got {snr_min_db} vs {snr_max_db})")
        self.snr_min_db = float(snr_min_db)
        self.snr_max_db = float(snr_max_db)
        self._snr_center_db = 0.5 * (self.snr_min_db + self.snr_max_db)
        self._snr_half_range_db = 0.5 * (self.snr_max_db - self.snr_min_db)
        self.time_out = max(1, self.seq_len - 4)
        self.pool = pool
        self.snr_cond = bool(snr_cond)
        self.snr_loss_detach_backbone = bool(snr_loss_detach_backbone)

        # ========== IQ Branch (identical to CLDNNAMC) ==========
        self.conv_iq = nn.Conv2d(1, conv_channels, kernel_size=(2, 8), padding="same")
        self.conv_i = CausalConv1d(1, conv_channels, kernel_size=8)
        self.conv_q = CausalConv1d(1, conv_channels, kernel_size=8)
        self.conv_fuse = nn.Conv2d(
            conv_channels, conv_channels, kernel_size=(1, 8), padding="same",
        )
        self.conv_merge = nn.Conv2d(
            2 * conv_channels, merge_channels, kernel_size=(2, 5), padding=0,
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(float(dropout))

        # ========== STFT Branch ==========
        self.stft_branch = STFTBranch(
            n_fft=stft_nfft,
            hop_length=stft_hop,
            out_channels=stft_channels,
            dropout=float(dropout),
        )
        self.stft_proj = nn.Sequential(
            nn.Linear(stft_channels, merge_channels),
            nn.LayerNorm(merge_channels),
        )

        # ========== Temporal Reduction (IQ branch) ==========
        self._target_lstm_len = 128
        reduce_layers: list = []
        t_len = self.time_out
        reduce_in = merge_channels
        while t_len > self._target_lstm_len * 1.5:
            out_ch = merge_channels
            k = 5 if t_len > 256 else 3
            pad = k // 2
            reduce_layers.append(
                nn.Conv1d(reduce_in, out_ch, kernel_size=k, stride=2, padding=pad)
            )
            reduce_layers.append(nn.BatchNorm1d(out_ch))
            reduce_layers.append(nn.ReLU())
            reduce_layers.append(nn.Dropout(float(dropout)))
            reduce_in = out_ch
            t_len = (t_len + 2 * pad - k) // 2 + 1

        self.temporal_reduce = nn.Sequential(*reduce_layers) if reduce_layers else None
        self._iq_time_len = t_len

        # ========== Cross-View Attention Fusion ==========
        self.cross_view_fusion = CrossViewFusion(
            dim=merge_channels,
            num_heads=cross_view_heads,
            dropout=float(dropout) * 0.5,
            snr_gate=snr_gate,
        )

        # Early SNR head (for gate conditioning in predict mode)
        self.snr_head_early = nn.Linear(merge_channels, 1)

        # ========== BiLSTM ==========
        self.lstm = nn.LSTM(
            input_size=merge_channels,
            hidden_size=lstm_hidden,
            num_layers=int(lstm_layers),
            batch_first=True,
            bidirectional=bool(bidirectional),
            dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.lstm_out_dim = lstm_out_dim

        # ========== Attention Pooling ==========
        self.attn_pool = TemporalAttentionPool(
            lstm_out_dim, hidden=max(64, lstm_out_dim // 2)
        )

        # ========== SNR Head + FiLM ==========
        self.snr_head = nn.Linear(lstm_out_dim, 1)

        if self.snr_cond:
            self.snr_embed = nn.Sequential(
                nn.Linear(1, snr_embed_dim),
                nn.GELU(),
                nn.Linear(snr_embed_dim, snr_embed_dim),
                nn.GELU(),
            )
            self.film_gamma = nn.Linear(snr_embed_dim, lstm_out_dim)
            self.film_beta = nn.Linear(snr_embed_dim, lstm_out_dim)
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.ones_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
        else:
            self.snr_embed = None
            self.film_gamma = None
            self.film_beta = None

        # ========== Classifier Head ==========
        if cls_hidden > 0:
            _cls_h = int(cls_hidden)
        else:
            _cls_h = max(128, num_classes * 8)
        self.fc1 = nn.Linear(lstm_out_dim, _cls_h)
        self.fc2 = nn.Linear(_cls_h, _cls_h)
        self.fc_out = nn.Linear(_cls_h, num_classes)
        self.fc_act = nn.SELU()

    def _normalize_snr(self, snr_db: torch.Tensor) -> torch.Tensor:
        """Normalize SNR to [-1, 1] using dataset min/max."""
        snr_db = torch.clamp(snr_db, min=self.snr_min_db, max=self.snr_max_db)
        return (snr_db - self._snr_center_db) / max(1e-6, self._snr_half_range_db)

    def _forward_windows(
        self,
        x_flat: torch.Tensor,
        snr_flat: Optional[torch.Tensor] = None,
        snr_mode: str = "none",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core forward pass for single windows (no group handling).

        Args:
            x_flat: (B*, 2, L)
            snr_flat: (B*,) optional SNR values
            snr_mode: "known" | "predict" | "none"
        Returns:
            feat: (B*, D_cls)
            snr_pred: (B*,)
        """
        # ===== IQ Branch =====
        x_iq = x_flat.unsqueeze(1)  # (B,1,2,L)
        h_iq = self.drop(self.act(self.conv_iq(x_iq)))  # (B,conv,2,L)

        i = x_flat[:, 0:1, :]  # (B,1,L)
        q = x_flat[:, 1:2, :]  # (B,1,L)
        h_i = self.drop(self.act(self.conv_i(i))).unsqueeze(2)   # (B,conv,1,L)
        h_q = self.drop(self.act(self.conv_q(q))).unsqueeze(2)   # (B,conv,1,L)
        h_iq_sep = torch.cat([h_i, h_q], dim=2)                  # (B,conv,2,L)
        h_sep = self.drop(self.act(self.conv_fuse(h_iq_sep)))     # (B,conv,2,L)

        h = torch.cat([h_iq, h_sep], dim=1)                      # (B,2*conv,2,L)
        h = self.drop(self.act(self.conv_merge(h)))               # (B,merge,1,L-4)
        h = h.squeeze(2)  # (B, merge_ch, L-4)

        # Temporal reduction (strided convolutions)
        if self.temporal_reduce is not None:
            h = self.temporal_reduce(h)  # (B, merge_ch, T_reduced)

        h_iq_feat = h.transpose(1, 2)  # (B, T, merge_ch)

        # ===== STFT Branch =====
        h_stft = self.stft_branch(x_flat)  # (B, stft_ch, T_stft)
        h_stft_feat = self.stft_proj(h_stft.transpose(1, 2))  # (B, T_stft, merge_ch)

        # ===== SNR for gating =====
        iq_pooled = h_iq_feat.mean(dim=1)  # (B, merge_ch)
        snr_pred_early = self.snr_head_early(iq_pooled).squeeze(-1)  # (B,)

        if snr_mode == "known" and snr_flat is not None:
            snr_for_gate = snr_flat.view(-1, 1).float()
        elif snr_mode == "predict":
            snr_for_gate = snr_pred_early.view(-1, 1)  # end-to-end gradient
        else:
            snr_for_gate = None

        snr_norm = self._normalize_snr(snr_for_gate) if snr_for_gate is not None else None

        # ===== Cross-View Fusion =====
        h_fused = self.cross_view_fusion(h_iq_feat, h_stft_feat, snr_norm=snr_norm)

        # ===== BiLSTM =====
        lstm_out, _ = self.lstm(h_fused)

        # ===== Attention Pooling =====
        if self.pool == "attn":
            feat, _w = self.attn_pool(lstm_out)
        elif self.pool == "mean":
            feat = lstm_out.mean(dim=1)
        else:  # last
            feat = lstm_out[:, -1, :]

        # ===== SNR Prediction (main head, from final features) =====
        feat_for_snr = feat.detach() if self.snr_loss_detach_backbone else feat
        snr_pred = self.snr_head(feat_for_snr).squeeze(-1)

        # ===== FiLM Conditioning =====
        if self.snr_cond:
            if snr_mode == "known" and snr_flat is not None:
                snr_for_film = snr_flat.view(-1, 1).float()
            elif snr_mode == "predict":
                snr_for_film = snr_pred.detach().view(-1, 1)
            else:
                snr_for_film = None

            if snr_for_film is not None:
                snr_norm_film = self._normalize_snr(snr_for_film)
                snr_emb = self.snr_embed(snr_norm_film)
                gamma = self.film_gamma(snr_emb)
                beta = self.film_beta(snr_emb)
                feat = gamma * feat + beta

        return feat, snr_pred

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # unused, API compatibility
        snr: Optional[torch.Tensor] = None,
        snr_mode: str = "predict",
        group_mask: Optional[torch.Tensor] = None,
    ):
        # Accept (B,2,L) or (B,K,2,L)
        group_size = None
        if x.ndim == 4:
            group_size = x.shape[1]
            x_flat = x.reshape(-1, x.shape[2], x.shape[3])
        else:
            x_flat = x

        # Always pass SNR if available (needed for gate + FiLM)
        snr_flat = None
        if snr is not None:
            if group_size is not None:
                snr_flat = snr.repeat_interleave(group_size)
            else:
                snr_flat = snr

        feat_flat, snr_pred_flat = self._forward_windows(
            x_flat, snr_flat=snr_flat, snr_mode=snr_mode,
        )

        if group_size is not None and group_size > 1:
            b = feat_flat.shape[0] // group_size
            feat_g = feat_flat.view(b, group_size, -1)
            snr_g = snr_pred_flat.view(b, group_size)
            if group_mask is None:
                mask = torch.ones((b, group_size), device=feat_g.device, dtype=torch.float32)
            else:
                mask = group_mask.to(device=feat_g.device, dtype=torch.float32)
            denom = torch.clamp(mask.sum(dim=1, keepdim=True), min=1.0)
            feat = torch.sum(feat_g * mask.unsqueeze(-1), dim=1) / denom
            snr_pred = torch.sum(snr_g * mask, dim=1) / denom.squeeze(-1)
        else:
            feat = feat_flat
            snr_pred = snr_pred_flat

        # Classifier head
        z = self.fc_act(self.fc1(feat))
        z = self.drop(z)
        z = self.fc_act(self.fc2(z))
        z = self.drop(z)
        logits = self.fc_out(z)

        return logits, None, snr_pred

    def forward_features(
        self,
        x: torch.Tensor,
        snr: Optional[torch.Tensor] = None,
        snr_mode: str = "predict",
    ) -> torch.Tensor:
        """Return pooled features before classifier (for contrastive pretraining)."""
        if x.ndim == 4:
            x_flat = x.reshape(-1, x.shape[2], x.shape[3])
            group_size = x.shape[1]
        else:
            x_flat = x
            group_size = None

        snr_flat = None
        if snr is not None:
            if group_size is not None:
                snr_flat = snr.repeat_interleave(group_size)
            else:
                snr_flat = snr

        feat_flat, _ = self._forward_windows(x_flat, snr_flat=snr_flat, snr_mode=snr_mode)
        return feat_flat
