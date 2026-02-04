import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    ) -> None:
        super().__init__()
        if seq_len % patch_size != 0:
            raise ValueError("seq_len must be divisible by patch_size.")
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.dim = dim
        self.snr_scale = snr_scale

        if stem_channels and stem_channels > 0:
            layers = []
            ch_in = in_channels
            for _ in range(int(stem_layers)):
                layers.append(nn.Conv1d(ch_in, stem_channels, kernel_size=3, padding=1))
                layers.append(nn.GELU())
                ch_in = stem_channels
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

    def forward_tokens(
        self,
        tokens_in: torch.Tensor,
        t: torch.Tensor,
        snr: Optional[torch.Tensor] = None,
        snr_mode: str = "predict",
        group_size: Optional[int] = None,
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
            pooled = pooled.view(b, group_size, -1).mean(dim=1)
            snr_pred_out = snr_pred.view(b, group_size).mean(dim=1)
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
        return self.forward_tokens(tokens0, t, snr=snr, snr_mode=snr_mode, group_size=group_size)
