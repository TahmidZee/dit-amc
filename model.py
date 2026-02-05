import math
from typing import Optional, Tuple

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


class CLDNNAMC(nn.Module):
    """
    K=1-first AMC model inspired by MCLDNN / CLDNN family (CNN + LSTM), implemented in PyTorch.

    Design goals:
      - Strong inductive bias for short IQ sequences (L=128)
      - Small parameter count (<< DiT-AMC) to reduce overfitting
      - Optional attention pooling over LSTM outputs (better than "last hidden" at low SNR)

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
        dropout: float = 0.5,
        pool: str = "attn",  # attn|last|mean
    ) -> None:
        super().__init__()
        if seq_len != 128:
            # This architecture is hard-coded for RML2016.10a window length.
            raise ValueError("CLDNNAMC currently supports seq_len=128 only.")
        if pool not in {"attn", "last", "mean"}:
            raise ValueError(f"pool must be one of attn|last|mean, got {pool}")
        self.seq_len = int(seq_len)
        self.pool = pool

        # Branch 1: IQ joint Conv2D over (2 x 128)
        self.conv_iq = nn.Conv2d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=(2, 8),
            padding="same",
        )

        # Branch 2/3: I-only and Q-only causal Conv1D
        self.conv_i = CausalConv1d(1, conv_channels, kernel_size=8)
        self.conv_q = CausalConv1d(1, conv_channels, kernel_size=8)

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

        # LSTM over time axis (T=124 after conv_merge)
        self.lstm = nn.LSTM(
            input_size=merge_channels,
            hidden_size=lstm_hidden,
            num_layers=int(lstm_layers),
            batch_first=True,
            bidirectional=bool(bidirectional),
            dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        self.attn_pool = TemporalAttentionPool(lstm_out_dim, hidden=max(64, lstm_out_dim // 2))

        # Classifier head (mirrors MCLDNN-style DNN)
        self.fc1 = nn.Linear(lstm_out_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_out = nn.Linear(128, num_classes)
        self.fc_act = nn.SELU()

        # Optional SNR head for auxiliary supervision (kept for compatibility with train.py)
        self.snr_head = nn.Linear(lstm_out_dim, 1)

    def _forward_windows(self, x_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_flat: (B*, 2, 128)
        returns:
          feat: (B*, D)
          snr_pred: (B*,)
        """
        b = x_flat.shape[0]
        # IQ joint branch
        x_iq = x_flat.unsqueeze(1)  # (B,1,2,128)
        h_iq = self.drop(self.act(self.conv_iq(x_iq)))  # (B,conv,2,128)

        # I/Q separate branches (causal Conv1D)
        i = x_flat[:, 0:1, :]  # (B,1,128)
        q = x_flat[:, 1:2, :]  # (B,1,128)
        h_i = self.drop(self.act(self.conv_i(i)))  # (B,conv,128)
        h_q = self.drop(self.act(self.conv_q(q)))  # (B,conv,128)
        h_i = h_i.unsqueeze(2)  # (B,conv,1,128)
        h_q = h_q.unsqueeze(2)  # (B,conv,1,128)
        h_iq_sep = torch.cat([h_i, h_q], dim=2)  # (B,conv,2,128)
        h_sep = self.drop(self.act(self.conv_fuse(h_iq_sep)))  # (B,conv,2,128)

        # Merge the two streams along channels
        h = torch.cat([h_iq, h_sep], dim=1)  # (B,2*conv,2,128)
        h = self.drop(self.act(self.conv_merge(h)))  # (B,merge,1,124)

        # Sequence for LSTM
        h = h.squeeze(2).transpose(1, 2)  # (B,124,merge)

        lstm_out, _ = self.lstm(h)  # (B,124,D)
        if self.pool == "attn":
            feat, _w = self.attn_pool(lstm_out)
        elif self.pool == "mean":
            feat = lstm_out.mean(dim=1)
        else:  # last
            feat = lstm_out[:, -1, :]

        snr_pred = self.snr_head(feat).squeeze(-1)
        return feat, snr_pred

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # unused, for API compatibility
        snr: Optional[torch.Tensor] = None,  # unused unless snr_mode == "known" (we don't condition yet)
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

        feat_flat, snr_pred_flat = self._forward_windows(x_flat)  # (B*,D), (B*,)

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
        else:
            feat = feat_flat
            snr_pred = snr_pred_flat

        # Classifier head
        z = self.fc_act(self.fc1(feat))
        z = self.drop(z)
        z = self.fc_act(self.fc2(z))
        z = self.drop(z)
        logits = self.fc_out(z)

        # Return (logits, x0_pred_dummy, snr_pred) to match evaluate() expectations.
        return logits, None, snr_pred
