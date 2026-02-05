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
        # --- New architecture options for improved K=1 ---
        use_cls_token: bool = False,  # Use learnable [CLS] token instead of mean pooling
        cls_head_hidden: int = 0,     # Hidden dim for MLP classifier (0 = single linear)
        cls_head_layers: int = 1,     # Number of layers in classifier head
        use_lstm: bool = False,       # Add LSTM after Transformer for temporal modeling
        lstm_hidden: int = 128,       # LSTM hidden size (bidirectional, so 2x this)
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
        self.use_cls_token = use_cls_token
        self.use_lstm = use_lstm

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
        
        # Positional embedding: +1 for CLS token if used
        num_pos = self.num_patches + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_pos, dim))
        
        # Learnable [CLS] token for classification
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

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
        
        # Optional LSTM for temporal sequence modeling (like MCLDNN)
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=dim,
                hidden_size=lstm_hidden,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )
            self.lstm_norm = nn.LayerNorm(lstm_hidden * 2)
            cls_input_dim = lstm_hidden * 2  # bidirectional
        else:
            self.lstm = None
            cls_input_dim = dim
        
        # Deeper MLP classification head (vs single linear layer)
        if cls_head_hidden > 0 and cls_head_layers > 1:
            cls_layers = []
            in_dim = cls_input_dim
            for i in range(cls_head_layers - 1):
                out_dim = cls_head_hidden if i < cls_head_layers - 2 else cls_head_hidden
                cls_layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
                in_dim = out_dim
            cls_layers.append(nn.Linear(in_dim, num_classes))
            self.cls_head = nn.Sequential(*cls_layers)
        else:
            # Single linear (original behavior)
            self.cls_head = nn.Linear(cls_input_dim, num_classes)

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
        
        B = tokens_in.shape[0]
        
        # Prepend [CLS] token if using learned pooling
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            tokens = torch.cat([cls_tokens, tokens_in], dim=1)  # (B, 1+N, D)
        else:
            tokens = tokens_in
        
        tokens = tokens + self.pos_embed

        pooled0 = tokens[:, 1:, :].mean(dim=1) if self.use_cls_token else tokens.mean(dim=1)
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
        
        # Extract patch tokens (excluding CLS if present) for x0 prediction
        if self.use_cls_token:
            patch_tokens = tokens[:, 1:, :]  # (B, N, D) - exclude CLS
            cls_out = tokens[:, 0, :]        # (B, D) - CLS token output
        else:
            patch_tokens = tokens
            cls_out = None
        
        x0_pred = self.x0_head(patch_tokens)

        # Compute pooled representation for classification
        if self.use_cls_token:
            # Use CLS token as the pooled representation
            pooled = cls_out  # (B*, D)
        else:
            pooled = x0_pred.mean(dim=1)  # (B*, D)
        
        # Optional LSTM for temporal modeling (process patch sequence)
        if self.lstm is not None:
            # Apply LSTM to patch tokens for richer temporal features
            lstm_out, _ = self.lstm(patch_tokens)  # (B, N, 2*lstm_hidden)
            lstm_pooled = lstm_out.mean(dim=1)     # (B, 2*lstm_hidden)
            lstm_pooled = self.lstm_norm(lstm_pooled)
            pooled = lstm_pooled  # Override with LSTM features
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
