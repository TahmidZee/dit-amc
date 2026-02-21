import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

try:
    from torch.amp import GradScaler as AmpGradScaler
    _GRADSCALER_USES_DEVICE = True
except ImportError:  # pragma: no cover - older torch
    from torch.cuda.amp import GradScaler as AmpGradScaler
    _GRADSCALER_USES_DEVICE = False

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    def tqdm(iterable, **_kwargs):
        return iterable

from data import (
    build_tensors,
    filter_indices_by_snrs,
    load_rml2018a_hdf5,
    load_rml2016a,
    parse_snrs,
    RML2016aDataset,
    RML2016aGroupedDataset,
    RML2016aVariableGroupedDataset,
)
from diffusion import DiffusionSchedule
from model import CLDNNAMC, DiffusionAMC, MultiViewCLDNNAMC


PRESETS = {
    "S": {"patch_size": 8, "dim": 192, "depth": 10, "heads": 6, "lr": 2e-4},
    "B": {"patch_size": 4, "dim": 256, "depth": 12, "heads": 8, "lr": 1e-4},
}


# =============================================================================
# SNR-path consistency: calibrated noise injection + KL consistency loss
# =============================================================================
def snr_path_degrade(
    x: torch.Tensor,
    snr_db: torch.Tensor,
    delta_min: Union[float, torch.Tensor] = 2.0,
    delta_max: Union[float, torch.Tensor] = 8.0,
    snr_floor: float = -20.0,
    snr_target_min: Optional[float] = None,
    snr_target_max: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a harder view of each sample by adding calibrated AWGN to reduce
    the effective SNR by a random amount Δ ∈ [delta_min, delta_max] dB.

    The noise power is computed analytically so that:
        SNR_new = SNR_original − Δ

    Args:
        x: (B, 2, L) or (B, K, 2, L) — IQ signal (assumed RMS-normalized)
        snr_db: (B,) — per-sample SNR labels in dB
        delta_min: minimum SNR degradation in dB
        delta_max: maximum SNR degradation in dB
        snr_floor: don't degrade below this SNR (dataset minimum)

    Returns:
        x_low: degraded signal (same shape as x)
        delta_actual: (B,) actual degradation applied (may be clipped by floor)
    """
    device = x.device
    B = snr_db.shape[0]

    snr_db_f = snr_db.float()
    if isinstance(delta_min, torch.Tensor):
        dmin = delta_min.to(device=device, dtype=snr_db_f.dtype).view(-1)
    else:
        dmin = torch.full((B,), float(delta_min), device=device, dtype=snr_db_f.dtype)
    if isinstance(delta_max, torch.Tensor):
        dmax = delta_max.to(device=device, dtype=snr_db_f.dtype).view(-1)
    else:
        dmax = torch.full((B,), float(delta_max), device=device, dtype=snr_db_f.dtype)
    if dmin.numel() != B or dmax.numel() != B:
        raise ValueError(f"delta_min/delta_max must broadcast to batch size {B}.")
    dmax = torch.maximum(dmax, dmin)

    # Sample random Δ per sample
    delta = dmin + (dmax - dmin) * torch.rand((B,), device=device, dtype=snr_db_f.dtype)

    # Optional targeted sampling: sample new SNR directly from an allowed range,
    # then infer delta.  Falls back to delta-based degradation when infeasible.
    if snr_target_min is not None or snr_target_max is not None:
        tmin = float(snr_target_min) if snr_target_min is not None else -1e9
        tmax = float(snr_target_max) if snr_target_max is not None else 1e9
        low = torch.maximum(snr_db_f - dmax, torch.full_like(snr_db_f, max(float(snr_floor), tmin)))
        high = torch.minimum(snr_db_f - dmin, torch.full_like(snr_db_f, tmax))
        span = torch.clamp(high - low, min=0.0)
        snr_new_target = low + span * torch.rand((B,), device=device, dtype=snr_db_f.dtype)
        snr_new_fallback = torch.clamp(snr_db_f - delta, min=float(snr_floor))
        feasible = high > low
        snr_new = torch.where(feasible, snr_new_target, snr_new_fallback)
    else:
        # Clamp so we don't go below the dataset's minimum SNR
        snr_new = torch.clamp(snr_db_f - delta, min=float(snr_floor))

    delta_actual = snr_db.float() - snr_new  # may be less than requested if clamped

    # Convert SNR from dB to linear scale
    # SNR_linear = 10^(SNR_dB / 10)
    snr_orig_lin = 10.0 ** (snr_db_f / 10.0)
    snr_new_lin = 10.0 ** (snr_new / 10.0)

    # ---------------------------------------------------------------
    # CRITICAL: x is already noisy (x = signal + existing_noise).
    # What we measure is P_total = P_signal + P_noise_existing.
    # We must disentangle P_signal using the known SNR label:
    #   P_total = P_signal × (1 + 1/SNR_orig_lin)
    #   P_signal = P_total / (1 + 1/SNR_orig_lin)
    # ---------------------------------------------------------------
    p_total = x.view(B, -1).pow(2).mean(dim=1)  # (B,) — total power

    # Derive pure signal power from total power and label SNR
    p_signal = p_total / (1.0 + 1.0 / (snr_orig_lin + 1e-8))  # (B,)

    # Additional noise variance needed:
    # We want P_noise_new  = P_signal / SNR_new_lin
    # We have P_noise_orig = P_signal / SNR_orig_lin
    # Added:  σ²_add = P_signal × (1/SNR_new_lin − 1/SNR_orig_lin)
    noise_var = p_signal * (1.0 / (snr_new_lin + 1e-8) - 1.0 / (snr_orig_lin + 1e-8))
    noise_var = torch.clamp(noise_var, min=0.0)  # safety: never subtract noise
    noise_std = torch.sqrt(noise_var + 1e-12)

    # Reshape for broadcasting
    if x.ndim == 4:
        noise_std = noise_std.view(B, 1, 1, 1)
    else:
        noise_std = noise_std.view(B, 1, 1)

    # Add calibrated Gaussian noise
    noise = torch.randn_like(x) * noise_std
    x_low = x + noise

    return x_low, delta_actual


def snr_db_to_eta_target(
    snr_db: torch.Tensor,
    rho_min: float = 1e-4,
    rho_max: float = 1.0 - 1e-4,
    eta_min: float = -8.0,
    eta_max: float = 5.5,
) -> torch.Tensor:
    """
    Convert SNR(dB) labels to eta=logit(rho) targets, where rho=P_noise/P_total.
    """
    snr_lin = torch.pow(10.0, snr_db.float() / 10.0)
    rho = 1.0 / (1.0 + snr_lin)
    rho = torch.clamp(rho, min=float(rho_min), max=float(rho_max))
    eta = torch.log(rho) - torch.log1p(-rho)
    eta = torch.clamp(eta, min=float(eta_min), max=float(eta_max))
    return eta


def high_snr_soft_mask(snr_db: torch.Tensor, center_db: float = 10.0, width_db: float = 2.0) -> torch.Tensor:
    return torch.sigmoid((snr_db.float() - float(center_db)) / max(1e-6, float(width_db)))


def get_cls2dn_scale(epoch: int, args: argparse.Namespace) -> float:
    if not bool(getattr(args, "cldnn_denoiser", False)):
        return 1.0
    stage_a = int(max(0, getattr(args, "stage_a_epochs", 0)))
    stage_b = int(max(0, getattr(args, "stage_b_epochs", 0)))
    if epoch < stage_a:
        return 0.0
    if stage_b <= 0:
        return 1.0
    in_b = epoch - stage_a
    if in_b >= stage_b:
        return 1.0
    b_half = max(1, stage_b // 2)
    if in_b < b_half:
        return float(getattr(args, "stage_b1_cls2dn_scale", 0.0))
    return float(getattr(args, "stage_b2_cls2dn_scale", 0.1))


def get_lambda_feat(epoch: int, args: argparse.Namespace) -> float:
    """
    Ramp feature-preservation weight after Stage-A (or from epoch 0 if Stage-A disabled).
    """
    target = float(getattr(args, "lambda_feat", 0.0))
    if target <= 0.0:
        return 0.0
    start_ep = int(max(0, getattr(args, "stage_a_epochs", 0)))
    if epoch < start_ep:
        return 0.0
    ramp_epochs = int(max(0, getattr(args, "feat_ramp_epochs", 0)))
    if ramp_epochs <= 0:
        return target
    progress = (epoch - start_ep + 1) / float(max(1, ramp_epochs))
    progress = min(1.0, max(0.0, progress))
    return target * progress


def get_kd_warmup_epoch(args: argparse.Namespace) -> int:
    """
    Effective KD warmup epoch, with optional stage-aware clamping.
    """
    warmup = int(max(0, getattr(args, "kd_warmup", 0)))
    if bool(getattr(args, "kd_warmup_after_stages", False)):
        stage_total = int(max(0, getattr(args, "stage_a_epochs", 0))) + int(max(0, getattr(args, "stage_b_epochs", 0)))
        post_delay = int(max(0, getattr(args, "kd_post_stage_delay", 0)))
        warmup = max(warmup, stage_total + post_delay)
    return warmup


def get_lambda_kd(
    epoch: int,
    args: argparse.Namespace,
    teacher_enabled: bool,
    target_attr: str = "lambda_kd",
) -> float:
    """
    Optional warmup/ramp for external KD weight.
    """
    target = float(getattr(args, target_attr, 0.0))
    if target <= 0.0 or not teacher_enabled:
        return 0.0
    warmup = get_kd_warmup_epoch(args)
    if epoch < warmup:
        return 0.0
    ramp_epochs = int(max(0, getattr(args, "kd_ramp", 0)))
    if ramp_epochs <= 0:
        return target
    progress = (epoch - warmup + 1) / float(max(1, ramp_epochs))
    progress = min(1.0, max(0.0, progress))
    return target * progress


def kd_distillation_loss(
    logits_student: torch.Tensor,
    logits_teacher: torch.Tensor,
    temperature: float = 2.0,
    snr_db: Optional[torch.Tensor] = None,
    snr_lo: float = -999.0,
    snr_hi: float = 999.0,
    conf_thresh: float = 0.0,
    labels: Optional[torch.Tensor] = None,
    correctness_filter: bool = False,
    normalize_by_active: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    KL distillation loss from teacher logits to student logits, with optional
    confidence, correctness, and SNR gating.

    By default this uses batch-mean scaling (`mean` over all samples after
    weighting), which avoids over-amplifying per-sample KD gradients when the
    KD mask is sparse (e.g., low-band-only KD).
    """
    if logits_student.shape != logits_teacher.shape:
        raise ValueError(
            f"KD logits shape mismatch: student={tuple(logits_student.shape)} teacher={tuple(logits_teacher.shape)}"
        )
    t = max(1e-6, float(temperature))
    p_teacher = F.softmax(logits_teacher.detach() / t, dim=1)
    log_p_student = F.log_softmax(logits_student / t, dim=1)
    kl_per_sample = F.kl_div(log_p_student, p_teacher, reduction="none").sum(dim=1)

    weight = torch.ones(logits_student.shape[0], device=logits_student.device, dtype=logits_student.dtype)
    if conf_thresh > 0.0:
        conf, _ = p_teacher.max(dim=1)
        weight = weight * (conf >= float(conf_thresh)).float()
    if snr_db is not None:
        snr_f = snr_db.float()
        weight = weight * ((snr_f >= float(snr_lo)) & (snr_f <= float(snr_hi))).float()
    if correctness_filter and labels is not None:
        labels_f = labels.reshape(-1).to(device=logits_teacher.device, dtype=torch.long)
        if labels_f.shape[0] == logits_teacher.shape[0]:
            teacher_pred = logits_teacher.detach().argmax(dim=1)
            weight = weight * (teacher_pred == labels_f).float()

    if normalize_by_active:
        denom = torch.clamp(weight.sum(), min=1.0)
        loss = (kl_per_sample * weight).sum() / denom * (t ** 2)
    else:
        loss = (kl_per_sample * weight).mean() * (t ** 2)
    active_frac = weight.mean()
    return loss, active_frac


def fit_noise_proxy_calibration(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    rho_min: float,
    rho_max: float,
    eta_min: float,
    eta_max: float,
    max_batches: int = 256,
) -> Dict[str, float]:
    """
    Fit a monotonic linear calibration eta0 = a * e_proxy + b for the analytic proxy.
    """
    if not hasattr(model, "noise_fraction_net") or getattr(model, "noise_fraction_net") is None:
        return {}

    e_all: List[torch.Tensor] = []
    eta_all: List[torch.Tensor] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for bidx, batch in enumerate(loader):
            if len(batch) == 4:
                x, _y, snr, _mask = batch
            else:
                x, _y, snr = batch[:3]
            x = x.to(device)
            snr = snr.to(device)
            if x.ndim == 4:
                g = x.shape[1]
                x_flat = x.reshape(-1, x.shape[2], x.shape[3])
                snr_flat = snr.repeat_interleave(g)
            else:
                x_flat = x
                snr_flat = snr
            e_proxy = model.noise_fraction_net.proxy.compute_proxy_energy(x_flat[:, :2, :])  # type: ignore[attr-defined]
            eta_tgt = snr_db_to_eta_target(
                snr_flat,
                rho_min=float(rho_min),
                rho_max=float(rho_max),
                eta_min=float(eta_min),
                eta_max=float(eta_max),
            )
            e_all.append(e_proxy.detach().float().cpu())
            eta_all.append(eta_tgt.detach().float().cpu())
            if (bidx + 1) >= int(max_batches):
                break
    if was_training:
        model.train()

    if not e_all:
        return {}

    e = torch.cat(e_all, dim=0).numpy()
    y = torch.cat(eta_all, dim=0).numpy()
    xmat = np.stack([e, np.ones_like(e)], axis=1)
    sol, *_ = np.linalg.lstsq(xmat, y, rcond=None)
    scale = float(max(sol[0], 1e-6))
    bias = float(sol[1])
    model.set_noise_proxy_calibration(scale=scale, bias=bias)  # type: ignore[attr-defined]
    return {"proxy_cal_scale": scale, "proxy_cal_bias": bias}


def snr_consistency_loss(
    logits_clean: torch.Tensor,
    logits_noisy: torch.Tensor,
    temperature: float = 2.0,
    snr_db: Optional[torch.Tensor] = None,
    snr_lo: float = -6.0,
    snr_hi: float = 6.0,
    conf_thresh: float = 0.0,
    snr_new_db: Optional[torch.Tensor] = None,
    snr_new_lo: float = -999.0,
    snr_new_hi: float = 999.0,
) -> torch.Tensor:
    """
    KL-divergence consistency loss between clean-view and noisy-view predictions.

    The clean view's softmax is treated as the target (stop-gradient).
    Temperature > 1 softens the distributions for better gradient flow.

    Optional improvements over naive KL:
      - **Confidence weighting**: only distill from teacher when its max-prob
        exceeds conf_thresh (prevents garbage-teaching at low SNR).
      - **SNR gating**: only apply loss to samples in the transition zone
        [snr_lo, snr_hi] dB where the consistency signal is most useful.
        (Outside: high-SNR is trivially consistent; very-low-SNR is hopeless.)
      - **Student SNR gating** (optional): additionally gate based on the
        degraded-view SNR (snr_new_db) to focus robustness where you want it.

    Args:
        logits_clean: (B, C) logits from the original (higher-SNR) view
        logits_noisy: (B, C) logits from the degraded (lower-SNR) view
        temperature: softmax temperature (higher = softer targets)
        snr_db: (B,) per-sample SNR in dB (optional; enables SNR gating)
        snr_lo: lower SNR bound for gating (dB)
        snr_hi: upper SNR bound for gating (dB)
        conf_thresh: minimum teacher confidence to apply loss (0 = disabled)
        snr_new_db: (B,) degraded-view SNR in dB (optional; enables student SNR gating)
        snr_new_lo: lower bound for degraded-view SNR gating (dB)
        snr_new_hi: upper bound for degraded-view SNR gating (dB)

    Returns:
        loss: scalar weighted KL divergence
    """
    B = logits_clean.shape[0]

    # Soft targets from clean view (stop gradient)
    p_clean = F.softmax(logits_clean.detach() / temperature, dim=1)

    # Log-softmax from noisy view
    log_p_noisy = F.log_softmax(logits_noisy / temperature, dim=1)

    # Per-sample KL divergence
    kl_per_sample = F.kl_div(log_p_noisy, p_clean, reduction="none").sum(dim=1)  # (B,)

    # Build per-sample weight mask
    weight = torch.ones(B, device=logits_clean.device)

    # Confidence gating: only distill when teacher is confident
    if conf_thresh > 0:
        teacher_conf, _ = p_clean.max(dim=1)
        weight = weight * (teacher_conf >= conf_thresh).float()

    # SNR gating: only apply in transition zone
    if snr_db is not None:
        snr_f = snr_db.float()
        in_zone = (snr_f >= float(snr_lo)) & (snr_f <= float(snr_hi))
        weight = weight * in_zone.float()

    # Student SNR gating: optionally gate based on degraded-view SNR
    if snr_new_db is not None:
        snr_new_f = snr_new_db.float()
        in_zone_new = (snr_new_f >= float(snr_new_lo)) & (snr_new_f <= float(snr_new_hi))
        weight = weight * in_zone_new.float()

    # Weighted mean, scaled by T² to match CE gradient magnitude
    denom = torch.clamp(weight.sum(), min=1.0)
    kl = (kl_per_sample * weight).sum() / denom * (temperature ** 2)

    return kl


# =============================================================================
# Focal loss
# =============================================================================
def focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Focal loss: FL(p_t) = -(1 - p_t)^gamma * CE(p, y)

    Down-weights easy (high confidence) samples, focuses on hard samples.
    When gamma=0, this is standard cross-entropy.
    """
    ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=label_smoothing)
    if gamma == 0.0:
        return ce if reduction == "none" else ce.mean()
    # p_t = probability of correct class
    p_t = torch.exp(-ce)
    focal_weight = (1.0 - p_t) ** gamma
    loss = focal_weight * ce
    if reduction == "mean":
        return loss.mean()
    return loss


# =============================================================================
# Mixup augmentation helper
# =============================================================================
def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    snr: torch.Tensor,
    alpha: float = 0.4,
    snr_min: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply mixup augmentation.
    Returns: (mixed_x, y_a, y_b, lam, mixed_snr)
    """
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    lam_t = torch.full((batch_size,), lam, device=x.device, dtype=torch.float32)
    if snr_min is not None:
        snr_f = snr.float()
        # Only mix pairs where BOTH samples are above the SNR threshold.
        eligible = (snr_f >= float(snr_min)) & (snr_f[index] >= float(snr_min))
        lam_t = torch.where(eligible, lam_t, torch.ones_like(lam_t))

    lam_x = lam_t.view(batch_size, *([1] * (x.ndim - 1))).to(dtype=x.dtype)
    mixed_x = lam_x * x + (1.0 - lam_x) * x[index]
    y_a, y_b = y, y[index]
    snr_f = snr.float()
    mixed_snr = lam_t * snr_f + (1.0 - lam_t) * snr_f[index]

    return mixed_x, y_a, y_b, lam_t, mixed_snr


def mixup_criterion(
    criterion_fn,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixup loss as weighted combination of losses for both labels."""
    return lam * criterion_fn(logits, y_a) + (1 - lam) * criterion_fn(logits, y_b)


# =============================================================================
# Curriculum learning helper
# =============================================================================
def get_curriculum_snr_min(epoch: int, curriculum_epochs: int, snr_start: float, snr_end: float = -20.0) -> float:
    """
    Compute the minimum SNR for curriculum learning.
    Linearly decreases from snr_start to snr_end over curriculum_epochs.
    """
    if curriculum_epochs <= 0:
        return snr_end
    progress = min(1.0, float(epoch) / float(curriculum_epochs))
    return snr_start + progress * (snr_end - snr_start)


def curriculum_weights(
    snr: torch.Tensor,
    curriculum_snr_min: float,
    epoch: int,
    curriculum_epochs: int,
    soft: bool,
    soft_low_weight: float,
) -> torch.Tensor:
    """
    Per-sample curriculum weights (float32).
    - Hard curriculum: weights are {0,1} based on snr >= curriculum_snr_min.
    - Soft curriculum: below-threshold samples get a small weight that ramps to 1.0 by curriculum_epochs.
    """
    if curriculum_epochs <= 0:
        return torch.ones_like(snr, dtype=torch.float32)
    snr_f = snr.float()
    if not soft:
        return (snr_f >= float(curriculum_snr_min)).float()
    progress = min(1.0, float(epoch) / float(curriculum_epochs))
    w0 = float(soft_low_weight)
    w_low = w0 + progress * (1.0 - w0)
    ones = torch.ones_like(snr_f, dtype=torch.float32)
    low = torch.full_like(snr_f, w_low, dtype=torch.float32)
    return torch.where(snr_f >= float(curriculum_snr_min), ones, low)


# =============================================================================
# Consistency loss helper
# =============================================================================
def consistency_loss(logits_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute consistency loss as the mean variance of predictions across K windows.
    logits_list: list of K tensors, each (B, C)
    Returns: scalar loss encouraging all K predictions to be the same.
    """
    if len(logits_list) <= 1:
        return torch.tensor(0.0, device=logits_list[0].device)
    
    # Stack: (K, B, C)
    stacked = torch.stack(logits_list, dim=0)
    # Compute softmax probs
    probs = F.softmax(stacked, dim=-1)  # (K, B, C)
    # Mean across K
    mean_probs = probs.mean(dim=0, keepdim=True)  # (1, B, C)
    # KL divergence from each to mean
    kl_divs = F.kl_div(
        probs.log(),
        mean_probs.expand_as(probs),
        reduction="none",
    ).sum(dim=-1)  # (K, B)
    return kl_divs.mean()


# =============================================================================
# Contrastive learning (InfoNCE / NT-Xent loss)
# =============================================================================
def info_nce_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Compute InfoNCE (NT-Xent) loss for contrastive learning.

    Args:
        features: (B*K, D) - features from K windows per sample, flattened
        labels: (B,) - class labels for each sample (repeated K times internally)
        temperature: softmax temperature (lower = sharper)

    Returns:
        loss: scalar contrastive loss

    Windows from the same sample (same label and same batch position) are positive pairs.
    Windows from different samples are negative pairs.
    """
    device = features.device
    batch_size = labels.shape[0]
    k = features.shape[0] // batch_size  # number of windows per sample

    # Normalize features (important for cosine similarity)
    features = F.normalize(features, dim=1)

    # Compute similarity matrix: (B*K, B*K)
    sim_matrix = torch.mm(features, features.T) / temperature

    # Create mask for positive pairs (same sample = same batch index)
    # sample_ids: [0,0,0,0, 1,1,1,1, 2,2,2,2, ...] if K=4
    sample_ids = torch.arange(batch_size, device=device).repeat_interleave(k)
    pos_mask = (sample_ids.unsqueeze(0) == sample_ids.unsqueeze(1)).float()

    # Remove self-similarity from positive mask
    eye = torch.eye(features.shape[0], device=device)
    pos_mask = pos_mask - eye

    # For each row, we want: log(sum(exp(pos)) / sum(exp(all except self)))
    # Mask out self-similarity with large negative
    sim_matrix = sim_matrix - eye * 1e9

    # Compute log-softmax over all pairs (except self)
    log_softmax = F.log_softmax(sim_matrix, dim=1)

    # Average log-prob of positive pairs
    # Each sample has (K-1) positive pairs (other windows from same sample)
    num_positives = pos_mask.sum(dim=1).clamp(min=1)
    loss = -(log_softmax * pos_mask).sum(dim=1) / num_positives

    return loss.mean()


# =============================================================================
# Supervised Contrastive Loss  (Khosla et al., NeurIPS 2020)
# =============================================================================
def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised contrastive (SupCon) loss using *in-batch* positives.

    For each anchor i the set of positives is
        P(i) = { j ≠ i : y_j == y_i }
    and the loss is

        L_i = -(1/|P(i)|) Σ_{p∈P(i)} log  exp(z_i·z_p/τ)
                                            ────────────────────
                                            Σ_{a≠i} exp(z_i·z_a/τ)

    Anchors with no positives in the batch are excluded from the mean.

    Args:
        z:      (B, D) — **already L2-normalised** embeddings
        labels: (B,)   — integer class labels
        temperature: τ  (lower → sharper; 0.07 is standard for normalised embeddings)

    Returns:
        scalar loss
    """
    device = z.device
    B = z.shape[0]
    if B <= 1:
        return torch.tensor(0.0, device=device)

    # ------- similarity matrix (B, B) -------
    sim = torch.mm(z, z.T) / temperature          # (B, B)

    # numerical stability: subtract row-wise max before exp
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # ------- masks -------
    self_mask = torch.eye(B, dtype=torch.bool, device=device)          # True on diagonal
    pos_mask  = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~self_mask  # (B, B)

    # ------- denominator: log Σ_{a≠i} exp(sim_ia) -------
    exp_sim = torch.exp(sim)
    exp_sim = exp_sim.masked_fill(self_mask, 0.0)          # zero diagonal
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)   # (B, 1)

    # ------- per-pair log-prob -------
    log_prob = sim - log_denom                             # (B, B)

    # ------- mean over positives for each anchor -------
    num_pos = pos_mask.float().sum(dim=1)                  # (B,)
    has_pos = num_pos > 0                                  # bool (B,)

    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / (num_pos + 1e-8)

    # loss = - mean positive log-prob, averaged over anchors that have positives
    loss = -mean_log_prob_pos * has_pos.float()
    n_valid = has_pos.float().sum().clamp(min=1.0)
    return loss.sum() / n_valid


def _cuda_amp_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, _minor = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


@dataclass
class EMA:
    decay: float
    shadow: Dict[str, torch.Tensor]
    backup: Dict[str, torch.Tensor]

    @classmethod
    def create(cls, model: torch.nn.Module, decay: float) -> "EMA":
        shadow: Dict[str, torch.Tensor] = {}
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                shadow[k] = v.detach().clone()
        return cls(decay=float(decay), shadow=shadow, backup={})

    def to(self, device: torch.device) -> None:
        self.shadow = {k: v.to(device) for k, v in self.shadow.items()}
        self.backup = {k: v.to(device) for k, v in self.backup.items()}

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            msd = model.state_dict()
            d = self.decay
            for k, v in msd.items():
                if k in self.shadow:
                    self.shadow[k].mul_(d).add_(v.detach(), alpha=1.0 - d)

    def store(self, model: torch.nn.Module) -> None:
        self.backup = {}
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.backup[k] = v.detach().clone()

    def copy_to(self, model: torch.nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.shadow.items():
            msd[k].copy_(v)

    def restore(self, model: torch.nn.Module) -> None:
        if not self.backup:
            return
        msd = model.state_dict()
        for k, v in self.backup.items():
            msd[k].copy_(v)
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.detach().clone() for k, v in sd.items()}

def parse_args() -> argparse.Namespace:
    import os
    # Auto-detect default data path: check current dir first, then fallback to absolute
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_rml2016 = None
    _default_rml2018 = None
    # Check for RML2016 in current directory
    if os.path.exists(os.path.join(_script_dir, "RML2016.10a_dict.pkl")):
        _default_rml2016 = os.path.join(_script_dir, "RML2016.10a_dict.pkl")
    elif os.path.exists("/home/tahit/Modulation/RML2016.10a_dict.pkl"):
        _default_rml2016 = "/home/tahit/Modulation/RML2016.10a_dict.pkl"
    else:
        _default_rml2016 = os.path.join(_script_dir, "RML2016.10a_dict.pkl")  # Will error if missing, user must specify
    # Check for RML2018
    if os.path.exists("/home/tahit/Modulation/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5"):
        _default_rml2018 = "/home/tahit/Modulation/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5"
    else:
        _default_rml2018 = "/home/tahit/Modulation/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5"  # Will error if missing, user must specify
    
    parser = argparse.ArgumentParser(description="Diffusion-regularized AMC (RML2016.10a / RML2018.01A)")
    parser.add_argument(
        "--data-path",
        type=str,
        default=_default_rml2016,
        help=f"Path to dataset file. Default: {_default_rml2016} (auto-detected). For RML2018, use --data-path with --dataset rml2018a.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["rml2016a", "rml2018a"],
        default="rml2016a",
        help="Dataset format. rml2016a=pickle dict (RML2016.10a). rml2018a=HDF5 with X/Y/Z (RadioML 2018.01A).",
    )
    parser.add_argument("--out-dir", type=str, default="./runs/dit_amc")
    parser.add_argument("--preset", type=str, choices=["S", "B"], default="S")
    parser.add_argument(
        "--arch",
        type=str,
        choices=["dit", "cldnn", "multiview"],
        default="dit",
        help="Model architecture. dit=DiffusionAMC. cldnn=CNN+LSTM. multiview=IQ+STFT dual-branch with cross-view attention.",
    )

    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--stem-channels", type=int, default=64)
    parser.add_argument("--stem-layers", type=int, default=2)
    parser.add_argument("--group-k", type=int, default=1, help="Number of random windows per (mod,SNR) bucket per sample.")
    parser.add_argument("--group-pool", type=str, choices=["mean", "attn"], default="mean", help="How to pool over group-k windows.")
    parser.add_argument("--k-max", type=int, default=None, help="Max windows for variable-K mode (defaults to --group-k).")
    parser.add_argument("--k-choices", type=str, default=None, help="Comma-separated K choices for variable-K training (e.g. 4,8,16).")
    parser.add_argument("--window-dropout", type=float, default=0.0, help="Extra random window dropout fraction applied on top of sampled K.")

    # CLDNN architecture options (used when --arch cldnn)
    parser.add_argument("--cldnn-conv-ch", type=int, default=50, help="Conv channels for CLDNN branches.")
    parser.add_argument("--cldnn-merge-ch", type=int, default=100, help="Channels after merging branches (Conv2D(2,5)).")
    parser.add_argument("--cldnn-lstm-hidden", type=int, default=128, help="LSTM hidden size.")
    parser.add_argument("--cldnn-lstm-layers", type=int, default=2, help="Number of LSTM layers.")
    parser.add_argument("--cldnn-bidir", action="store_true", help="Use bidirectional LSTM.")
    parser.add_argument("--cldnn-pool", type=str, default="attn", choices=["attn", "last", "mean"], help="Temporal pooling over LSTM outputs.")
    parser.add_argument("--cldnn-snr-cond", action="store_true", help="Enable SNR conditioning via FiLM for CLDNN.")
    parser.add_argument("--cldnn-noise-cond", action="store_true", help="Enable noise-fraction conditioning via eta=logit(rho) FiLM for CLDNN.")
    parser.add_argument("--cldnn-denoiser", action="store_true", help="Enable residual conditional U-Net denoiser preprocessor.")
    parser.add_argument("--cldnn-denoiser-dual-path", action="store_true", help="Use mandatory dual-path classifier input [x_raw, x_dn] when denoiser is enabled.")
    parser.add_argument(
        "--cldnn-raw-low-snr-drop-prob",
        type=float,
        default=0.0,
        help="Dual-path only: probability of attenuating raw branch for low-SNR samples so classifier must use denoised path.",
    )
    parser.add_argument(
        "--cldnn-raw-low-snr-drop-gate",
        type=str,
        default="auto",
        choices=["auto", "eta", "snr"],
        help="Gate source for low-SNR raw attenuation: auto (eta then SNR fallback), eta-only, or snr-only.",
    )
    parser.add_argument(
        "--cldnn-raw-low-snr-drop-eta-thresh",
        type=float,
        default=1.0,
        help="Apply raw-branch attenuation when eta_cond >= threshold (higher eta = noisier sample).",
    )
    parser.add_argument(
        "--cldnn-raw-low-snr-drop-snr-thresh",
        type=float,
        default=-6.0,
        help="Apply raw-branch attenuation when SNR <= threshold (dB); used directly in gate=snr or as fallback in gate=auto.",
    )
    parser.add_argument(
        "--cldnn-raw-low-snr-drop-min-scale",
        type=float,
        default=0.0,
        help="Minimum multiplicative raw-branch scale for gated samples (0 = hard zero).",
    )
    parser.add_argument(
        "--cldnn-raw-low-snr-drop-max-scale",
        type=float,
        default=0.0,
        help="Maximum multiplicative raw-branch scale for gated samples.",
    )
    parser.add_argument(
        "--cldnn-raw-low-snr-drop-prob-lo",
        type=float,
        default=-1.0,
        help="Optional SNR-shaped schedule: drop probability when SNR <= --cldnn-raw-low-snr-drop-snr-lo (set all lo/mid/hi >=0 to enable).",
    )
    parser.add_argument(
        "--cldnn-raw-low-snr-drop-prob-mid",
        type=float,
        default=-1.0,
        help="Optional SNR-shaped schedule: drop probability for --cldnn-raw-low-snr-drop-snr-lo < SNR <= --cldnn-raw-low-snr-drop-snr-mid.",
    )
    parser.add_argument(
        "--cldnn-raw-low-snr-drop-prob-hi",
        type=float,
        default=-1.0,
        help="Optional SNR-shaped schedule: drop probability when SNR > --cldnn-raw-low-snr-drop-snr-mid.",
    )
    parser.add_argument(
        "--cldnn-raw-low-snr-drop-zero-hi",
        action="store_true",
        help="Force high-SNR raw-drop probability to 0.0 in SNR-shaped schedule (recommended to prevent high-SNR leakage).",
    )
    parser.add_argument(
        "--cldnn-raw-low-snr-drop-snr-lo",
        type=float,
        default=-10.0,
        help="Lower SNR split (dB) for SNR-shaped raw-drop schedule.",
    )
    parser.add_argument(
        "--cldnn-raw-low-snr-drop-snr-mid",
        type=float,
        default=-6.0,
        help="Middle SNR split (dB) for SNR-shaped raw-drop schedule.",
    )
    parser.add_argument("--cldnn-denoiser-base-ch", type=int, default=32, help="Base channels for the denoiser U-Net.")
    parser.add_argument("--cldnn-denoiser-dropout", type=float, default=0.0, help="Dropout inside denoiser residual blocks.")
    parser.add_argument("--cldnn-denoiser-soft-hi-blend", action="store_true", help="Enable soft high-SNR residual suppression in denoiser.")
    parser.add_argument("--cldnn-denoiser-bypass-eval", action="store_true", help="Bypass denoiser at eval/inference (A2b sanity mode).")
    parser.add_argument("--noise-head-hidden", type=int, default=32, help="Hidden channels for lightweight NoiseFractionNet.")
    parser.add_argument("--fit-noise-proxy-calibration", action="store_true", help="Fit analytic proxy calibration eta0=a*e+b on train data before training.")
    parser.add_argument("--noise-proxy-calibration-batches", type=int, default=256, help="Max train batches for fitting analytic proxy calibration.")
    parser.add_argument("--noise-eta-min", type=float, default=-8.0, help="Minimum eta clamp for noise-fraction conditioning/prediction.")
    parser.add_argument("--noise-eta-max", type=float, default=5.5, help="Maximum eta clamp for noise-fraction conditioning/prediction.")
    parser.add_argument("--noise-rho-min", type=float, default=1e-4, help="Minimum rho clamp when converting SNR labels to eta targets.")
    parser.add_argument("--noise-rho-max", type=float, default=1.0 - 1e-4, help="Maximum rho clamp when converting SNR labels to eta targets.")
    parser.add_argument("--cldnn-expert-features", action="store_true", help="Enable expert feature branch (conjugate products, cyclostationary stats).")
    parser.add_argument("--cldnn-expert-ch", type=int, default=64, help="Channels for expert feature CNN.")
    parser.add_argument(
        "--cldnn-expert-stacf-win",
        type=int,
        default=0,
        help="Short-time ACF smoothing window for expert features (0/1 disables, odd windows recommended: 5/9/13).",
    )
    parser.add_argument(
        "--cldnn-expert-v2",
        action="store_true",
        help="Enable expert-v2 map: normalized local correlations (Re/Im/|r_k|) + stable phase-diff (no atan2).",
    )
    parser.add_argument(
        "--cldnn-expert-corr-eps",
        type=float,
        default=1e-6,
        help="Epsilon for normalized local-correlation expert features (v2).",
    )
    parser.add_argument(
        "--cldnn-expert-eta-gate",
        action="store_true",
        help="Gate expert branch by eta (predicted noise level) so expert contribution is low at high SNR and high at low SNR.",
    )
    parser.add_argument("--cldnn-expert-eta-gate-center", type=float, default=0.8, help="eta center for expert gate sigmoid (gate ~0.5 here).")
    parser.add_argument("--cldnn-expert-eta-gate-tau", type=float, default=0.7, help="Temperature/softness for expert eta gate sigmoid.")
    parser.add_argument("--cldnn-expert-eta-gate-min", type=float, default=0.0, help="Minimum expert gate scale.")
    parser.add_argument("--cldnn-expert-eta-gate-max", type=float, default=1.0, help="Maximum expert gate scale.")
    parser.add_argument(
        "--cldnn-cyclo-stats",
        dest="cldnn_cyclo_stats",
        action="store_true",
        help="Append global cyclostationary scalar statistics in expert branch.",
    )
    parser.add_argument(
        "--cldnn-no-cyclo-stats",
        dest="cldnn_cyclo_stats",
        action="store_false",
        help="Disable global cyclostationary scalar statistics (expert-map-only ablation).",
    )
    parser.set_defaults(cldnn_cyclo_stats=True)
    parser.add_argument("--cldnn-cls-hidden", type=int, default=0, help="Classifier head hidden dim (0=auto: max(128, num_classes*8)).")

    # Multi-view architecture options (used when --arch multiview)
    parser.add_argument("--stft-nfft", type=int, default=64, help="STFT FFT size for spectral branch.")
    parser.add_argument("--stft-hop", type=int, default=8, help="STFT hop length for spectral branch.")
    parser.add_argument("--stft-channels", type=int, default=64, help="Output channels of STFT Conv2D stack.")
    parser.add_argument("--cross-view-heads", type=int, default=4, help="Number of heads for cross-view attention fusion.")
    parser.add_argument("--snr-gate", action="store_true", help="Enable SNR-conditioned gating in cross-view fusion.")

    parser.add_argument("--focal-gamma", type=float, default=0.0, help="Focal loss gamma (0=standard CE, 2=strong focal).")

    # Mixup augmentation
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Mixup alpha (0 = disabled). Recommended: 0.2-0.4.")
    parser.add_argument("--mixup-prob", type=float, default=0.5, help="Probability of applying mixup per batch (when alpha > 0).")
    parser.add_argument(
        "--mixup-snr-min",
        type=float,
        default=None,
        help="If set, only mix pairs where BOTH samples have SNR >= this threshold (training-time label only).",
    )
    parser.add_argument(
        "--mixup-cls-only",
        dest="mixup_cls_only",
        action="store_true",
        help="Apply mixup only to classification loss; keep denoiser/noise auxiliary losses on clean (unmixed) inputs.",
    )
    parser.add_argument(
        "--mixup-all-losses",
        dest="mixup_cls_only",
        action="store_false",
        help="Legacy behavior: apply mixup to all losses (including denoiser/noise auxiliary losses).",
    )
    parser.set_defaults(mixup_cls_only=True)

    # Curriculum learning (SNR-based)
    parser.add_argument("--curriculum-epochs", type=int, default=0, help="Number of epochs for curriculum (0 = disabled). SNR min increases from curriculum-snr-start to -20.")
    parser.add_argument("--curriculum-snr-start", type=float, default=0.0, help="Starting SNR min for curriculum (e.g., 0 means start with SNR >= 0 only).")
    parser.add_argument(
        "--curriculum-soft",
        action="store_true",
        help="Soft curriculum: downweight below-threshold samples instead of dropping them (weight ramps to 1.0 by curriculum-epochs).",
    )
    parser.add_argument(
        "--curriculum-soft-low-weight",
        type=float,
        default=0.1,
        help="Below-threshold sample weight at epoch 0 when --curriculum-soft is enabled (ramps to 1.0).",
    )

    # Consistency loss for multi-window training
    parser.add_argument("--consistency-lambda", type=float, default=0.0, help="Weight for consistency loss across K windows (0 = disabled).")
    parser.add_argument("--consistency-k", type=int, default=4, help="Number of windows for consistency loss (used when consistency-lambda > 0).")

    # SNR-path consistency training (core novelty)
    parser.add_argument("--snr-consist", action="store_true", help="Enable SNR-path consistency training (add calibrated noise → enforce prediction consistency).")
    parser.add_argument("--snr-consist-lambda", type=float, default=1.0, help="Weight for SNR-path consistency KL loss.")
    parser.add_argument("--snr-consist-delta-min", type=float, default=2.0, help="Minimum SNR degradation in dB for consistency view.")
    parser.add_argument("--snr-consist-delta-max", type=float, default=8.0, help="Maximum SNR degradation in dB for consistency view.")
    parser.add_argument("--snr-consist-temp", type=float, default=2.0, help="Temperature for consistency soft targets (higher = softer).")
    parser.add_argument("--snr-consist-warmup", type=int, default=5, help="Number of epochs before consistency loss kicks in (let CE stabilize first).")
    parser.add_argument(
        "--snr-consist-adaptive-delta",
        action="store_true",
        help="Use smaller degradation deltas for lower-SNR anchors to keep consistency pairs learnable.",
    )
    parser.add_argument(
        "--snr-consist-low-snr-thresh",
        type=float,
        default=-6.0,
        help="Samples with SNR <= this threshold use low-SNR delta range when --snr-consist-adaptive-delta is enabled.",
    )
    parser.add_argument(
        "--snr-consist-low-delta-min",
        type=float,
        default=2.0,
        help="Minimum consistency degradation delta (dB) for low-SNR samples when adaptive delta is enabled.",
    )
    parser.add_argument(
        "--snr-consist-low-delta-max",
        type=float,
        default=4.0,
        help="Maximum consistency degradation delta (dB) for low-SNR samples when adaptive delta is enabled.",
    )
    parser.add_argument(
        "--snr-consist-ramp",
        type=int,
        default=0,
        help="Linearly ramp consistency weight over this many epochs after warmup (0 = no ramp; immediate full weight).",
    )
    parser.add_argument(
        "--snr-consist-teacher-eval",
        action="store_true",
        help="Use a 'clean-eval' teacher for consistency: recompute clean-view logits with model.eval() + no_grad() (dropout off) to stabilize targets.",
    )
    parser.add_argument("--snr-consist-conf-thresh", type=float, default=0.0, help="Min teacher confidence to apply consistency (0=disabled). E.g. 0.5 filters out uncertain teachers.")
    parser.add_argument("--snr-consist-snr-lo", type=float, default=-999.0, help="Lower SNR bound for consistency gating (dB). Default -999 = no lower bound.")
    parser.add_argument("--snr-consist-snr-hi", type=float, default=999.0, help="Upper SNR bound for consistency gating (dB). Default 999 = no upper bound.")
    parser.add_argument("--snr-consist-snr-new-lo", type=float, default=-999.0, help="Lower bound for degraded-view SNR gating (dB). Default -999 = no lower bound.")
    parser.add_argument("--snr-consist-snr-new-hi", type=float, default=999.0, help="Upper bound for degraded-view SNR gating (dB). Default 999 = no upper bound.")

    # External teacher KD (LUPI-style): oracle teacher at train-time, blind student at inference.
    parser.add_argument("--teacher-ckpt", type=str, default=None, help="Path to external teacher checkpoint for logit distillation.")
    parser.add_argument("--lambda-kd", type=float, default=0.0, help="Weight for external KD KL loss.")
    parser.add_argument("--kd-temp", type=float, default=2.0, help="Temperature for KD soft targets.")
    parser.add_argument("--kd-warmup", type=int, default=0, help="Epochs to wait before KD starts.")
    parser.add_argument(
        "--kd-warmup-after-stages",
        dest="kd_warmup_after_stages",
        action="store_true",
        help="Clamp KD warmup to start after Stage-A/B plus --kd-post-stage-delay.",
    )
    parser.add_argument(
        "--no-kd-warmup-after-stages",
        dest="kd_warmup_after_stages",
        action="store_false",
        help="Disable stage-aware KD warmup clamp; use --kd-warmup as-is.",
    )
    parser.set_defaults(kd_warmup_after_stages=True)
    parser.add_argument(
        "--kd-post-stage-delay",
        type=int,
        default=8,
        help="Extra epochs to wait after Stage-A/B before enabling KD when --kd-warmup-after-stages is enabled.",
    )
    parser.add_argument("--kd-ramp", type=int, default=0, help="Ramp KD weight over this many epochs after warmup (0 = no ramp).")
    parser.add_argument("--kd-conf-thresh", type=float, default=0.0, help="Minimum teacher confidence to apply KD (0 = disabled).")
    parser.add_argument(
        "--kd-correctness-filter",
        dest="kd_correctness_filter",
        action="store_true",
        help="Apply KD only when teacher argmax matches ground-truth label (recommended for noisy low-SNR distillation).",
    )
    parser.add_argument(
        "--no-kd-correctness-filter",
        dest="kd_correctness_filter",
        action="store_false",
        help="Disable teacher correctness filtering for KD.",
    )
    parser.set_defaults(kd_correctness_filter=True)
    parser.add_argument(
        "--kd-normalize-by-active",
        action="store_true",
        help="Legacy KD scaling: normalize by active mask count instead of batch mean.",
    )
    parser.add_argument("--kd-snr-lo", type=float, default=-999.0, help="Lower SNR bound for KD (low-band KD when set, e.g. -14).")
    parser.add_argument("--kd-snr-hi", type=float, default=999.0, help="Upper SNR bound for KD (low-band KD when set, e.g. -6).")
    parser.add_argument(
        "--kd-hi-preserve-scale",
        type=float,
        default=0.0,
        help="Relative high-SNR KD weight. Effective high-band KD weight = lambda_kd * kd_hi_preserve_scale.",
    )
    parser.add_argument(
        "--kd-hi-snr-lo",
        type=float,
        default=10.0,
        help="Lower SNR bound for optional high-band KD preservation mask.",
    )
    parser.add_argument(
        "--kd-hi-snr-hi",
        type=float,
        default=18.0,
        help="Upper SNR bound for optional high-band KD preservation mask.",
    )
    parser.add_argument(
        "--kd-hi-conf-thresh",
        type=float,
        default=-1.0,
        help="Optional confidence threshold for high-band KD. If <0, reuses --kd-conf-thresh.",
    )
    parser.add_argument(
        "--kd-teacher-snr-mode",
        type=str,
        choices=["known", "predict", "none"],
        default="known",
        help="Teacher conditioning mode during KD forward pass. Use 'known' for oracle teacher distillation.",
    )
    parser.add_argument(
        "--kd-disable-mixup",
        action="store_true",
        help="Disable mixup whenever any KD objective is enabled (sanity mode to reduce gradient conflict).",
    )
    parser.add_argument(
        "--lambda-kd-denoise",
        type=float,
        default=0.0,
        help="Weight for teacher->student denoiser-output distillation (L1).",
    )
    parser.add_argument(
        "--kd-denoise-snr-lo",
        type=float,
        default=-14.0,
        help="Lower SNR bound for denoiser KD.",
    )
    parser.add_argument(
        "--kd-denoise-snr-hi",
        type=float,
        default=-6.0,
        help="Upper SNR bound for denoiser KD.",
    )
    parser.add_argument(
        "--lambda-kd-feat",
        type=float,
        default=0.0,
        help="Weight for pre-FiLM pooled feature KD (teacher->student).",
    )
    parser.add_argument(
        "--kd-feat-snr-lo",
        type=float,
        default=-14.0,
        help="Lower SNR bound for pre-FiLM feature KD.",
    )
    parser.add_argument(
        "--kd-feat-snr-hi",
        type=float,
        default=-6.0,
        help="Upper SNR bound for pre-FiLM feature KD.",
    )

    # Contrastive pre-training
    parser.add_argument("--contrastive-pretrain-epochs", type=int, default=0, help="Number of epochs for contrastive pre-training (0 = disabled).")
    parser.add_argument("--contrastive-k", type=int, default=4, help="Number of windows per sample for contrastive learning.")
    parser.add_argument("--contrastive-temp", type=float, default=0.1, help="Temperature for InfoNCE loss.")
    parser.add_argument("--contrastive-lr", type=float, default=None, help="Learning rate for contrastive pre-training (defaults to --lr).")

    # Supervised Contrastive Learning (SupCon)  –  Khosla et al., NeurIPS 2020
    parser.add_argument("--supcon", action="store_true", help="Enable supervised contrastive loss (SupCon) alongside CE.")
    parser.add_argument("--supcon-lambda", type=float, default=0.1, help="Weight for SupCon loss (relative to CE). Start with 0.1.")
    parser.add_argument("--supcon-temp", type=float, default=0.07, help="Temperature τ for cosine similarities (0.07 is standard for L2-normed embeddings).")
    parser.add_argument("--supcon-proj-dim", type=int, default=128, help="Output dimension of the MLP projection head for SupCon.")
    parser.add_argument("--supcon-warmup", type=int, default=0, help="Number of epochs before SupCon loss kicks in (0 = from the start).")

    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--lr-decay-start-epoch",
        type=int,
        default=0,
        help="Keep LR flat (after warmup) until this epoch, then cosine decay to --min-lr. Useful to align LR decay with curriculum (e.g., set to --curriculum-epochs).",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2016)
    parser.add_argument(
        "--train-per",
        type=int,
        default=None,
        help="Samples per (mod,SNR) bucket for training. Default: auto (600 for rml2016a, 3200 for rml2018a).",
    )
    parser.add_argument(
        "--val-per",
        type=int,
        default=None,
        help="Samples per (mod,SNR) bucket for validation. Default: auto (200 for rml2016a, 500 for rml2018a).",
    )
    parser.add_argument("--normalize", type=str, choices=["rms", "none"], default="rms")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--snr-balanced", action="store_true", help="Use SNR-balanced sampling for the training loader.")
    parser.add_argument("--snr-balance-power", type=float, default=1.0, help="Sampling weight exponent: w ~ (1/count)^power.")

    # Train-only, label-preserving signal augmentations (applied after normalization).
    parser.add_argument("--aug-phase", action="store_true", help="Random global phase rotation per window.")
    parser.add_argument("--aug-shift", action="store_true", help="Random circular time shift per window.")
    parser.add_argument("--aug-gain", type=float, default=0.0, help="Random gain jitter magnitude (e.g., 0.2 => x*[0.8,1.2]).")
    parser.add_argument("--aug-cfo", type=float, default=0.0, help="Max normalized CFO in cycles/sample (e.g., 0.01).")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9996)
    parser.add_argument("--ema-start", type=int, default=500)
    parser.add_argument("--ema-every", type=int, default=1)
    parser.add_argument("--train-eval-batches", type=int, default=0)
    parser.add_argument("--phase1-epochs", type=int, default=0)
    parser.add_argument("--phase1-lambda-diff", type=float, default=0.0)
    parser.add_argument("--phase1-p-clean", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--snr-floor-db",
        type=float,
        default=None,
        help="If set, cap diffusion timestep per-sample using labeled SNR so low-SNR samples are not over-noised (training-time only).",
    )
    parser.add_argument(
        "--snr-cap-max-db",
        type=float,
        default=None,
        help="Upper SNR (dB) used for per-sample timestep cap scaling. Auto-detected from dataset if not set.",
    )
    parser.add_argument(
        "--low-snr-boost",
        type=float,
        default=0.0,
        help="Optional multiplier to upweight CE loss for low-SNR samples (training-time only).",
    )

    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--t-schedule", type=str, choices=["uniform", "snr"], default="uniform")
    parser.add_argument(
        "--t-max",
        type=int,
        default=None,
        help="Max diffusion timestep for noise injection (default: full range).",
    )
    parser.add_argument(
        "--p-clean",
        type=float,
        default=0.0,
        help="Probability to use t=0 (no extra noise) during training.",
    )
    parser.add_argument("--lambda-diff", type=float, default=0.2)
    parser.add_argument("--lambda-snr", type=float, default=0.1)
    parser.add_argument("--lambda-noise", type=float, default=0.0, help="Weight for eta(noise-fraction) regression loss.")
    parser.add_argument("--lambda-dn", type=float, default=0.0, help="Weight for paired denoiser reconstruction loss L_dn.")
    parser.add_argument("--lambda-id", type=float, default=0.0, help="Weight for high-SNR identity loss L_id on denoiser output.")
    parser.add_argument("--lambda-feat", type=float, default=0.0, help="Weight for feature-preservation loss L_feat.")
    parser.add_argument(
        "--feat-ramp-epochs",
        type=int,
        default=5,
        help="Linear ramp epochs for lambda-feat after Stage-A start point (0 = no ramp).",
    )
    parser.add_argument(
        "--dn-pair-delta-min",
        type=float,
        default=2.0,
        help="Minimum SNR degradation (dB) for denoiser paired losses (L_dn/L_feat).",
    )
    parser.add_argument(
        "--dn-pair-delta-max",
        type=float,
        default=8.0,
        help="Maximum SNR degradation (dB) for denoiser paired losses (L_dn/L_feat).",
    )
    parser.add_argument(
        "--dn-pair-snr-floor-db",
        type=float,
        default=None,
        help="Minimum degraded SNR (dB) for denoiser paired losses. Default: dataset minimum SNR.",
    )
    parser.add_argument(
        "--dn-pair-snr-new-lo",
        type=float,
        default=-999.0,
        help="Optional lower bound for degraded-view SNR in denoiser paired losses (set > -900 to enable).",
    )
    parser.add_argument(
        "--dn-pair-snr-new-hi",
        type=float,
        default=999.0,
        help="Optional upper bound for degraded-view SNR in denoiser paired losses (set < 900 to enable).",
    )
    parser.add_argument(
        "--lfeat-snr-lo",
        type=float,
        default=-999.0,
        help="Optional source-SNR lower bound for applying L_feat (set > -900 to enable).",
    )
    parser.add_argument(
        "--lfeat-snr-hi",
        type=float,
        default=999.0,
        help="Optional source-SNR upper bound for applying L_feat (set < 900 to enable).",
    )
    parser.add_argument(
        "--lfeat-snr-new-lo",
        type=float,
        default=-999.0,
        help="Optional degraded-SNR lower bound for applying L_feat (set > -900 to enable).",
    )
    parser.add_argument(
        "--lfeat-snr-new-hi",
        type=float,
        default=999.0,
        help="Optional degraded-SNR upper bound for applying L_feat (set < 900 to enable).",
    )
    parser.add_argument(
        "--feat-encoder-ckpt",
        type=str,
        default=None,
        help="Optional checkpoint path used to build a fixed frozen early-feature encoder snapshot.",
    )
    parser.add_argument("--stage-a-epochs", type=int, default=0, help="Stage A epochs (denoiser/noise-head bootstrap).")
    parser.add_argument("--stage-b-epochs", type=int, default=0, help="Stage B epochs (classifier warm start with controlled cls->dn gradients).")
    parser.add_argument("--stage-b1-cls2dn-scale", type=float, default=0.0, help="Classifier->denoiser gradient scale in first half of Stage B.")
    parser.add_argument("--stage-b2-cls2dn-scale", type=float, default=0.1, help="Classifier->denoiser gradient scale in second half of Stage B.")
    parser.add_argument("--stage-a-no-cls", action="store_true", help="Disable classification loss during Stage A.")
    parser.add_argument(
        "--snr-loss-detach-backbone",
        action="store_true",
        help="Head-only SNR supervision (CLDNN): do not backprop SNR loss into the backbone.",
    )
    parser.add_argument("--snr-mode", type=str, choices=["predict", "known", "none"], default="predict")
    parser.add_argument("--snr-scale", type=float, default=20.0)
    parser.add_argument("--t-eval", type=int, default=0)
    parser.add_argument("--report-low-snr-lo", type=float, default=-14.0, help="Lower SNR bound (dB) for reporting low-band macro metrics.")
    parser.add_argument("--report-low-snr-hi", type=float, default=-6.0, help="Upper SNR bound (dB) for reporting low-band macro metrics.")

    parser.add_argument("--train-snrs", type=str, default=None)
    parser.add_argument("--val-snrs", type=str, default=None)
    parser.add_argument("--test-snrs", type=str, default=None)

    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dynamic-k-eval", action="store_true", help="Evaluate with dynamic-K (confidence-adaptive windows) at end of training.")
    parser.add_argument("--dynamic-k-start", type=int, default=4)
    parser.add_argument("--dynamic-k-step", type=int, default=4)
    parser.add_argument("--dynamic-k-max", type=int, default=None, help="Defaults to --k-max (or --group-k).")
    parser.add_argument("--dynamic-conf-thresh", type=float, default=0.85)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA autocast (BF16 on Ampere/Hopper; FP16 otherwise).",
    )
    return parser.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    preset = PRESETS.get(args.preset, {})
    for key, value in preset.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def _cfg_get(
    cfg: Optional[Dict[str, object]],
    fallback: argparse.Namespace,
    key: str,
    default,
):
    if cfg is not None and key in cfg:
        return cfg[key]
    return getattr(fallback, key, default)


def build_model_from_cfg(
    cfg: Optional[Dict[str, object]],
    fallback: argparse.Namespace,
    num_classes: int,
    seq_len: int,
    snr_min_db: float,
    snr_max_db: float,
    device: torch.device,
) -> Tuple[torch.nn.Module, str]:
    arch = str(_cfg_get(cfg, fallback, "arch", "cldnn"))
    if arch == "dit":
        model = DiffusionAMC(
            num_classes=num_classes,
            seq_len=seq_len,
            patch_size=int(_cfg_get(cfg, fallback, "patch_size", 8)),
            dim=int(_cfg_get(cfg, fallback, "dim", 192)),
            depth=int(_cfg_get(cfg, fallback, "depth", 10)),
            heads=int(_cfg_get(cfg, fallback, "heads", 6)),
            mlp_ratio=float(_cfg_get(cfg, fallback, "mlp_ratio", 4.0)),
            dropout=float(_cfg_get(cfg, fallback, "dropout", 0.1)),
            snr_scale=float(_cfg_get(cfg, fallback, "snr_scale", 20.0)),
            stem_channels=int(_cfg_get(cfg, fallback, "stem_channels", 64)),
            stem_layers=int(_cfg_get(cfg, fallback, "stem_layers", 2)),
            group_pool=str(_cfg_get(cfg, fallback, "group_pool", "mean")),
        ).to(device)
    elif arch == "multiview":
        model = MultiViewCLDNNAMC(
            num_classes=num_classes,
            seq_len=seq_len,
            conv_channels=int(_cfg_get(cfg, fallback, "cldnn_conv_ch", 50)),
            merge_channels=int(_cfg_get(cfg, fallback, "cldnn_merge_ch", 100)),
            lstm_hidden=int(_cfg_get(cfg, fallback, "cldnn_lstm_hidden", 128)),
            lstm_layers=int(_cfg_get(cfg, fallback, "cldnn_lstm_layers", 2)),
            bidirectional=bool(_cfg_get(cfg, fallback, "cldnn_bidir", False)),
            dropout=float(_cfg_get(cfg, fallback, "dropout", 0.1)),
            pool=str(_cfg_get(cfg, fallback, "cldnn_pool", "attn")),
            snr_cond=bool(_cfg_get(cfg, fallback, "cldnn_snr_cond", False)),
            snr_loss_detach_backbone=bool(_cfg_get(cfg, fallback, "snr_loss_detach_backbone", False)),
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            cls_hidden=int(_cfg_get(cfg, fallback, "cldnn_cls_hidden", 0)),
            stft_nfft=int(_cfg_get(cfg, fallback, "stft_nfft", 64)),
            stft_hop=int(_cfg_get(cfg, fallback, "stft_hop", 8)),
            stft_channels=int(_cfg_get(cfg, fallback, "stft_channels", 64)),
            cross_view_heads=int(_cfg_get(cfg, fallback, "cross_view_heads", 4)),
            snr_gate=bool(_cfg_get(cfg, fallback, "snr_gate", False)),
        ).to(device)
    else:
        model = CLDNNAMC(
            num_classes=num_classes,
            seq_len=seq_len,
            conv_channels=int(_cfg_get(cfg, fallback, "cldnn_conv_ch", 50)),
            merge_channels=int(_cfg_get(cfg, fallback, "cldnn_merge_ch", 100)),
            lstm_hidden=int(_cfg_get(cfg, fallback, "cldnn_lstm_hidden", 128)),
            lstm_layers=int(_cfg_get(cfg, fallback, "cldnn_lstm_layers", 2)),
            bidirectional=bool(_cfg_get(cfg, fallback, "cldnn_bidir", False)),
            dropout=float(_cfg_get(cfg, fallback, "dropout", 0.1)),
            pool=str(_cfg_get(cfg, fallback, "cldnn_pool", "attn")),
            snr_cond=bool(_cfg_get(cfg, fallback, "cldnn_snr_cond", False)),
            noise_cond=bool(_cfg_get(cfg, fallback, "cldnn_noise_cond", False)),
            snr_loss_detach_backbone=bool(_cfg_get(cfg, fallback, "snr_loss_detach_backbone", False)),
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            noise_eta_min=float(_cfg_get(cfg, fallback, "noise_eta_min", -8.0)),
            noise_eta_max=float(_cfg_get(cfg, fallback, "noise_eta_max", 5.5)),
            denoiser=bool(_cfg_get(cfg, fallback, "cldnn_denoiser", False)),
            denoiser_dual_path=bool(_cfg_get(cfg, fallback, "cldnn_denoiser_dual_path", False)),
            denoiser_base_channels=int(_cfg_get(cfg, fallback, "cldnn_denoiser_base_ch", 32)),
            denoiser_dropout=float(_cfg_get(cfg, fallback, "cldnn_denoiser_dropout", 0.0)),
            denoiser_soft_high_snr_blend=bool(_cfg_get(cfg, fallback, "cldnn_denoiser_soft_hi_blend", False)),
            noise_head_hidden=int(_cfg_get(cfg, fallback, "noise_head_hidden", 32)),
            expert_features=bool(_cfg_get(cfg, fallback, "cldnn_expert_features", False)),
            expert_channels=int(_cfg_get(cfg, fallback, "cldnn_expert_ch", 64)),
            expert_stacf_window=int(_cfg_get(cfg, fallback, "cldnn_expert_stacf_win", 0)),
            expert_v2=bool(_cfg_get(cfg, fallback, "cldnn_expert_v2", False)),
            expert_corr_norm_eps=float(_cfg_get(cfg, fallback, "cldnn_expert_corr_eps", 1e-6)),
            expert_eta_gate=bool(_cfg_get(cfg, fallback, "cldnn_expert_eta_gate", False)),
            expert_eta_gate_center=float(_cfg_get(cfg, fallback, "cldnn_expert_eta_gate_center", 0.8)),
            expert_eta_gate_tau=float(_cfg_get(cfg, fallback, "cldnn_expert_eta_gate_tau", 0.7)),
            expert_eta_gate_min=float(_cfg_get(cfg, fallback, "cldnn_expert_eta_gate_min", 0.0)),
            expert_eta_gate_max=float(_cfg_get(cfg, fallback, "cldnn_expert_eta_gate_max", 1.0)),
            expert_use_cyclo_stats=bool(_cfg_get(cfg, fallback, "cldnn_cyclo_stats", True)),
            raw_low_snr_drop_prob=float(_cfg_get(cfg, fallback, "cldnn_raw_low_snr_drop_prob", 0.0)),
            raw_low_snr_drop_gate=str(_cfg_get(cfg, fallback, "cldnn_raw_low_snr_drop_gate", "auto")),
            raw_low_snr_drop_eta_thresh=float(_cfg_get(cfg, fallback, "cldnn_raw_low_snr_drop_eta_thresh", 1.0)),
            raw_low_snr_drop_snr_thresh=float(_cfg_get(cfg, fallback, "cldnn_raw_low_snr_drop_snr_thresh", -6.0)),
            raw_low_snr_drop_min_scale=float(_cfg_get(cfg, fallback, "cldnn_raw_low_snr_drop_min_scale", 0.0)),
            raw_low_snr_drop_max_scale=float(_cfg_get(cfg, fallback, "cldnn_raw_low_snr_drop_max_scale", 0.0)),
            raw_low_snr_drop_prob_lo=float(_cfg_get(cfg, fallback, "cldnn_raw_low_snr_drop_prob_lo", -1.0)),
            raw_low_snr_drop_prob_mid=float(_cfg_get(cfg, fallback, "cldnn_raw_low_snr_drop_prob_mid", -1.0)),
            raw_low_snr_drop_prob_hi=float(_cfg_get(cfg, fallback, "cldnn_raw_low_snr_drop_prob_hi", -1.0)),
            raw_low_snr_drop_snr_lo=float(_cfg_get(cfg, fallback, "cldnn_raw_low_snr_drop_snr_lo", -10.0)),
            raw_low_snr_drop_snr_mid=float(_cfg_get(cfg, fallback, "cldnn_raw_low_snr_drop_snr_mid", -6.0)),
            cls_hidden=int(_cfg_get(cfg, fallback, "cldnn_cls_hidden", 0)),
            supcon_proj_dim=int(_cfg_get(cfg, fallback, "supcon_proj_dim", 0)) if bool(_cfg_get(cfg, fallback, "supcon", False)) else 0,
        ).to(device)
    return model, arch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(args: argparse.Namespace, device: torch.device):
    if str(getattr(args, "dataset", "rml2016a")) == "rml2018a":
        X, y, snr, mods, snrs, train_idx, val_idx, test_idx = load_rml2018a_hdf5(
            args.data_path,
            seed=args.seed,
            train_per=args.train_per,
            val_per=args.val_per,
        )
    else:
        X, y, snr, mods, snrs, train_idx, val_idx, test_idx = load_rml2016a(
            args.data_path,
            seed=args.seed,
            train_per=args.train_per,
            val_per=args.val_per,
        )
    seq_len = int(X.shape[-1])
    train_snrs = parse_snrs(args.train_snrs)
    val_snrs = parse_snrs(args.val_snrs) if args.val_snrs is not None else train_snrs
    test_snrs = parse_snrs(args.test_snrs)

    train_idx = filter_indices_by_snrs(train_idx, snr, train_snrs)
    val_idx = filter_indices_by_snrs(val_idx, snr, val_snrs)
    test_idx = filter_indices_by_snrs(test_idx, snr, test_snrs)

    X_t, y_t, snr_t = build_tensors(X, y, snr)
    k_max = int(args.k_max) if args.k_max is not None else int(args.group_k)
    k_choices = None
    if args.k_choices is not None and args.k_choices.strip() != "":
        k_choices = [int(v.strip()) for v in args.k_choices.split(",") if v.strip() != ""]

    use_variable_k = k_choices is not None and len(k_choices) > 0 and k_max > 1
    if use_variable_k:
        train_ds = RML2016aVariableGroupedDataset(
            X_t,
            y_t,
            snr_t,
            train_idx,
            k_max=k_max,
            k_choices=k_choices,
            normalize=args.normalize,
            window_dropout=args.window_dropout,
            aug_phase=args.aug_phase,
            aug_shift=args.aug_shift,
            aug_gain=args.aug_gain,
            aug_cfo=args.aug_cfo,
        )
        # For val/test in variable-K mode, we use full k_max windows (mask=all ones).
        val_ds = RML2016aVariableGroupedDataset(
            X_t,
            y_t,
            snr_t,
            val_idx,
            k_max=k_max,
            k_choices=None,
            normalize=args.normalize,
        )
        test_ds = RML2016aVariableGroupedDataset(
            X_t,
            y_t,
            snr_t,
            test_idx,
            k_max=k_max,
            k_choices=None,
            normalize=args.normalize,
        )
    elif args.group_k > 1:
        train_ds = RML2016aGroupedDataset(
            X_t,
            y_t,
            snr_t,
            train_idx,
            group_k=args.group_k,
            normalize=args.normalize,
            aug_phase=args.aug_phase,
            aug_shift=args.aug_shift,
            aug_gain=args.aug_gain,
            aug_cfo=args.aug_cfo,
        )
        val_ds = RML2016aGroupedDataset(X_t, y_t, snr_t, val_idx, group_k=args.group_k, normalize=args.normalize)
        test_ds = RML2016aGroupedDataset(X_t, y_t, snr_t, test_idx, group_k=args.group_k, normalize=args.normalize)
    else:
        train_ds = RML2016aDataset(
            X_t,
            y_t,
            snr_t,
            train_idx,
            normalize=args.normalize,
            aug_phase=args.aug_phase,
            aug_shift=args.aug_shift,
            aug_gain=args.aug_gain,
            aug_cfo=args.aug_cfo,
        )
        val_ds = RML2016aDataset(X_t, y_t, snr_t, val_idx, normalize=args.normalize)
        test_ds = RML2016aDataset(X_t, y_t, snr_t, test_idx, normalize=args.normalize)

    pin_memory = device.type == "cuda"
    num_workers = args.num_workers
    common = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    sampler = None
    if args.snr_balanced:
        # Balance sampling across SNR bins using inverse-frequency weights.
        snr_np = snr_t[torch.from_numpy(train_ds.indices)].cpu().numpy().astype(np.int32)
        unique, counts = np.unique(snr_np, return_counts=True)
        count_map = {int(u): int(c) for u, c in zip(unique, counts)}
        power = float(args.snr_balance_power)
        weights = np.asarray([(1.0 / max(1, count_map[int(s)])) ** power for s in snr_np], dtype=np.float64)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        worker_init_fn=_seed_worker,
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        worker_init_fn=_seed_worker,
        **common,
    )
    return train_loader, val_loader, test_loader, mods, snrs, seq_len


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float,
    decay_start_step: int = 0,
):
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    decay_start_step = max(int(warmup_steps), int(decay_start_step))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        if step < decay_start_step:
            return 1.0
        progress = (step - decay_start_step) / max(1, total_steps - decay_start_step)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_ratio = min_lr / max(1e-12, base_lrs[0])
        return max(min_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    args,
    ema: EMA = None,
):
    ckpt: Dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "args": vars(args),
    }
    if ema is not None:
        ckpt["ema"] = ema.state_dict()
    ckpt["rng"] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, scheduler=None, ema: EMA = None):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
        ema.to(next(model.parameters()).device)
    return ckpt


def load_state_dict_flexible(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], prefix: str = "model") -> Dict[str, int]:
    """
    Load only matching keys/shapes from a state_dict (useful when optional heads differ).
    Returns simple load stats and prints compact diagnostics.
    """
    model_sd = model.state_dict()
    filtered_sd: Dict[str, torch.Tensor] = {}
    skipped_shape = []
    missing_key = []
    for k, v in state_dict.items():
        if k not in model_sd:
            missing_key.append(k)
            continue
        if model_sd[k].shape != v.shape:
            skipped_shape.append(k)
            continue
        filtered_sd[k] = v
    missing_after, unexpected_after = model.load_state_dict(filtered_sd, strict=False)
    if missing_after:
        print(f"[{prefix}] missing keys after flexible load (first 8): {missing_after[:8]}")
    if unexpected_after:
        print(f"[{prefix}] unexpected keys after flexible load (first 8): {unexpected_after[:8]}")
    if skipped_shape:
        print(f"[{prefix}] shape-mismatch keys skipped (first 8): {skipped_shape[:8]}")
    if missing_key:
        print(f"[{prefix}] keys absent in target model (first 8): {missing_key[:8]}")
    return {
        "loaded": int(len(filtered_sd)),
        "skipped_shape": int(len(skipped_shape)),
        "missing_in_target": int(len(missing_key)),
        "missing_after": int(len(missing_after)),
        "unexpected_after": int(len(unexpected_after)),
    }


def write_jsonl(path: str, record: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _macro_metrics_from_confmat(conf_mat: np.ndarray) -> Dict[str, float]:
    """
    Compute macro-accuracy (mean class recall) and macro-F1 from confusion matrix.
    conf_mat is expected in [true_class, predicted_class] layout.
    """
    if conf_mat.size == 0:
        return {"macro_acc": 0.0, "macro_f1": 0.0}
    conf = conf_mat.astype(np.float64, copy=False)
    tp = np.diag(conf)
    support = conf.sum(axis=1)  # true counts per class
    pred_count = conf.sum(axis=0)  # predicted counts per class

    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, pred_count, out=np.zeros_like(tp), where=pred_count > 0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)

    valid = support > 0
    if not np.any(valid):
        return {"macro_acc": 0.0, "macro_f1": 0.0}
    return {
        "macro_acc": float(np.mean(recall[valid])),
        "macro_f1": float(np.mean(f1[valid])),
    }


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    t_eval: int,
    snr_mode: str,
    amp: bool = False,
    low_snr_lo: float = -14.0,
    low_snr_hi: float = -6.0,
) -> Tuple[float, float, Dict[int, float], Dict[str, float]]:
    model.eval()
    total_correct = 0
    total = 0
    snr_correct: Dict[int, int] = {}
    snr_total: Dict[int, int] = {}
    conf_all: Optional[np.ndarray] = None
    conf_low: Optional[np.ndarray] = None

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                x, y, snr, mask = batch
            else:
                x, y, snr = batch
                mask = None
            x = x.to(device)
            y = y.to(device)
            snr = snr.to(device)
            t = torch.full((x.shape[0],), t_eval, device=device, dtype=torch.long)
            snr_in = snr if snr_mode == "known" else None
            if amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=_cuda_amp_dtype(), enabled=True):
                    logits, _, _ = model(x, t, snr=snr_in, snr_mode=snr_mode, group_mask=mask)
            else:
                logits, _, _ = model(x, t, snr=snr_in, snr_mode=snr_mode, group_mask=mask)

            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()
            total_correct += correct
            total += y.shape[0]

            # Class-macro metrics (overall + low-band).
            ncls = int(logits.shape[1])
            if conf_all is None:
                conf_all = np.zeros((ncls, ncls), dtype=np.int64)
            y_cpu_i64 = y.detach().cpu().to(torch.int64)
            preds_cpu_i64 = preds.detach().cpu().to(torch.int64)
            binc = torch.bincount(
                y_cpu_i64 * ncls + preds_cpu_i64,
                minlength=ncls * ncls,
            ).view(ncls, ncls)
            conf_all += binc.numpy()

            snr_cpu_t = snr.detach().cpu().float()
            low_mask_t = (snr_cpu_t >= float(low_snr_lo)) & (snr_cpu_t <= float(low_snr_hi))
            if bool(torch.any(low_mask_t)):
                if conf_low is None:
                    conf_low = np.zeros((ncls, ncls), dtype=np.int64)
                y_low = y_cpu_i64[low_mask_t]
                p_low = preds_cpu_i64[low_mask_t]
                binc_low = torch.bincount(
                    y_low * ncls + p_low,
                    minlength=ncls * ncls,
                ).view(ncls, ncls)
                conf_low += binc_low.numpy()

            snr_cpu = snr.detach().cpu().numpy().astype(np.int32)
            preds_cpu = preds.detach().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()
            for snr_val, pred_val, y_val in zip(snr_cpu, preds_cpu, y_cpu):
                snr_key = int(snr_val)
                snr_correct[snr_key] = snr_correct.get(snr_key, 0) + int(pred_val == y_val)
                snr_total[snr_key] = snr_total.get(snr_key, 0) + 1

    acc = float(total_correct) / max(1, total)
    acc_by_snr = {snr: snr_correct[snr] / snr_total[snr] for snr in snr_total.keys()}
    macro_all = _macro_metrics_from_confmat(conf_all if conf_all is not None else np.zeros((0, 0), dtype=np.int64))
    macro_low = _macro_metrics_from_confmat(conf_low if conf_low is not None else np.zeros((0, 0), dtype=np.int64))
    summary = {
        "macro_acc": float(macro_all.get("macro_acc", 0.0)),
        "macro_f1": float(macro_all.get("macro_f1", 0.0)),
        "low_macro_acc": float(macro_low.get("macro_acc", 0.0)),
        "low_macro_f1": float(macro_low.get("macro_f1", 0.0)),
        "low_snr_lo": float(low_snr_lo),
        "low_snr_hi": float(low_snr_hi),
    }
    return acc, total, acc_by_snr, summary


def evaluate_dynamic_k(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    t_eval: int,
    snr_mode: str,
    amp: bool,
    k_start: int,
    k_step: int,
    k_max: int,
    conf_thresh: float,
) -> Tuple[float, float, Dict[int, float], float]:
    """
    Dynamic evidence evaluation:
    progressively unmask more windows until confidence >= threshold (or k_max reached).

    Requires loader to yield (x, y, snr, mask) with x shape (B, Kmax, 2, 128).
    Returns (acc, total, acc_by_snr, avg_k_used).
    """
    model.eval()
    total_correct = 0
    total = 0
    total_k_used = 0
    snr_correct: Dict[int, int] = {}
    snr_total: Dict[int, int] = {}

    k_start = max(1, int(k_start))
    k_step = max(1, int(k_step))
    k_max = max(1, int(k_max))
    conf_thresh = float(conf_thresh)

    with torch.no_grad():
        for batch in loader:
            if len(batch) != 4:
                raise ValueError("Dynamic-K evaluation requires batches of (x, y, snr, mask).")
            x, y, snr, _mask_full = batch
            x = x.to(device)
            y = y.to(device)
            snr = snr.to(device)

            if x.ndim != 4:
                raise ValueError("Expected x with shape (B, Kmax, 2, 128) for dynamic-K.")
            bsz = x.shape[0]
            kmax_here = x.shape[1]
            kmax_use = min(k_max, kmax_here)

            t = torch.full((bsz,), t_eval, device=device, dtype=torch.long)
            snr_in = snr if snr_mode == "known" else None

            decided = torch.zeros(bsz, device=device, dtype=torch.bool)
            preds = torch.zeros(bsz, device=device, dtype=torch.long)
            k_used = torch.full((bsz,), kmax_use, device=device, dtype=torch.long)

            k = k_start
            while True:
                k = min(k, kmax_use)
                mask = torch.zeros((bsz, kmax_here), device=device, dtype=torch.float32)
                mask[:, :k] = 1.0

                if amp and device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=_cuda_amp_dtype(), enabled=True):
                        logits, _, _ = model(x, t, snr=snr_in, snr_mode=snr_mode, group_mask=mask)
                else:
                    logits, _, _ = model(x, t, snr=snr_in, snr_mode=snr_mode, group_mask=mask)

                prob = torch.softmax(logits, dim=1)
                conf, pred = torch.max(prob, dim=1)

                newly_decided = (~decided) & (conf >= conf_thresh)
                preds[newly_decided] = pred[newly_decided]
                k_used[newly_decided] = k
                decided = decided | newly_decided

                if decided.all() or k >= kmax_use:
                    preds[~decided] = pred[~decided]
                    k_used[~decided] = k
                    break
                k += k_step

            total_correct += (preds == y).sum().item()
            total += y.shape[0]
            total_k_used += int(k_used.sum().item())

            snr_cpu = snr.detach().cpu().numpy().astype(np.int32)
            preds_cpu = preds.detach().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()
            for snr_val, pred_val, y_val in zip(snr_cpu, preds_cpu, y_cpu):
                snr_key = int(snr_val)
                snr_correct[snr_key] = snr_correct.get(snr_key, 0) + int(pred_val == y_val)
                snr_total[snr_key] = snr_total.get(snr_key, 0) + 1

    acc = float(total_correct) / max(1, total)
    acc_by_snr = {snr: snr_correct[snr] / snr_total[snr] for snr in snr_total.keys()}
    avg_k = float(total_k_used) / max(1, total)
    return acc, total, acc_by_snr, avg_k


def evaluate_subset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    t_eval: int,
    snr_mode: str,
    amp: bool,
    max_batches: int,
) -> float:
    if max_batches <= 0:
        return 0.0
    model.eval()
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            if len(batch) == 4:
                x, y, snr, mask = batch
            else:
                x, y, snr = batch
                mask = None
            x = x.to(device)
            y = y.to(device)
            snr = snr.to(device)
            t = torch.full((x.shape[0],), t_eval, device=device, dtype=torch.long)
            snr_in = snr if snr_mode == "known" else None
            if amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=_cuda_amp_dtype(), enabled=True):
                    logits, _, _ = model(x, t, snr=snr_in, snr_mode=snr_mode, group_mask=mask)
            else:
                logits, _, _ = model(x, t, snr=snr_in, snr_mode=snr_mode, group_mask=mask)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += y.shape[0]
    return float(total_correct) / max(1, total)


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return 0.0
    x = x.float()
    y = y.float()
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(torch.clamp((x * x).sum() * (y * y).sum(), min=1e-12))
    return float(((x * y).sum() / denom).item())


def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return 0.0
    # Approximate rank transform via double argsort (ties are rare for continuous preds).
    xr = torch.argsort(torch.argsort(x)).float()
    yr = torch.argsort(torch.argsort(y)).float()
    return _pearson_corr(xr, yr)


def evaluate_eta_calibration(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    t_eval: int,
    snr_mode: str,
    amp: bool,
    rho_min: float,
    rho_max: float,
    eta_min: float,
    eta_max: float,
) -> Dict[str, object]:
    """
    Evaluate eta-head quality:
      - Pearson/Spearman correlation vs eta target
      - Per-SNR mean/std prediction and gap to target mean
    """
    model.eval()
    eta_pred_all: List[torch.Tensor] = []
    eta_tgt_all: List[torch.Tensor] = []
    snr_all: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                x, _y, snr, mask = batch
            else:
                x, _y, snr = batch
                mask = None
            x = x.to(device)
            snr = snr.to(device)
            t = torch.full((x.shape[0],), t_eval, device=device, dtype=torch.long)
            snr_in = snr if snr_mode == "known" else None
            if amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=_cuda_amp_dtype(), enabled=True):
                    _logits, _x0, _aux = model(x, t, snr=snr_in, snr_mode=snr_mode, group_mask=mask)
            else:
                _logits, _x0, _aux = model(x, t, snr=snr_in, snr_mode=snr_mode, group_mask=mask)

            eta_pred = getattr(model, "_eta_pred", None)
            if eta_pred is None:
                continue
            eta_tgt = snr_db_to_eta_target(
                snr,
                rho_min=float(rho_min),
                rho_max=float(rho_max),
                eta_min=float(eta_min),
                eta_max=float(eta_max),
            )
            eta_pred_all.append(eta_pred.detach().float().cpu())
            eta_tgt_all.append(eta_tgt.detach().float().cpu())
            snr_all.append(snr.detach().float().cpu())

    if not eta_pred_all:
        return {}

    eta_pred_cat = torch.cat(eta_pred_all, dim=0)
    eta_tgt_cat = torch.cat(eta_tgt_all, dim=0)
    snr_cat = torch.cat(snr_all, dim=0)

    pearson = _pearson_corr(eta_pred_cat, eta_tgt_cat)
    spearman = _spearman_corr(eta_pred_cat, eta_tgt_cat)

    by_snr: Dict[str, Dict[str, float]] = {}
    unique_snr = torch.unique(snr_cat).tolist()
    for s in sorted(unique_snr):
        mask = snr_cat == float(s)
        p = eta_pred_cat[mask]
        t = eta_tgt_cat[mask]
        if p.numel() == 0:
            continue
        by_snr[str(int(round(float(s))))] = {
            "eta_pred_mean": float(p.mean().item()),
            "eta_pred_std": float(p.std(unbiased=False).item()),
            "eta_tgt_mean": float(t.mean().item()),
            "eta_mae": float(torch.mean(torch.abs(p - t)).item()),
        }

    return {
        "eta_pearson": pearson,
        "eta_spearman": spearman,
        "eta_by_snr": by_snr,
    }


def train(args: argparse.Namespace) -> None:
    apply_preset(args)
    if bool(getattr(args, "cldnn_snr_cond", False)) and bool(getattr(args, "cldnn_noise_cond", False)):
        raise ValueError("Use only one conditioning path: --cldnn-snr-cond OR --cldnn-noise-cond.")
    if args.arch != "cldnn" and (
        bool(getattr(args, "cldnn_noise_cond", False))
        or bool(getattr(args, "cldnn_denoiser", False))
        or float(getattr(args, "lambda_noise", 0.0)) > 0
        or float(getattr(args, "lambda_dn", 0.0)) > 0
        or float(getattr(args, "lambda_id", 0.0)) > 0
        or float(getattr(args, "lambda_feat", 0.0)) > 0
    ):
        raise ValueError("Noise/denoiser options are currently supported only for --arch cldnn.")
    if bool(getattr(args, "cldnn_denoiser_dual_path", False)) and not bool(getattr(args, "cldnn_denoiser", False)):
        raise ValueError("--cldnn-denoiser-dual-path requires --cldnn-denoiser.")
    raw_drop_prob = float(getattr(args, "cldnn_raw_low_snr_drop_prob", 0.0))
    raw_drop_min = float(getattr(args, "cldnn_raw_low_snr_drop_min_scale", 0.0))
    raw_drop_max = float(getattr(args, "cldnn_raw_low_snr_drop_max_scale", 0.0))
    raw_drop_gate = str(getattr(args, "cldnn_raw_low_snr_drop_gate", "auto")).strip().lower()
    raw_prob_lo = float(getattr(args, "cldnn_raw_low_snr_drop_prob_lo", -1.0))
    raw_prob_mid = float(getattr(args, "cldnn_raw_low_snr_drop_prob_mid", -1.0))
    raw_prob_hi = float(getattr(args, "cldnn_raw_low_snr_drop_prob_hi", -1.0))
    if bool(getattr(args, "cldnn_raw_low_snr_drop_zero_hi", False)) and raw_prob_hi >= 0.0:
        raw_prob_hi = 0.0
        args.cldnn_raw_low_snr_drop_prob_hi = 0.0
    raw_sched_enabled = raw_prob_lo >= 0.0 and raw_prob_mid >= 0.0 and raw_prob_hi >= 0.0
    if (raw_prob_lo >= 0.0) or (raw_prob_mid >= 0.0) or (raw_prob_hi >= 0.0):
        if not raw_sched_enabled:
            raise ValueError(
                "Enable SNR-shaped raw-drop schedule by setting all of "
                "cldnn_raw_low_snr_drop_prob_lo/mid/hi >= 0."
            )
    if raw_drop_prob < 0.0 or raw_drop_prob > 1.0:
        raise ValueError("cldnn_raw_low_snr_drop_prob must be in [0,1].")
    if raw_drop_gate not in {"auto", "eta", "snr"}:
        raise ValueError("cldnn_raw_low_snr_drop_gate must be one of: auto|eta|snr.")
    if raw_drop_min < 0.0 or raw_drop_min > 1.0 or raw_drop_max < 0.0 or raw_drop_max > 1.0:
        raise ValueError("cldnn_raw_low_snr_drop_min_scale/max_scale must be in [0,1].")
    if raw_drop_max < raw_drop_min:
        raise ValueError("cldnn_raw_low_snr_drop_max_scale must be >= cldnn_raw_low_snr_drop_min_scale.")
    if raw_drop_prob > 0.0 and not bool(getattr(args, "cldnn_denoiser_dual_path", False)):
        raise ValueError("--cldnn-raw-low-snr-drop-prob requires --cldnn-denoiser-dual-path.")
    if raw_sched_enabled and not bool(getattr(args, "cldnn_denoiser_dual_path", False)):
        raise ValueError("SNR-shaped raw-drop schedule requires --cldnn-denoiser-dual-path.")
    if raw_sched_enabled:
        for p in (raw_prob_lo, raw_prob_mid, raw_prob_hi):
            if p < 0.0 or p > 1.0:
                raise ValueError("cldnn_raw_low_snr_drop_prob_lo/mid/hi must be in [0,1] when schedule is enabled.")
        if float(getattr(args, "cldnn_raw_low_snr_drop_snr_lo", -10.0)) > float(getattr(args, "cldnn_raw_low_snr_drop_snr_mid", -6.0)):
            raise ValueError("cldnn_raw_low_snr_drop_snr_lo must be <= cldnn_raw_low_snr_drop_snr_mid.")
        if raw_prob_hi > 0.0:
            print(
                "[warn] cldnn_raw_low_snr_drop_prob_hi > 0 enables high-SNR raw attenuation. "
                "This can hurt high-band accuracy; consider --cldnn-raw-low-snr-drop-zero-hi."
            )
    if int(getattr(args, "cldnn_expert_stacf_win", 0)) < 0:
        raise ValueError("cldnn_expert_stacf_win must be >= 0.")
    if float(getattr(args, "cldnn_expert_corr_eps", 1e-6)) <= 0.0:
        raise ValueError("cldnn_expert_corr_eps must be > 0.")
    if float(getattr(args, "cldnn_expert_eta_gate_tau", 0.7)) <= 0.0:
        raise ValueError("cldnn_expert_eta_gate_tau must be > 0.")
    eg_min = float(getattr(args, "cldnn_expert_eta_gate_min", 0.0))
    eg_max = float(getattr(args, "cldnn_expert_eta_gate_max", 1.0))
    if eg_min < 0.0 or eg_max < 0.0 or eg_min > 1.0 or eg_max > 1.0 or eg_max < eg_min:
        raise ValueError("cldnn_expert_eta_gate_min/max must be in [0,1] and max >= min.")
    if bool(getattr(args, "cldnn_expert_eta_gate", False)) and not bool(getattr(args, "cldnn_expert_features", False)):
        raise ValueError("--cldnn-expert-eta-gate requires --cldnn-expert-features.")
    if int(getattr(args, "stage_a_epochs", 0)) < 0 or int(getattr(args, "stage_b_epochs", 0)) < 0:
        raise ValueError("stage_a_epochs and stage_b_epochs must be >= 0.")
    if int(getattr(args, "feat_ramp_epochs", 0)) < 0:
        raise ValueError("feat_ramp_epochs must be >= 0.")
    if float(getattr(args, "lambda_feat", 0.0)) > 0 and not bool(getattr(args, "cldnn_denoiser", False)):
        raise ValueError("lambda_feat > 0 requires --cldnn-denoiser.")
    lambda_kd = float(getattr(args, "lambda_kd", 0.0))
    lambda_kd_denoise = float(getattr(args, "lambda_kd_denoise", 0.0))
    lambda_kd_feat = float(getattr(args, "lambda_kd_feat", 0.0))
    if lambda_kd < 0.0:
        raise ValueError("lambda_kd must be >= 0.")
    if lambda_kd_denoise < 0.0:
        raise ValueError("lambda_kd_denoise must be >= 0.")
    if lambda_kd_feat < 0.0:
        raise ValueError("lambda_kd_feat must be >= 0.")
    kd_any_target = (lambda_kd > 0.0) or (lambda_kd_denoise > 0.0) or (lambda_kd_feat > 0.0)
    if kd_any_target and not getattr(args, "teacher_ckpt", None):
        raise ValueError("Any KD objective requires --teacher-ckpt.")
    if int(getattr(args, "kd_warmup", 0)) < 0 or int(getattr(args, "kd_ramp", 0)) < 0:
        raise ValueError("kd_warmup and kd_ramp must be >= 0.")
    if int(getattr(args, "kd_post_stage_delay", 0)) < 0:
        raise ValueError("kd_post_stage_delay must be >= 0.")
    if float(getattr(args, "kd_temp", 2.0)) <= 0.0:
        raise ValueError("kd_temp must be > 0.")
    if float(getattr(args, "kd_snr_lo", -999.0)) > float(getattr(args, "kd_snr_hi", 999.0)):
        raise ValueError("kd_snr_lo must be <= kd_snr_hi.")
    if float(getattr(args, "kd_denoise_snr_lo", -14.0)) > float(getattr(args, "kd_denoise_snr_hi", -6.0)):
        raise ValueError("kd_denoise_snr_lo must be <= kd_denoise_snr_hi.")
    if float(getattr(args, "kd_feat_snr_lo", -14.0)) > float(getattr(args, "kd_feat_snr_hi", -6.0)):
        raise ValueError("kd_feat_snr_lo must be <= kd_feat_snr_hi.")
    if float(getattr(args, "kd_hi_preserve_scale", 0.0)) < 0.0:
        raise ValueError("kd_hi_preserve_scale must be >= 0.")
    if float(getattr(args, "kd_hi_snr_lo", 10.0)) > float(getattr(args, "kd_hi_snr_hi", 18.0)):
        raise ValueError("kd_hi_snr_lo must be <= kd_hi_snr_hi.")
    if float(getattr(args, "kd_hi_preserve_scale", 0.0)) > 0.0 and float(getattr(args, "lambda_kd", 0.0)) <= 0.0:
        raise ValueError("kd_hi_preserve_scale > 0 requires lambda_kd > 0.")
    if float(getattr(args, "kd_hi_conf_thresh", -1.0)) >= 1.0:
        raise ValueError("kd_hi_conf_thresh must be < 1.0 (or <0 to reuse kd-conf-thresh).")
    if float(getattr(args, "lambda_kd_denoise", 0.0)) > 0.0:
        if args.arch != "cldnn":
            raise ValueError("lambda_kd_denoise > 0 currently requires --arch cldnn.")
        if not bool(getattr(args, "cldnn_denoiser", False)):
            raise ValueError("lambda_kd_denoise > 0 requires --cldnn-denoiser.")
    if float(getattr(args, "lambda_kd_feat", 0.0)) > 0.0 and args.arch != "cldnn":
        raise ValueError("lambda_kd_feat > 0 currently requires --arch cldnn.")
    if not kd_any_target and getattr(args, "teacher_ckpt", None):
        print("[kd] teacher_ckpt provided but all KD lambdas <= 0; external KD objectives disabled.")
    if float(getattr(args, "report_low_snr_lo", -14.0)) > float(getattr(args, "report_low_snr_hi", -6.0)):
        raise ValueError("report_low_snr_lo must be <= report_low_snr_hi.")
    if float(getattr(args, "noise_rho_min", 1e-4)) <= 0 or float(getattr(args, "noise_rho_max", 1.0 - 1e-4)) >= 1:
        raise ValueError("noise_rho_min/max must satisfy 0 < min < max < 1.")
    if float(getattr(args, "noise_rho_min", 1e-4)) >= float(getattr(args, "noise_rho_max", 1.0 - 1e-4)):
        raise ValueError("noise_rho_min must be < noise_rho_max.")
    if float(getattr(args, "noise_eta_min", -8.0)) >= float(getattr(args, "noise_eta_max", 5.5)):
        raise ValueError("noise_eta_min must be < noise_eta_max.")
    if float(getattr(args, "dn_pair_delta_min", 2.0)) < 0.0 or float(getattr(args, "dn_pair_delta_max", 8.0)) < 0.0:
        raise ValueError("dn_pair_delta_min/max must be >= 0.")
    if float(getattr(args, "dn_pair_delta_max", 8.0)) < float(getattr(args, "dn_pair_delta_min", 2.0)):
        raise ValueError("dn_pair_delta_max must be >= dn_pair_delta_min.")
    if float(getattr(args, "lfeat_snr_lo", -999.0)) > float(getattr(args, "lfeat_snr_hi", 999.0)):
        raise ValueError("lfeat_snr_lo must be <= lfeat_snr_hi.")
    if float(getattr(args, "lfeat_snr_new_lo", -999.0)) > float(getattr(args, "lfeat_snr_new_hi", 999.0)):
        raise ValueError("lfeat_snr_new_lo must be <= lfeat_snr_new_hi.")
    if float(getattr(args, "dn_pair_snr_new_lo", -999.0)) > float(getattr(args, "dn_pair_snr_new_hi", 999.0)):
        raise ValueError("dn_pair_snr_new_lo must be <= dn_pair_snr_new_hi.")
    if float(getattr(args, "snr_consist_low_delta_min", 2.0)) < 0.0 or float(getattr(args, "snr_consist_low_delta_max", 4.0)) < 0.0:
        raise ValueError("snr_consist_low_delta_min/max must be >= 0.")
    if float(getattr(args, "snr_consist_low_delta_max", 4.0)) < float(getattr(args, "snr_consist_low_delta_min", 2.0)):
        raise ValueError("snr_consist_low_delta_max must be >= snr_consist_low_delta_min.")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    train_loader, val_loader, test_loader, mods, snrs, seq_len = build_loaders(args, device)
    snr_min_db = float(min(snrs)) if snrs else -20.0
    snr_max_db = float(max(snrs)) if snrs else 18.0
    train_eval_loader = None
    if args.train_eval_batches > 0:
        pin_memory = device.type == "cuda"
        num_workers = args.num_workers
        train_eval_loader = DataLoader(
            train_loader.dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            worker_init_fn=_seed_worker,
        )
    if args.arch == "dit":
        model = DiffusionAMC(
            num_classes=len(mods),
            seq_len=seq_len,
            patch_size=args.patch_size,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            snr_scale=args.snr_scale,
            stem_channels=args.stem_channels,
            stem_layers=args.stem_layers,
            group_pool=args.group_pool,
        ).to(device)
        schedule = DiffusionSchedule(timesteps=args.timesteps).to(device)
    elif args.arch == "multiview":
        model = MultiViewCLDNNAMC(
            num_classes=len(mods),
            seq_len=seq_len,
            conv_channels=int(args.cldnn_conv_ch),
            merge_channels=int(args.cldnn_merge_ch),
            lstm_hidden=int(args.cldnn_lstm_hidden),
            lstm_layers=int(args.cldnn_lstm_layers),
            bidirectional=bool(args.cldnn_bidir),
            dropout=float(args.dropout),
            pool=str(args.cldnn_pool),
            snr_cond=bool(args.cldnn_snr_cond),
            snr_loss_detach_backbone=bool(getattr(args, "snr_loss_detach_backbone", False)),
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            cls_hidden=int(getattr(args, 'cldnn_cls_hidden', 0)),
            stft_nfft=int(getattr(args, 'stft_nfft', 64)),
            stft_hop=int(getattr(args, 'stft_hop', 8)),
            stft_channels=int(getattr(args, 'stft_channels', 64)),
            cross_view_heads=int(getattr(args, 'cross_view_heads', 4)),
            snr_gate=bool(getattr(args, 'snr_gate', False)),
        ).to(device)
        schedule = None
    else:
        # K=1-first CNN+LSTM model (no diffusion)
        model = CLDNNAMC(
            num_classes=len(mods),
            seq_len=seq_len,
            conv_channels=int(args.cldnn_conv_ch),
            merge_channels=int(args.cldnn_merge_ch),
            lstm_hidden=int(args.cldnn_lstm_hidden),
            lstm_layers=int(args.cldnn_lstm_layers),
            bidirectional=bool(args.cldnn_bidir),
            dropout=float(args.dropout),
            pool=str(args.cldnn_pool),
            snr_cond=bool(args.cldnn_snr_cond),
            noise_cond=bool(getattr(args, "cldnn_noise_cond", False)),
            snr_loss_detach_backbone=bool(getattr(args, "snr_loss_detach_backbone", False)),
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            noise_eta_min=float(getattr(args, "noise_eta_min", -8.0)),
            noise_eta_max=float(getattr(args, "noise_eta_max", 5.5)),
            denoiser=bool(getattr(args, "cldnn_denoiser", False)),
            denoiser_dual_path=bool(getattr(args, "cldnn_denoiser_dual_path", False)),
            denoiser_base_channels=int(getattr(args, "cldnn_denoiser_base_ch", 32)),
            denoiser_dropout=float(getattr(args, "cldnn_denoiser_dropout", 0.0)),
            denoiser_soft_high_snr_blend=bool(getattr(args, "cldnn_denoiser_soft_hi_blend", False)),
            noise_head_hidden=int(getattr(args, "noise_head_hidden", 32)),
            expert_features=bool(getattr(args, 'cldnn_expert_features', False)),
            expert_channels=int(getattr(args, 'cldnn_expert_ch', 64)),
            expert_stacf_window=int(getattr(args, "cldnn_expert_stacf_win", 0)),
            expert_v2=bool(getattr(args, "cldnn_expert_v2", False)),
            expert_corr_norm_eps=float(getattr(args, "cldnn_expert_corr_eps", 1e-6)),
            expert_eta_gate=bool(getattr(args, "cldnn_expert_eta_gate", False)),
            expert_eta_gate_center=float(getattr(args, "cldnn_expert_eta_gate_center", 0.8)),
            expert_eta_gate_tau=float(getattr(args, "cldnn_expert_eta_gate_tau", 0.7)),
            expert_eta_gate_min=float(getattr(args, "cldnn_expert_eta_gate_min", 0.0)),
            expert_eta_gate_max=float(getattr(args, "cldnn_expert_eta_gate_max", 1.0)),
            expert_use_cyclo_stats=bool(getattr(args, "cldnn_cyclo_stats", True)),
            raw_low_snr_drop_prob=float(getattr(args, "cldnn_raw_low_snr_drop_prob", 0.0)),
            raw_low_snr_drop_gate=str(getattr(args, "cldnn_raw_low_snr_drop_gate", "auto")),
            raw_low_snr_drop_eta_thresh=float(getattr(args, "cldnn_raw_low_snr_drop_eta_thresh", 1.0)),
            raw_low_snr_drop_snr_thresh=float(getattr(args, "cldnn_raw_low_snr_drop_snr_thresh", -6.0)),
            raw_low_snr_drop_min_scale=float(getattr(args, "cldnn_raw_low_snr_drop_min_scale", 0.0)),
            raw_low_snr_drop_max_scale=float(getattr(args, "cldnn_raw_low_snr_drop_max_scale", 0.0)),
            raw_low_snr_drop_prob_lo=float(getattr(args, "cldnn_raw_low_snr_drop_prob_lo", -1.0)),
            raw_low_snr_drop_prob_mid=float(getattr(args, "cldnn_raw_low_snr_drop_prob_mid", -1.0)),
            raw_low_snr_drop_prob_hi=float(getattr(args, "cldnn_raw_low_snr_drop_prob_hi", -1.0)),
            raw_low_snr_drop_snr_lo=float(getattr(args, "cldnn_raw_low_snr_drop_snr_lo", -10.0)),
            raw_low_snr_drop_snr_mid=float(getattr(args, "cldnn_raw_low_snr_drop_snr_mid", -6.0)),
            cls_hidden=int(getattr(args, 'cldnn_cls_hidden', 0)),
            supcon_proj_dim=int(getattr(args, 'supcon_proj_dim', 0)) if getattr(args, 'supcon', False) else 0,
        ).to(device)
        schedule = None

    teacher_model: Optional[torch.nn.Module] = None
    kd_teacher_needed = (
        float(getattr(args, "lambda_kd", 0.0)) > 0.0
        or float(getattr(args, "lambda_kd_denoise", 0.0)) > 0.0
        or float(getattr(args, "lambda_kd_feat", 0.0)) > 0.0
    )
    if kd_teacher_needed:
        teacher_ckpt = str(getattr(args, "teacher_ckpt"))
        if not os.path.exists(teacher_ckpt):
            raise FileNotFoundError(f"teacher_ckpt not found: {teacher_ckpt}")
        try:
            t_ckpt = torch.load(teacher_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            t_ckpt = torch.load(teacher_ckpt, map_location="cpu")
        t_cfg = t_ckpt.get("args", {}) if isinstance(t_ckpt, dict) else {}
        teacher_model, teacher_arch = build_model_from_cfg(
            t_cfg if isinstance(t_cfg, dict) else None,
            args,
            num_classes=len(mods),
            seq_len=seq_len,
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            device=device,
        )
        if teacher_arch != args.arch:
            print(
                f"[kd] teacher arch ({teacher_arch}) differs from student arch ({args.arch}). "
                "This is allowed but can reduce KD effectiveness."
            )
        t_state = t_ckpt.get("model", t_ckpt) if isinstance(t_ckpt, dict) else t_ckpt
        load_stats = load_state_dict_flexible(teacher_model, t_state, prefix="kd-teacher")
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        if float(getattr(args, "lambda_kd_denoise", 0.0)) > 0.0 and not hasattr(teacher_model, "denoise_only"):
            raise ValueError("lambda_kd_denoise > 0 requires a teacher checkpoint/model with denoise_only().")
        print(
            f"[kd] loaded teacher from {teacher_ckpt} "
            f"(loaded={load_stats.get('loaded', 0)}, skipped_shape={load_stats.get('skipped_shape', 0)})."
        )

    proxy_fit_info: Dict[str, float] = {}
    if (
        args.arch == "cldnn"
        and bool(getattr(args, "cldnn_denoiser", False))
        and bool(getattr(args, "fit_noise_proxy_calibration", False))
    ):
        proxy_fit_info = fit_noise_proxy_calibration(
            model,
            train_loader,
            device,
            rho_min=float(getattr(args, "noise_rho_min", 1e-4)),
            rho_max=float(getattr(args, "noise_rho_max", 1.0 - 1e-4)),
            eta_min=float(getattr(args, "noise_eta_min", -8.0)),
            eta_max=float(getattr(args, "noise_eta_max", 5.5)),
            max_batches=int(getattr(args, "noise_proxy_calibration_batches", 256)),
        )
        if proxy_fit_info:
            print(
                f"[proxy-cal] scale={proxy_fit_info.get('proxy_cal_scale', 0.0):.6f}, "
                f"bias={proxy_fit_info.get('proxy_cal_bias', 0.0):.6f}"
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    decay_start_step = int(getattr(args, "lr_decay_start_epoch", 0)) * steps_per_epoch
    scheduler = build_scheduler(
        optimizer, args.warmup_steps, total_steps, args.min_lr, decay_start_step=decay_start_step
    )

    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = _cuda_amp_dtype() if amp_enabled else torch.float32
    if _GRADSCALER_USES_DEVICE:
        scaler = AmpGradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    else:
        scaler = AmpGradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    ema = EMA.create(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    start_epoch = 0
    global_step = 0

    if args.ckpt is not None and args.resume:
        ckpt = load_checkpoint(args.ckpt, model, optimizer=optimizer, scheduler=scheduler, ema=ema)
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)

    # Optional fixed early-feature encoder for L_feat, or lazy snapshot at Stage-B start.
    if (
        args.arch == "cldnn"
        and float(getattr(args, "lambda_feat", 0.0)) > 0
        and hasattr(model, "build_feat_encoder")
        and hasattr(model, "set_feat_encoder")
    ):
        feat_ckpt = getattr(args, "feat_encoder_ckpt", None)
        if feat_ckpt:
            if not os.path.exists(feat_ckpt):
                raise FileNotFoundError(f"feat_encoder_ckpt not found: {feat_ckpt}")
            feat_model = CLDNNAMC(
                num_classes=len(mods),
                seq_len=seq_len,
                conv_channels=int(args.cldnn_conv_ch),
                merge_channels=int(args.cldnn_merge_ch),
                lstm_hidden=int(args.cldnn_lstm_hidden),
                lstm_layers=int(args.cldnn_lstm_layers),
                bidirectional=bool(args.cldnn_bidir),
                dropout=float(args.dropout),
                pool=str(args.cldnn_pool),
                snr_cond=bool(args.cldnn_snr_cond),
                noise_cond=bool(getattr(args, "cldnn_noise_cond", False)),
                snr_loss_detach_backbone=bool(getattr(args, "snr_loss_detach_backbone", False)),
                snr_min_db=snr_min_db,
                snr_max_db=snr_max_db,
                noise_eta_min=float(getattr(args, "noise_eta_min", -8.0)),
                noise_eta_max=float(getattr(args, "noise_eta_max", 5.5)),
                denoiser=bool(getattr(args, "cldnn_denoiser", False)),
                denoiser_dual_path=bool(getattr(args, "cldnn_denoiser_dual_path", False)),
                denoiser_base_channels=int(getattr(args, "cldnn_denoiser_base_ch", 32)),
                denoiser_dropout=float(getattr(args, "cldnn_denoiser_dropout", 0.0)),
                denoiser_soft_high_snr_blend=bool(getattr(args, "cldnn_denoiser_soft_hi_blend", False)),
                noise_head_hidden=int(getattr(args, "noise_head_hidden", 32)),
                expert_features=bool(getattr(args, "cldnn_expert_features", False)),
                expert_channels=int(getattr(args, "cldnn_expert_ch", 64)),
                expert_stacf_window=int(getattr(args, "cldnn_expert_stacf_win", 0)),
                expert_v2=bool(getattr(args, "cldnn_expert_v2", False)),
                expert_corr_norm_eps=float(getattr(args, "cldnn_expert_corr_eps", 1e-6)),
                expert_eta_gate=bool(getattr(args, "cldnn_expert_eta_gate", False)),
                expert_eta_gate_center=float(getattr(args, "cldnn_expert_eta_gate_center", 0.8)),
                expert_eta_gate_tau=float(getattr(args, "cldnn_expert_eta_gate_tau", 0.7)),
                expert_eta_gate_min=float(getattr(args, "cldnn_expert_eta_gate_min", 0.0)),
                expert_eta_gate_max=float(getattr(args, "cldnn_expert_eta_gate_max", 1.0)),
                expert_use_cyclo_stats=bool(getattr(args, "cldnn_cyclo_stats", True)),
                raw_low_snr_drop_prob=float(getattr(args, "cldnn_raw_low_snr_drop_prob", 0.0)),
                raw_low_snr_drop_gate=str(getattr(args, "cldnn_raw_low_snr_drop_gate", "auto")),
                raw_low_snr_drop_eta_thresh=float(getattr(args, "cldnn_raw_low_snr_drop_eta_thresh", 1.0)),
                raw_low_snr_drop_snr_thresh=float(getattr(args, "cldnn_raw_low_snr_drop_snr_thresh", -6.0)),
                raw_low_snr_drop_min_scale=float(getattr(args, "cldnn_raw_low_snr_drop_min_scale", 0.0)),
                raw_low_snr_drop_max_scale=float(getattr(args, "cldnn_raw_low_snr_drop_max_scale", 0.0)),
                raw_low_snr_drop_prob_lo=float(getattr(args, "cldnn_raw_low_snr_drop_prob_lo", -1.0)),
                raw_low_snr_drop_prob_mid=float(getattr(args, "cldnn_raw_low_snr_drop_prob_mid", -1.0)),
                raw_low_snr_drop_prob_hi=float(getattr(args, "cldnn_raw_low_snr_drop_prob_hi", -1.0)),
                raw_low_snr_drop_snr_lo=float(getattr(args, "cldnn_raw_low_snr_drop_snr_lo", -10.0)),
                raw_low_snr_drop_snr_mid=float(getattr(args, "cldnn_raw_low_snr_drop_snr_mid", -6.0)),
                cls_hidden=int(getattr(args, "cldnn_cls_hidden", 0)),
                supcon_proj_dim=int(getattr(args, "supcon_proj_dim", 0)) if getattr(args, "supcon", False) else 0,
            ).to(device)
            try:
                ckpt_feat = torch.load(feat_ckpt, map_location="cpu", weights_only=False)
            except TypeError:
                ckpt_feat = torch.load(feat_ckpt, map_location="cpu")
            state_dict = ckpt_feat.get("model", ckpt_feat) if isinstance(ckpt_feat, dict) else ckpt_feat
            model_sd = feat_model.state_dict()
            filtered_sd = {}
            skipped_shape = []
            for k, v in state_dict.items():
                if k in model_sd and model_sd[k].shape == v.shape:
                    filtered_sd[k] = v
                elif k in model_sd:
                    skipped_shape.append(k)
            missing, unexpected = feat_model.load_state_dict(filtered_sd, strict=False)
            if missing:
                print(f"[l_feat] feat_encoder_ckpt missing keys (first 8): {missing[:8]}")
            if unexpected:
                print(f"[l_feat] feat_encoder_ckpt unexpected keys (first 8): {unexpected[:8]}")
            if skipped_shape:
                print(f"[l_feat] feat_encoder_ckpt shape-mismatch keys skipped (first 8): {skipped_shape[:8]}")
            feat_model.build_feat_encoder()
            feat_enc = feat_model.__dict__.get("_feat_encoder", None)
            if feat_enc is None:
                raise RuntimeError("Failed to build feature encoder snapshot from feat_encoder_ckpt.")
            model.set_feat_encoder(feat_enc)
            del feat_model
            print(f"[l_feat] loaded fixed feature encoder from ckpt: {feat_ckpt}")
        elif int(getattr(args, "stage_a_epochs", 0)) <= 0:
            model.build_feat_encoder()
            print("[l_feat] built feature encoder snapshot at epoch 0 (stage_a_epochs=0).")

    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    best_val = -1.0
    best_epoch = None
    epochs_no_improve = 0

    # =========================================================================
    # CONTRASTIVE PRE-TRAINING PHASE (if enabled)
    # =========================================================================
    if args.contrastive_pretrain_epochs > 0 and args.arch in ("cldnn", "multiview"):
        print(f"\n{'='*60}")
        print(f"CONTRASTIVE PRE-TRAINING: {args.contrastive_pretrain_epochs} epochs, K={args.contrastive_k}")
        print(f"{'='*60}\n")

        # Build a grouped dataset for contrastive learning
        from data import RML2016aGroupedDataset, load_rml2016a, load_rml2018a_hdf5, build_tensors
        if str(getattr(args, "dataset", "rml2016a")) == "rml2018a":
            X_cl, y_cl, snr_cl, mods_cl, snrs_cl, train_idx_cl, val_idx_cl, test_idx_cl = load_rml2018a_hdf5(
                args.data_path, seed=args.seed, train_per=args.train_per, val_per=args.val_per
            )
        else:
            X_cl, y_cl, snr_cl, mods_cl, snrs_cl, train_idx_cl, val_idx_cl, test_idx_cl = load_rml2016a(
                args.data_path, seed=args.seed, train_per=args.train_per, val_per=args.val_per
            )
        X_t_cl, y_t_cl, snr_t_cl = build_tensors(X_cl, y_cl, snr_cl)
        contrastive_dataset = RML2016aGroupedDataset(
            X_t_cl, y_t_cl, snr_t_cl, train_idx_cl,
            group_k=args.contrastive_k,
            normalize=args.normalize,
            aug_phase=args.aug_phase,
            aug_shift=args.aug_shift,
            aug_gain=args.aug_gain,
            aug_cfo=args.aug_cfo,
        )
        contrastive_loader = DataLoader(
            contrastive_dataset,
            batch_size=args.batch_size // args.contrastive_k,  # Adjust for K windows
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=_seed_worker,
        )

        # Optimizer for contrastive phase
        contrastive_lr = args.contrastive_lr if args.contrastive_lr is not None else args.lr
        contrastive_optimizer = torch.optim.AdamW(
            model.parameters(), lr=contrastive_lr, weight_decay=args.weight_decay
        )
        contrastive_steps = args.contrastive_pretrain_epochs * len(contrastive_loader)
        contrastive_scheduler = build_scheduler(
            contrastive_optimizer, args.warmup_steps, contrastive_steps, args.min_lr, decay_start_step=0
        )

        for cl_epoch in range(args.contrastive_pretrain_epochs):
            model.train()
            epoch_loss = 0.0
            epoch_total = 0

            progress = tqdm(
                contrastive_loader,
                desc=f"Contrastive {cl_epoch + 1}/{args.contrastive_pretrain_epochs}",
                unit="batch",
                dynamic_ncols=True,
            )
            for batch in progress:
                x, y, snr = batch[:3]
                x = x.to(device)  # (B, K, 2, 128)
                y = y.to(device)  # (B,)
                snr = snr.to(device)

                contrastive_optimizer.zero_grad(set_to_none=True)

                snr_in = snr if args.snr_mode == "known" else None
                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                        features = model.forward_features(x, snr=snr_in, snr_mode=args.snr_mode)
                        loss = info_nce_loss(features, y, temperature=args.contrastive_temp)
                else:
                    features = model.forward_features(x, snr=snr_in, snr_mode=args.snr_mode)
                    loss = info_nce_loss(features, y, temperature=args.contrastive_temp)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(contrastive_optimizer)
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(contrastive_optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    contrastive_optimizer.step()

                contrastive_scheduler.step()
                epoch_loss += loss.item() * x.shape[0]
                epoch_total += x.shape[0]

                progress.set_postfix(
                    loss=f"{epoch_loss / max(1, epoch_total):.4f}",
                    lr=f"{contrastive_optimizer.param_groups[0]['lr']:.2e}",
                )

            tqdm.write(f"Contrastive epoch {cl_epoch + 1}: loss={epoch_loss / max(1, epoch_total):.4f}")

        print(f"\n{'='*60}")
        print("CONTRASTIVE PRE-TRAINING COMPLETE. Starting fine-tuning...")
        print(f"{'='*60}\n")

        # Reset optimizer for fine-tuning phase
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = max(1, len(train_loader))
        total_steps = args.epochs * steps_per_epoch
        decay_start_step = int(getattr(args, "lr_decay_start_epoch", 0)) * steps_per_epoch
        scheduler = build_scheduler(
            optimizer, args.warmup_steps, total_steps, args.min_lr, decay_start_step=decay_start_step
        )

    for epoch in range(start_epoch, args.epochs):
        phase1 = epoch < args.phase1_epochs
        lambda_diff = args.phase1_lambda_diff if phase1 else args.lambda_diff
        p_clean = args.phase1_p_clean if phase1 else args.p_clean
        cls2dn_scale = get_cls2dn_scale(epoch, args)
        cls_loss_mult = 1.0
        if bool(getattr(args, "cldnn_denoiser", False)) and bool(getattr(args, "stage_a_no_cls", False)):
            if epoch < int(getattr(args, "stage_a_epochs", 0)):
                cls_loss_mult = 0.0
        lambda_feat = 0.0
        if (
            args.arch == "cldnn"
            and float(getattr(args, "lambda_feat", 0.0)) > 0.0
            and hasattr(model, "build_feat_encoder")
            and bool(getattr(model, "has_feat_encoder", False)) is False
            and epoch >= int(getattr(args, "stage_a_epochs", 0))
        ):
            model.build_feat_encoder()
            print(f"[l_feat] built feature encoder snapshot at epoch {epoch}.")
        if args.arch == "cldnn" and bool(getattr(model, "has_feat_encoder", False)):
            lambda_feat = get_lambda_feat(epoch, args)
        lambda_kd_eff = get_lambda_kd(
            epoch,
            args,
            teacher_enabled=(teacher_model is not None),
            target_attr="lambda_kd",
        )
        lambda_kd_denoise_eff = get_lambda_kd(
            epoch,
            args,
            teacher_enabled=(teacher_model is not None),
            target_attr="lambda_kd_denoise",
        )
        lambda_kd_feat_eff = get_lambda_kd(
            epoch,
            args,
            teacher_enabled=(teacher_model is not None),
            target_attr="lambda_kd_feat",
        )

        # Curriculum learning: compute minimum SNR for this epoch
        curriculum_snr_min = get_curriculum_snr_min(
            epoch,
            args.curriculum_epochs,
            args.curriculum_snr_start,
            snr_end=-20.0,
        )

        model.train()
        epoch_loss = 0.0
        epoch_loss_feat = 0.0
        epoch_lfeat_active = 0.0
        epoch_loss_kd = 0.0
        epoch_kd_active = 0.0
        epoch_loss_kd_hi = 0.0
        epoch_kd_hi_active = 0.0
        epoch_loss_kd_denoise = 0.0
        epoch_kd_denoise_active = 0.0
        epoch_loss_kd_feat = 0.0
        epoch_kd_feat_active = 0.0
        epoch_correct = 0
        epoch_total = 0
        start_time = time.time()

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for batch in progress:
            if len(batch) == 4:
                x, y, snr, mask = batch
            else:
                x, y, snr = batch
                mask = None
            x = x.to(device)
            y = y.to(device)
            snr = snr.to(device)
            mask = mask.to(device) if mask is not None else None

            # --- Training path depends on architecture ---
            # group_size is used only for DiT-AMC token diffusion (flatten windows) and for grouped pooling.
            group_size = x.shape[1] if x.ndim == 4 else None
            if args.arch == "dit":
                # Token-space diffusion: z_t = sqrt(a)*z0 + sqrt(1-a)*eps
                # z0 are patch embeddings BEFORE positional embedding; model adds pos internally.
                if x.ndim == 4:
                    x_flat = x.reshape(-1, x.shape[2], x.shape[3])
                else:
                    x_flat = x
                z0 = model.encode(x_flat)  # type: ignore[attr-defined]
                if schedule is None:
                    raise RuntimeError("schedule is None but arch=dit")
                t_max = args.t_max if args.t_max is not None and args.t_max > 0 else schedule.timesteps
                t_max = min(t_max, schedule.timesteps)
                if args.t_schedule == "snr":
                    if group_size is not None:
                        snr_rep = snr.repeat_interleave(group_size)
                    else:
                        snr_rep = snr
                    t = schedule.snr_to_t(snr_rep)
                    if t_max < schedule.timesteps:
                        t = torch.clamp(t, max=t_max - 1)
                else:
                    t = torch.randint(0, t_max, (z0.shape[0],), device=device, dtype=torch.long)

                # Cap augmentation strength for low-SNR samples (training-time label only).
                if args.snr_floor_db is not None:
                    snr_floor = float(args.snr_floor_db)
                    snr_cap_max = float(args.snr_cap_max_db)
                    denom = max(1e-6, snr_cap_max - snr_floor)
                    snr_clamped = torch.clamp(snr.float(), min=snr_floor, max=snr_cap_max)
                    frac = (snr_clamped - snr_floor) / denom  # 0 at floor, 1 at cap_max
                    t_cap = torch.floor(frac * float(t_max - 1)).long()
                    t = torch.minimum(t, t_cap)

                if p_clean > 0:
                    clean_mask = torch.rand(z0.shape[0], device=device) < p_clean
                    t = torch.where(clean_mask, torch.zeros_like(t), t)
                eps = torch.randn_like(z0)
                alpha_bar = schedule.alpha_bars.gather(0, t).view(-1, 1, 1)
                zt = torch.sqrt(alpha_bar) * z0 + torch.sqrt(1.0 - alpha_bar) * eps
            else:
                # CLDNN: standard supervised classification (no diffusion/noising)
                t = torch.zeros((x.shape[0],), device=device, dtype=torch.long)
                z0 = None
                zt = None

            # --- Curriculum learning: per-sample weighting (hard or soft) ---
            curriculum_mask_base = curriculum_weights(
                snr,
                curriculum_snr_min,
                epoch=epoch,
                curriculum_epochs=int(args.curriculum_epochs),
                soft=bool(getattr(args, "curriculum_soft", False)),
                soft_low_weight=float(getattr(args, "curriculum_soft_low_weight", 0.1)),
            )

            # Keep clean references for auxiliary losses.
            x_clean = x
            snr_clean = snr
            x_cls = x_clean
            snr_cls = snr_clean
            curriculum_mask_cls = curriculum_mask_base

            # --- Mixup augmentation (classification path only by default) ---
            use_mixup = False
            mixup_cls_only = bool(getattr(args, "mixup_cls_only", True))
            y_a, y_b, lam = y, y, 1.0
            kd_objectives_configured = (
                teacher_model is not None
                and (
                    float(getattr(args, "lambda_kd", 0.0)) > 0.0
                    or float(getattr(args, "lambda_kd_denoise", 0.0)) > 0.0
                    or float(getattr(args, "lambda_kd_feat", 0.0)) > 0.0
                )
            )
            allow_mixup = not (bool(getattr(args, "kd_disable_mixup", False)) and kd_objectives_configured)
            if allow_mixup and args.arch != "dit" and args.mixup_alpha > 0 and random.random() < args.mixup_prob:
                use_mixup = True
                x_cls, y_a, y_b, lam, snr_cls = mixup_data(
                    x_clean,
                    y,
                    snr_clean,
                    alpha=args.mixup_alpha,
                    snr_min=getattr(args, "mixup_snr_min", None),
                )
                curriculum_mask_cls = curriculum_weights(
                    snr_cls,
                    curriculum_snr_min,
                    epoch=epoch,
                    curriculum_epochs=int(args.curriculum_epochs),
                    soft=bool(getattr(args, "curriculum_soft", False)),
                    soft_low_weight=float(getattr(args, "curriculum_soft_low_weight", 0.1)),
                )

            # Auxiliary losses should run on clean inputs unless legacy mixup-all is requested.
            x_aux = x_clean if (not use_mixup or mixup_cls_only) else x_cls
            snr_aux = snr_clean if (not use_mixup or mixup_cls_only) else snr_cls

            optimizer.zero_grad(set_to_none=True)
            snr_in_aux = snr_aux if args.snr_mode == "known" else None
            snr_in_cls = snr_cls if args.snr_mode == "known" else None
            snr_in = snr_in_aux
            extra_cls2dn = {"cls_to_denoiser_scale": cls2dn_scale} if args.arch == "cldnn" else {}
            logits_for_acc = None
            logits_teacher_base = None
            eta_pred_for_noise = None
            loss_feat = torch.tensor(0.0, device=device)
            lfeat_active_frac_batch = torch.tensor(0.0, device=device)
            loss_kd = torch.tensor(0.0, device=device)
            kd_active_frac_batch = torch.tensor(0.0, device=device)
            loss_kd_hi = torch.tensor(0.0, device=device)
            kd_hi_active_frac_batch = torch.tensor(0.0, device=device)
            loss_kd_denoise = torch.tensor(0.0, device=device)
            kd_denoise_active_frac_batch = torch.tensor(0.0, device=device)
            loss_kd_feat = torch.tensor(0.0, device=device)
            kd_feat_active_frac_batch = torch.tensor(0.0, device=device)
            student_prefilm_kd: Optional[torch.Tensor] = None
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                    if args.arch == "dit":
                        logits, x0_pred, snr_pred = model.forward_tokens(  # type: ignore[attr-defined]
                            zt,
                            t,
                            snr=snr_in,
                            snr_mode=args.snr_mode,
                            group_size=group_size,
                            group_mask=mask,
                        )
                        loss_diff = F.mse_loss(x0_pred, z0)
                        logits_for_acc = logits
                        logits_teacher_base = logits
                    else:
                        if use_mixup and mixup_cls_only:
                            logits_clean, _x0_pred, snr_pred = model(
                                x_aux,
                                t,
                                snr=snr_in_aux,
                                snr_mode=args.snr_mode,
                                group_mask=mask,
                                **extra_cls2dn,
                            )
                            student_prefilm_kd = getattr(model, "_pooled_pre_film", None)
                            eta_pred_for_noise = getattr(model, "_eta_pred", None)
                            logits, _x0_pred_mix, _snr_pred_mix = model(
                                x_cls,
                                t,
                                snr=snr_in_cls,
                                snr_mode=args.snr_mode,
                                group_mask=mask,
                                **extra_cls2dn,
                            )
                            logits_for_acc = logits_clean
                            logits_teacher_base = logits_clean
                        else:
                            logits, _x0_pred, snr_pred = model(
                                x_cls,
                                t,
                                snr=snr_in_cls,
                                snr_mode=args.snr_mode,
                                group_mask=mask,
                                **extra_cls2dn,
                            )
                            student_prefilm_kd = getattr(model, "_pooled_pre_film", None)
                            eta_pred_for_noise = getattr(model, "_eta_pred", None)
                            logits_for_acc = logits
                            logits_teacher_base = logits
                        loss_diff = 0.0

                    # Compute classification loss (with optional mixup and curriculum)
                    focal_gamma = float(getattr(args, 'focal_gamma', 0.0))
                    if use_mixup:
                        ce_a = focal_cross_entropy(logits, y_a, gamma=focal_gamma, label_smoothing=float(args.label_smoothing))
                        ce_b = focal_cross_entropy(logits, y_b, gamma=focal_gamma, label_smoothing=float(args.label_smoothing))
                        ce = lam * ce_a + (1 - lam) * ce_b
                    else:
                        ce = focal_cross_entropy(logits, y, gamma=focal_gamma, label_smoothing=float(args.label_smoothing))

                    # Apply low-SNR boost weighting
                    if args.low_snr_boost and args.low_snr_boost > 0 and args.snr_floor_db is not None:
                        snr_floor = float(args.snr_floor_db)
                        snr_cap_max = float(args.snr_cap_max_db)
                        denom = max(1e-6, snr_cap_max - snr_floor)
                        snr_clamped = torch.clamp(snr_cls.float(), min=snr_floor, max=snr_cap_max)
                        frac = (snr_clamped - snr_floor) / denom
                        weights = 1.0 + float(args.low_snr_boost) * (1.0 - frac)
                        ce = ce * weights

                    # Apply curriculum mask (zero out samples below SNR threshold)
                    ce = ce * curriculum_mask_cls
                    denom_curriculum = torch.clamp(curriculum_mask_cls.sum(), min=1.0)
                    loss_cls = ce.sum() / denom_curriculum

                    loss_snr = F.smooth_l1_loss(snr_pred, snr_aux) if args.lambda_snr > 0 else 0.0
                    loss_noise = 0.0
                    if float(getattr(args, "lambda_noise", 0.0)) > 0:
                        eta_pred = eta_pred_for_noise if eta_pred_for_noise is not None else getattr(model, "_eta_pred", None)
                        if eta_pred is not None:
                            eta_target = snr_db_to_eta_target(
                                snr_aux,
                                rho_min=float(getattr(args, "noise_rho_min", 1e-4)),
                                rho_max=float(getattr(args, "noise_rho_max", 1.0 - 1e-4)),
                                eta_min=float(getattr(args, "noise_eta_min", -8.0)),
                                eta_max=float(getattr(args, "noise_eta_max", 5.5)),
                            )
                            loss_noise = F.smooth_l1_loss(eta_pred.float(), eta_target.float())
                    loss = (
                        cls_loss_mult * loss_cls
                        + lambda_diff * loss_diff
                        + args.lambda_snr * loss_snr
                        + float(getattr(args, "lambda_noise", 0.0)) * loss_noise
                    )
                    if (
                        bool(getattr(args, "cldnn_denoiser", False))
                        and args.arch == "cldnn"
                        and hasattr(model, "denoise_only")
                    ):
                        if x_aux.ndim == 4:
                            g = x_aux.shape[1]
                            x_dn_ref = x_aux.reshape(-1, x_aux.shape[2], x_aux.shape[3])
                            snr_dn_ref = snr_aux.repeat_interleave(g)
                        else:
                            x_dn_ref = x_aux
                            snr_dn_ref = snr_aux
                        dn_pair_delta_min = float(getattr(args, "dn_pair_delta_min", 2.0))
                        dn_pair_delta_max = float(getattr(args, "dn_pair_delta_max", 8.0))
                        dn_pair_snr_floor = (
                            float(getattr(args, "dn_pair_snr_floor_db"))
                            if getattr(args, "dn_pair_snr_floor_db", None) is not None
                            else float(snr_min_db)
                        )
                        dn_pair_snr_new_lo_raw = float(getattr(args, "dn_pair_snr_new_lo", -999.0))
                        dn_pair_snr_new_hi_raw = float(getattr(args, "dn_pair_snr_new_hi", 999.0))
                        dn_pair_snr_new_lo = dn_pair_snr_new_lo_raw if dn_pair_snr_new_lo_raw > -900.0 else None
                        dn_pair_snr_new_hi = dn_pair_snr_new_hi_raw if dn_pair_snr_new_hi_raw < 900.0 else None
                        x_low_dn, delta_dn = snr_path_degrade(
                            x_dn_ref,
                            snr_dn_ref,
                            delta_min=dn_pair_delta_min,
                            delta_max=dn_pair_delta_max,
                            snr_floor=dn_pair_snr_floor,
                            snr_target_min=dn_pair_snr_new_lo,
                            snr_target_max=dn_pair_snr_new_hi,
                        )
                        snr_new_dn = snr_dn_ref.float() - delta_dn.float()
                        snr_for_low = snr_new_dn if args.snr_mode == "known" else None
                        x_dn_low, _eta_low, _eta_cond_low = model.denoise_only(
                            x_low_dn, snr=snr_for_low, snr_mode=args.snr_mode
                        )
                        loss_dn = torch.mean(torch.abs(x_dn_low.float() - x_dn_ref.float()))
                        snr_for_hi = snr_dn_ref if args.snr_mode == "known" else None
                        x_dn_hi, _eta_hi, _eta_cond_hi = model.denoise_only(
                            x_dn_ref, snr=snr_for_hi, snr_mode=args.snr_mode
                        )
                        m_hi = high_snr_soft_mask(snr_dn_ref)
                        id_vec = torch.mean(torch.abs(x_dn_hi.float() - x_dn_ref.float()), dim=(1, 2))
                        loss_id = (m_hi * id_vec).sum() / torch.clamp(m_hi.sum(), min=1.0)
                        if (
                            float(lambda_feat) > 0.0
                            and hasattr(model, "early_features")
                            and bool(getattr(model, "has_feat_encoder", False))
                        ):
                            if bool(getattr(model, "denoiser_dual_path", False)):
                                feat_in_pred = torch.cat([x_low_dn.float(), x_dn_low.float()], dim=1)
                                feat_in_tgt = torch.cat([x_low_dn.float(), x_dn_ref.float()], dim=1)
                            else:
                                feat_in_pred = x_dn_low.float()
                                feat_in_tgt = x_dn_ref.float()
                            with torch.autocast(device_type="cuda", enabled=False):
                                feat_pred = model.early_features(feat_in_pred)
                                feat_tgt = model.early_features(feat_in_tgt).detach()
                                feat_vec = torch.mean(torch.abs(feat_pred - feat_tgt), dim=1)
                                lfeat_mask = torch.ones_like(snr_dn_ref.float())
                                lfeat_src_lo = float(getattr(args, "lfeat_snr_lo", -999.0))
                                lfeat_src_hi = float(getattr(args, "lfeat_snr_hi", 999.0))
                                if lfeat_src_lo > -900.0 or lfeat_src_hi < 900.0:
                                    lfeat_mask = lfeat_mask * (
                                        (snr_dn_ref.float() >= lfeat_src_lo)
                                        & (snr_dn_ref.float() <= lfeat_src_hi)
                                    ).float()
                                lfeat_new_lo = float(getattr(args, "lfeat_snr_new_lo", -999.0))
                                lfeat_new_hi = float(getattr(args, "lfeat_snr_new_hi", 999.0))
                                if lfeat_new_lo > -900.0 or lfeat_new_hi < 900.0:
                                    lfeat_mask = lfeat_mask * (
                                        (snr_new_dn.float() >= lfeat_new_lo)
                                        & (snr_new_dn.float() <= lfeat_new_hi)
                                    ).float()
                                loss_feat = (feat_vec * lfeat_mask).sum() / torch.clamp(lfeat_mask.sum(), min=1.0)
                                lfeat_active_frac_batch = lfeat_mask.mean()
                        loss = (
                            loss
                            + float(getattr(args, "lambda_dn", 0.0)) * loss_dn
                            + float(getattr(args, "lambda_id", 0.0)) * loss_id
                            + float(lambda_feat) * loss_feat
                        )

                    # --- External teacher KD (oracle teacher -> blind student) ---
                    if (
                        teacher_model is not None
                        and (
                            float(lambda_kd_eff) > 0.0
                            or float(lambda_kd_denoise_eff) > 0.0
                            or float(lambda_kd_feat_eff) > 0.0
                        )
                    ):
                        teacher_snr_mode = str(getattr(args, "kd_teacher_snr_mode", "known"))
                        teacher_snr_in = snr_aux if teacher_snr_mode == "known" else None
                        labels_kd = y if (not use_mixup or mixup_cls_only) else None
                        logits_teacher_kd: Optional[torch.Tensor] = None
                        teacher_prefilm_kd: Optional[torch.Tensor] = None

                        if float(lambda_kd_eff) > 0.0 or float(lambda_kd_feat_eff) > 0.0:
                            with torch.no_grad():
                                logits_teacher_kd, _, _ = teacher_model(
                                    x_aux,
                                    t,
                                    snr=teacher_snr_in,
                                    snr_mode=teacher_snr_mode,
                                    group_mask=mask,
                                )
                                teacher_prefilm_kd = getattr(teacher_model, "_pooled_pre_film", None)

                        if float(lambda_kd_eff) > 0.0 and logits_teacher_kd is not None:
                            logits_student_kd = logits_teacher_base if logits_teacher_base is not None else logits
                            loss_kd, kd_active_frac_batch = kd_distillation_loss(
                                logits_student_kd,
                                logits_teacher_kd,
                                temperature=float(getattr(args, "kd_temp", 2.0)),
                                snr_db=snr_aux,
                                snr_lo=float(getattr(args, "kd_snr_lo", -999.0)),
                                snr_hi=float(getattr(args, "kd_snr_hi", 999.0)),
                                conf_thresh=float(getattr(args, "kd_conf_thresh", 0.0)),
                                labels=labels_kd,
                                correctness_filter=bool(getattr(args, "kd_correctness_filter", True)),
                                normalize_by_active=bool(getattr(args, "kd_normalize_by_active", False)),
                            )
                            kd_hi_scale = float(getattr(args, "kd_hi_preserve_scale", 0.0))
                            if kd_hi_scale > 0.0:
                                kd_hi_conf = float(getattr(args, "kd_hi_conf_thresh", -1.0))
                                if kd_hi_conf < 0.0:
                                    kd_hi_conf = float(getattr(args, "kd_conf_thresh", 0.0))
                                loss_kd_hi, kd_hi_active_frac_batch = kd_distillation_loss(
                                    logits_student_kd,
                                    logits_teacher_kd,
                                    temperature=float(getattr(args, "kd_temp", 2.0)),
                                    snr_db=snr_aux,
                                    snr_lo=float(getattr(args, "kd_hi_snr_lo", 10.0)),
                                    snr_hi=float(getattr(args, "kd_hi_snr_hi", 18.0)),
                                    conf_thresh=kd_hi_conf,
                                    labels=labels_kd,
                                    correctness_filter=bool(getattr(args, "kd_correctness_filter", True)),
                                    normalize_by_active=bool(getattr(args, "kd_normalize_by_active", False)),
                                )
                                loss_kd = loss_kd + kd_hi_scale * loss_kd_hi
                            loss = loss + float(lambda_kd_eff) * loss_kd

                        if (
                            float(lambda_kd_denoise_eff) > 0.0
                            and args.arch == "cldnn"
                            and hasattr(model, "denoise_only")
                            and hasattr(teacher_model, "denoise_only")
                        ):
                            if x_aux.ndim == 4:
                                g_kd = x_aux.shape[1]
                                x_dn_kd = x_aux.reshape(-1, x_aux.shape[2], x_aux.shape[3])
                                snr_dn_kd = snr_aux.repeat_interleave(g_kd)
                            else:
                                x_dn_kd = x_aux
                                snr_dn_kd = snr_aux
                            snr_teacher_dn = snr_dn_kd if teacher_snr_mode == "known" else None
                            snr_student_dn = snr_dn_kd if args.snr_mode == "known" else None
                            with torch.no_grad():
                                x_dn_teacher, _, _ = teacher_model.denoise_only(
                                    x_dn_kd, snr=snr_teacher_dn, snr_mode=teacher_snr_mode
                                )
                            x_dn_student, _, _ = model.denoise_only(
                                x_dn_kd, snr=snr_student_dn, snr_mode=args.snr_mode
                            )
                            dn_vec = torch.mean(torch.abs(x_dn_student.float() - x_dn_teacher.float()), dim=(1, 2))
                            dn_mask = (
                                (snr_dn_kd.float() >= float(getattr(args, "kd_denoise_snr_lo", -14.0)))
                                & (snr_dn_kd.float() <= float(getattr(args, "kd_denoise_snr_hi", -6.0)))
                            ).float()
                            loss_kd_denoise = (dn_vec * dn_mask).mean()
                            kd_denoise_active_frac_batch = dn_mask.mean()
                            loss = loss + float(lambda_kd_denoise_eff) * loss_kd_denoise

                        if (
                            float(lambda_kd_feat_eff) > 0.0
                            and student_prefilm_kd is not None
                            and teacher_prefilm_kd is not None
                            and student_prefilm_kd.shape == teacher_prefilm_kd.shape
                        ):
                            feat_student = F.normalize(student_prefilm_kd.float(), dim=1)
                            feat_teacher = F.normalize(teacher_prefilm_kd.detach().float(), dim=1)
                            feat_vec = torch.sum((feat_student - feat_teacher) ** 2, dim=1)
                            snr_feat_kd = snr_aux
                            if snr_feat_kd.shape[0] != feat_vec.shape[0] and x_aux.ndim == 4:
                                snr_feat_kd = snr_aux.repeat_interleave(x_aux.shape[1])
                            if snr_feat_kd.shape[0] == feat_vec.shape[0]:
                                feat_mask = (
                                    (snr_feat_kd.float() >= float(getattr(args, "kd_feat_snr_lo", -14.0)))
                                    & (snr_feat_kd.float() <= float(getattr(args, "kd_feat_snr_hi", -6.0)))
                                ).float()
                                loss_kd_feat = (feat_vec * feat_mask).mean()
                                kd_feat_active_frac_batch = feat_mask.mean()
                                loss = loss + float(lambda_kd_feat_eff) * loss_kd_feat

                    # --- Supervised Contrastive Loss (SupCon) ---
                    # Skip on mixup batches — labels are ambiguous after interpolation.
                    if (
                        getattr(args, "supcon", False)
                        and args.arch != "dit"
                        and epoch >= int(getattr(args, "supcon_warmup", 0))
                        and not use_mixup
                        and hasattr(model, "proj_head")
                        and model.proj_head is not None
                    ):
                        # Project pre-FiLM pooled features → L2-normalised embedding
                        z_proj = model.proj_head(model._pooled_pre_film)
                        z_proj = F.normalize(z_proj, dim=1)
                        loss_supcon = supervised_contrastive_loss(
                            z_proj, y,
                            temperature=float(getattr(args, "supcon_temp", 0.07)),
                        )
                        loss = loss + float(getattr(args, "supcon_lambda", 0.1)) * loss_supcon

                    # --- SNR-path consistency loss (core novelty) ---
                    # Allow consistency when mixup is classification-only; the auxiliary
                    # path still uses clean (physical) samples in that case.
                    if (
                        getattr(args, "snr_consist", False)
                        and args.arch != "dit"
                        and epoch >= getattr(args, "snr_consist_warmup", 5)
                        and (not use_mixup or mixup_cls_only)
                    ):
                        # Optionally recompute teacher logits with dropout OFF (eval mode)
                        logits_teacher = logits_teacher_base if logits_teacher_base is not None else logits
                        if bool(getattr(args, "snr_consist_teacher_eval", False)):
                            was_training = model.training
                            model.eval()
                            with torch.no_grad():
                                logits_teacher, _, _ = model(
                                    x_aux,
                                    t,
                                    snr=snr_in_aux,
                                    snr_mode=args.snr_mode,
                                    group_mask=mask,
                                    **extra_cls2dn,
                                )
                            if was_training:
                                model.train()

                        consist_delta_min: Union[float, torch.Tensor] = float(getattr(args, "snr_consist_delta_min", 2.0))
                        consist_delta_max: Union[float, torch.Tensor] = float(getattr(args, "snr_consist_delta_max", 8.0))
                        if bool(getattr(args, "snr_consist_adaptive_delta", False)):
                            s_aux = snr_aux.float()
                            dmin_t = torch.full_like(s_aux, float(getattr(args, "snr_consist_delta_min", 2.0)))
                            dmax_t = torch.full_like(s_aux, float(getattr(args, "snr_consist_delta_max", 8.0)))
                            low_mask = s_aux <= float(getattr(args, "snr_consist_low_snr_thresh", -6.0))
                            dmin_t = torch.where(
                                low_mask,
                                torch.full_like(dmin_t, float(getattr(args, "snr_consist_low_delta_min", 2.0))),
                                dmin_t,
                            )
                            dmax_t = torch.where(
                                low_mask,
                                torch.full_like(dmax_t, float(getattr(args, "snr_consist_low_delta_max", 4.0))),
                                dmax_t,
                            )
                            consist_delta_min = dmin_t
                            consist_delta_max = dmax_t
                        x_low, _delta = snr_path_degrade(
                            x_aux, snr_aux,
                            delta_min=consist_delta_min,
                            delta_max=consist_delta_max,
                            snr_floor=float(snr_min_db),
                        )
                        snr_new = snr_aux.float() - _delta.float()
                        logits_low, _, _ = model(
                            x_low,
                            t,
                            snr=snr_in_aux,
                            snr_mode=args.snr_mode,
                            group_mask=mask,
                            **extra_cls2dn,
                        )
                        loss_consist = snr_consistency_loss(
                            logits_teacher, logits_low,
                            temperature=float(getattr(args, "snr_consist_temp", 2.0)),
                            snr_db=snr_aux,
                            snr_lo=float(getattr(args, "snr_consist_snr_lo", -999.0)),
                            snr_hi=float(getattr(args, "snr_consist_snr_hi", 999.0)),
                            conf_thresh=float(getattr(args, "snr_consist_conf_thresh", 0.0)),
                            snr_new_db=snr_new,
                            snr_new_lo=float(getattr(args, "snr_consist_snr_new_lo", -999.0)),
                            snr_new_hi=float(getattr(args, "snr_consist_snr_new_hi", 999.0)),
                        )
                        consist_lambda = float(getattr(args, "snr_consist_lambda", 1.0))
                        ramp_epochs = int(getattr(args, "snr_consist_ramp", 0))
                        warmup_ep = int(getattr(args, "snr_consist_warmup", 5))
                        if ramp_epochs and ramp_epochs > 0:
                            ramp = (epoch - warmup_ep + 1) / float(max(1, ramp_epochs))
                            ramp = max(0.0, min(1.0, ramp))
                        else:
                            ramp = 1.0
                        loss = loss + consist_lambda * ramp * loss_consist
            else:
                if args.arch == "dit":
                    logits, x0_pred, snr_pred = model.forward_tokens(  # type: ignore[attr-defined]
                        zt,
                        t,
                        snr=snr_in,
                        snr_mode=args.snr_mode,
                        group_size=group_size,
                        group_mask=mask,
                    )
                    loss_diff = F.mse_loss(x0_pred, z0)
                    logits_for_acc = logits
                    logits_teacher_base = logits
                else:
                    if use_mixup and mixup_cls_only:
                        logits_clean, _x0_pred, snr_pred = model(
                            x_aux,
                            t,
                            snr=snr_in_aux,
                            snr_mode=args.snr_mode,
                            group_mask=mask,
                            **extra_cls2dn,
                        )
                        student_prefilm_kd = getattr(model, "_pooled_pre_film", None)
                        eta_pred_for_noise = getattr(model, "_eta_pred", None)
                        logits, _x0_pred_mix, _snr_pred_mix = model(
                            x_cls,
                            t,
                            snr=snr_in_cls,
                            snr_mode=args.snr_mode,
                            group_mask=mask,
                            **extra_cls2dn,
                        )
                        logits_for_acc = logits_clean
                        logits_teacher_base = logits_clean
                    else:
                        logits, _x0_pred, snr_pred = model(
                            x_cls,
                            t,
                            snr=snr_in_cls,
                            snr_mode=args.snr_mode,
                            group_mask=mask,
                            **extra_cls2dn,
                        )
                        student_prefilm_kd = getattr(model, "_pooled_pre_film", None)
                        eta_pred_for_noise = getattr(model, "_eta_pred", None)
                        logits_for_acc = logits
                        logits_teacher_base = logits
                    loss_diff = 0.0

                # Compute classification loss (with optional mixup and curriculum)
                focal_gamma = float(getattr(args, 'focal_gamma', 0.0))
                if use_mixup:
                    ce_a = focal_cross_entropy(logits, y_a, gamma=focal_gamma, label_smoothing=float(args.label_smoothing))
                    ce_b = focal_cross_entropy(logits, y_b, gamma=focal_gamma, label_smoothing=float(args.label_smoothing))
                    ce = lam * ce_a + (1 - lam) * ce_b
                else:
                    ce = focal_cross_entropy(logits, y, gamma=focal_gamma, label_smoothing=float(args.label_smoothing))

                # Apply low-SNR boost weighting
                if args.low_snr_boost and args.low_snr_boost > 0 and args.snr_floor_db is not None:
                    snr_floor = float(args.snr_floor_db)
                    snr_cap_max = float(args.snr_cap_max_db)
                    denom = max(1e-6, snr_cap_max - snr_floor)
                    snr_clamped = torch.clamp(snr_cls.float(), min=snr_floor, max=snr_cap_max)
                    frac = (snr_clamped - snr_floor) / denom
                    weights = 1.0 + float(args.low_snr_boost) * (1.0 - frac)
                    ce = ce * weights

                # Apply curriculum mask (zero out samples below SNR threshold)
                ce = ce * curriculum_mask_cls
                denom_curriculum = torch.clamp(curriculum_mask_cls.sum(), min=1.0)
                loss_cls = ce.sum() / denom_curriculum

                loss_snr = F.smooth_l1_loss(snr_pred, snr_aux) if args.lambda_snr > 0 else 0.0
                loss_noise = 0.0
                if float(getattr(args, "lambda_noise", 0.0)) > 0:
                    eta_pred = eta_pred_for_noise if eta_pred_for_noise is not None else getattr(model, "_eta_pred", None)
                    if eta_pred is not None:
                        eta_target = snr_db_to_eta_target(
                            snr_aux,
                            rho_min=float(getattr(args, "noise_rho_min", 1e-4)),
                            rho_max=float(getattr(args, "noise_rho_max", 1.0 - 1e-4)),
                            eta_min=float(getattr(args, "noise_eta_min", -8.0)),
                            eta_max=float(getattr(args, "noise_eta_max", 5.5)),
                        )
                        loss_noise = F.smooth_l1_loss(eta_pred.float(), eta_target.float())
                loss = (
                    cls_loss_mult * loss_cls
                    + lambda_diff * loss_diff
                    + args.lambda_snr * loss_snr
                    + float(getattr(args, "lambda_noise", 0.0)) * loss_noise
                )
                if (
                    bool(getattr(args, "cldnn_denoiser", False))
                    and args.arch == "cldnn"
                    and hasattr(model, "denoise_only")
                ):
                    if x_aux.ndim == 4:
                        g = x_aux.shape[1]
                        x_dn_ref = x_aux.reshape(-1, x_aux.shape[2], x_aux.shape[3])
                        snr_dn_ref = snr_aux.repeat_interleave(g)
                    else:
                        x_dn_ref = x_aux
                        snr_dn_ref = snr_aux
                    dn_pair_delta_min = float(getattr(args, "dn_pair_delta_min", 2.0))
                    dn_pair_delta_max = float(getattr(args, "dn_pair_delta_max", 8.0))
                    dn_pair_snr_floor = (
                        float(getattr(args, "dn_pair_snr_floor_db"))
                        if getattr(args, "dn_pair_snr_floor_db", None) is not None
                        else float(snr_min_db)
                    )
                    dn_pair_snr_new_lo_raw = float(getattr(args, "dn_pair_snr_new_lo", -999.0))
                    dn_pair_snr_new_hi_raw = float(getattr(args, "dn_pair_snr_new_hi", 999.0))
                    dn_pair_snr_new_lo = dn_pair_snr_new_lo_raw if dn_pair_snr_new_lo_raw > -900.0 else None
                    dn_pair_snr_new_hi = dn_pair_snr_new_hi_raw if dn_pair_snr_new_hi_raw < 900.0 else None
                    x_low_dn, delta_dn = snr_path_degrade(
                        x_dn_ref,
                        snr_dn_ref,
                        delta_min=dn_pair_delta_min,
                        delta_max=dn_pair_delta_max,
                        snr_floor=dn_pair_snr_floor,
                        snr_target_min=dn_pair_snr_new_lo,
                        snr_target_max=dn_pair_snr_new_hi,
                    )
                    snr_new_dn = snr_dn_ref.float() - delta_dn.float()
                    snr_for_low = snr_new_dn if args.snr_mode == "known" else None
                    x_dn_low, _eta_low, _eta_cond_low = model.denoise_only(
                        x_low_dn, snr=snr_for_low, snr_mode=args.snr_mode
                    )
                    loss_dn = torch.mean(torch.abs(x_dn_low.float() - x_dn_ref.float()))
                    snr_for_hi = snr_dn_ref if args.snr_mode == "known" else None
                    x_dn_hi, _eta_hi, _eta_cond_hi = model.denoise_only(
                        x_dn_ref, snr=snr_for_hi, snr_mode=args.snr_mode
                    )
                    m_hi = high_snr_soft_mask(snr_dn_ref)
                    id_vec = torch.mean(torch.abs(x_dn_hi.float() - x_dn_ref.float()), dim=(1, 2))
                    loss_id = (m_hi * id_vec).sum() / torch.clamp(m_hi.sum(), min=1.0)
                    if (
                        float(lambda_feat) > 0.0
                        and hasattr(model, "early_features")
                        and bool(getattr(model, "has_feat_encoder", False))
                    ):
                        if bool(getattr(model, "denoiser_dual_path", False)):
                            feat_in_pred = torch.cat([x_low_dn.float(), x_dn_low.float()], dim=1)
                            feat_in_tgt = torch.cat([x_low_dn.float(), x_dn_ref.float()], dim=1)
                        else:
                            feat_in_pred = x_dn_low.float()
                            feat_in_tgt = x_dn_ref.float()
                        feat_pred = model.early_features(feat_in_pred)
                        feat_tgt = model.early_features(feat_in_tgt).detach()
                        feat_vec = torch.mean(torch.abs(feat_pred - feat_tgt), dim=1)
                        lfeat_mask = torch.ones_like(snr_dn_ref.float())
                        lfeat_src_lo = float(getattr(args, "lfeat_snr_lo", -999.0))
                        lfeat_src_hi = float(getattr(args, "lfeat_snr_hi", 999.0))
                        if lfeat_src_lo > -900.0 or lfeat_src_hi < 900.0:
                            lfeat_mask = lfeat_mask * (
                                (snr_dn_ref.float() >= lfeat_src_lo)
                                & (snr_dn_ref.float() <= lfeat_src_hi)
                            ).float()
                        lfeat_new_lo = float(getattr(args, "lfeat_snr_new_lo", -999.0))
                        lfeat_new_hi = float(getattr(args, "lfeat_snr_new_hi", 999.0))
                        if lfeat_new_lo > -900.0 or lfeat_new_hi < 900.0:
                            lfeat_mask = lfeat_mask * (
                                (snr_new_dn.float() >= lfeat_new_lo)
                                & (snr_new_dn.float() <= lfeat_new_hi)
                            ).float()
                        loss_feat = (feat_vec * lfeat_mask).sum() / torch.clamp(lfeat_mask.sum(), min=1.0)
                        lfeat_active_frac_batch = lfeat_mask.mean()
                    loss = (
                        loss
                        + float(getattr(args, "lambda_dn", 0.0)) * loss_dn
                        + float(getattr(args, "lambda_id", 0.0)) * loss_id
                        + float(lambda_feat) * loss_feat
                    )

                # --- External teacher KD (oracle teacher -> blind student) ---
                if (
                    teacher_model is not None
                    and (
                        float(lambda_kd_eff) > 0.0
                        or float(lambda_kd_denoise_eff) > 0.0
                        or float(lambda_kd_feat_eff) > 0.0
                    )
                ):
                    teacher_snr_mode = str(getattr(args, "kd_teacher_snr_mode", "known"))
                    teacher_snr_in = snr_aux if teacher_snr_mode == "known" else None
                    labels_kd = y if (not use_mixup or mixup_cls_only) else None
                    logits_teacher_kd: Optional[torch.Tensor] = None
                    teacher_prefilm_kd: Optional[torch.Tensor] = None

                    if float(lambda_kd_eff) > 0.0 or float(lambda_kd_feat_eff) > 0.0:
                        with torch.no_grad():
                            logits_teacher_kd, _, _ = teacher_model(
                                x_aux,
                                t,
                                snr=teacher_snr_in,
                                snr_mode=teacher_snr_mode,
                                group_mask=mask,
                            )
                            teacher_prefilm_kd = getattr(teacher_model, "_pooled_pre_film", None)

                    if float(lambda_kd_eff) > 0.0 and logits_teacher_kd is not None:
                        logits_student_kd = logits_teacher_base if logits_teacher_base is not None else logits
                        loss_kd, kd_active_frac_batch = kd_distillation_loss(
                            logits_student_kd,
                            logits_teacher_kd,
                            temperature=float(getattr(args, "kd_temp", 2.0)),
                            snr_db=snr_aux,
                            snr_lo=float(getattr(args, "kd_snr_lo", -999.0)),
                            snr_hi=float(getattr(args, "kd_snr_hi", 999.0)),
                            conf_thresh=float(getattr(args, "kd_conf_thresh", 0.0)),
                            labels=labels_kd,
                            correctness_filter=bool(getattr(args, "kd_correctness_filter", True)),
                            normalize_by_active=bool(getattr(args, "kd_normalize_by_active", False)),
                        )
                        kd_hi_scale = float(getattr(args, "kd_hi_preserve_scale", 0.0))
                        if kd_hi_scale > 0.0:
                            kd_hi_conf = float(getattr(args, "kd_hi_conf_thresh", -1.0))
                            if kd_hi_conf < 0.0:
                                kd_hi_conf = float(getattr(args, "kd_conf_thresh", 0.0))
                            loss_kd_hi, kd_hi_active_frac_batch = kd_distillation_loss(
                                logits_student_kd,
                                logits_teacher_kd,
                                temperature=float(getattr(args, "kd_temp", 2.0)),
                                snr_db=snr_aux,
                                snr_lo=float(getattr(args, "kd_hi_snr_lo", 10.0)),
                                snr_hi=float(getattr(args, "kd_hi_snr_hi", 18.0)),
                                conf_thresh=kd_hi_conf,
                                labels=labels_kd,
                                correctness_filter=bool(getattr(args, "kd_correctness_filter", True)),
                                normalize_by_active=bool(getattr(args, "kd_normalize_by_active", False)),
                            )
                            loss_kd = loss_kd + kd_hi_scale * loss_kd_hi
                        loss = loss + float(lambda_kd_eff) * loss_kd

                    if (
                        float(lambda_kd_denoise_eff) > 0.0
                        and args.arch == "cldnn"
                        and hasattr(model, "denoise_only")
                        and hasattr(teacher_model, "denoise_only")
                    ):
                        if x_aux.ndim == 4:
                            g_kd = x_aux.shape[1]
                            x_dn_kd = x_aux.reshape(-1, x_aux.shape[2], x_aux.shape[3])
                            snr_dn_kd = snr_aux.repeat_interleave(g_kd)
                        else:
                            x_dn_kd = x_aux
                            snr_dn_kd = snr_aux
                        snr_teacher_dn = snr_dn_kd if teacher_snr_mode == "known" else None
                        snr_student_dn = snr_dn_kd if args.snr_mode == "known" else None
                        with torch.no_grad():
                            x_dn_teacher, _, _ = teacher_model.denoise_only(
                                x_dn_kd, snr=snr_teacher_dn, snr_mode=teacher_snr_mode
                            )
                        x_dn_student, _, _ = model.denoise_only(
                            x_dn_kd, snr=snr_student_dn, snr_mode=args.snr_mode
                        )
                        dn_vec = torch.mean(torch.abs(x_dn_student.float() - x_dn_teacher.float()), dim=(1, 2))
                        dn_mask = (
                            (snr_dn_kd.float() >= float(getattr(args, "kd_denoise_snr_lo", -14.0)))
                            & (snr_dn_kd.float() <= float(getattr(args, "kd_denoise_snr_hi", -6.0)))
                        ).float()
                        loss_kd_denoise = (dn_vec * dn_mask).mean()
                        kd_denoise_active_frac_batch = dn_mask.mean()
                        loss = loss + float(lambda_kd_denoise_eff) * loss_kd_denoise

                    if (
                        float(lambda_kd_feat_eff) > 0.0
                        and student_prefilm_kd is not None
                        and teacher_prefilm_kd is not None
                        and student_prefilm_kd.shape == teacher_prefilm_kd.shape
                    ):
                        feat_student = F.normalize(student_prefilm_kd.float(), dim=1)
                        feat_teacher = F.normalize(teacher_prefilm_kd.detach().float(), dim=1)
                        feat_vec = torch.sum((feat_student - feat_teacher) ** 2, dim=1)
                        snr_feat_kd = snr_aux
                        if snr_feat_kd.shape[0] != feat_vec.shape[0] and x_aux.ndim == 4:
                            snr_feat_kd = snr_aux.repeat_interleave(x_aux.shape[1])
                        if snr_feat_kd.shape[0] == feat_vec.shape[0]:
                            feat_mask = (
                                (snr_feat_kd.float() >= float(getattr(args, "kd_feat_snr_lo", -14.0)))
                                & (snr_feat_kd.float() <= float(getattr(args, "kd_feat_snr_hi", -6.0)))
                            ).float()
                            loss_kd_feat = (feat_vec * feat_mask).mean()
                            kd_feat_active_frac_batch = feat_mask.mean()
                            loss = loss + float(lambda_kd_feat_eff) * loss_kd_feat

                # --- Supervised Contrastive Loss (SupCon) ---
                # Skip on mixup batches — labels are ambiguous after interpolation.
                if (
                    getattr(args, "supcon", False)
                    and args.arch != "dit"
                    and epoch >= int(getattr(args, "supcon_warmup", 0))
                    and not use_mixup
                    and hasattr(model, "proj_head")
                    and model.proj_head is not None
                ):
                    z_proj = model.proj_head(model._pooled_pre_film)
                    z_proj = F.normalize(z_proj, dim=1)
                    loss_supcon = supervised_contrastive_loss(
                        z_proj, y,
                        temperature=float(getattr(args, "supcon_temp", 0.07)),
                    )
                    loss = loss + float(getattr(args, "supcon_lambda", 0.1)) * loss_supcon

                # --- SNR-path consistency loss (core novelty) ---
                # Allow consistency when mixup is classification-only; the auxiliary
                # path still uses clean (physical) samples in that case.
                if (
                    getattr(args, "snr_consist", False)
                    and args.arch != "dit"
                    and epoch >= getattr(args, "snr_consist_warmup", 5)
                    and (not use_mixup or mixup_cls_only)
                ):
                    # Optionally recompute teacher logits with dropout OFF (eval mode)
                    logits_teacher = logits_teacher_base if logits_teacher_base is not None else logits
                    if bool(getattr(args, "snr_consist_teacher_eval", False)):
                        was_training = model.training
                        model.eval()
                        with torch.no_grad():
                            logits_teacher, _, _ = model(
                                x_aux,
                                t,
                                snr=snr_in_aux,
                                snr_mode=args.snr_mode,
                                group_mask=mask,
                                **extra_cls2dn,
                            )
                        if was_training:
                            model.train()

                    consist_delta_min: Union[float, torch.Tensor] = float(getattr(args, "snr_consist_delta_min", 2.0))
                    consist_delta_max: Union[float, torch.Tensor] = float(getattr(args, "snr_consist_delta_max", 8.0))
                    if bool(getattr(args, "snr_consist_adaptive_delta", False)):
                        s_aux = snr_aux.float()
                        dmin_t = torch.full_like(s_aux, float(getattr(args, "snr_consist_delta_min", 2.0)))
                        dmax_t = torch.full_like(s_aux, float(getattr(args, "snr_consist_delta_max", 8.0)))
                        low_mask = s_aux <= float(getattr(args, "snr_consist_low_snr_thresh", -6.0))
                        dmin_t = torch.where(
                            low_mask,
                            torch.full_like(dmin_t, float(getattr(args, "snr_consist_low_delta_min", 2.0))),
                            dmin_t,
                        )
                        dmax_t = torch.where(
                            low_mask,
                            torch.full_like(dmax_t, float(getattr(args, "snr_consist_low_delta_max", 4.0))),
                            dmax_t,
                        )
                        consist_delta_min = dmin_t
                        consist_delta_max = dmax_t
                    x_low, _delta = snr_path_degrade(
                        x_aux, snr_aux,
                        delta_min=consist_delta_min,
                        delta_max=consist_delta_max,
                        snr_floor=float(snr_min_db),
                    )
                    snr_new = snr_aux.float() - _delta.float()
                    logits_low, _, _ = model(
                        x_low,
                        t,
                        snr=snr_in_aux,
                        snr_mode=args.snr_mode,
                        group_mask=mask,
                        **extra_cls2dn,
                    )
                    loss_consist = snr_consistency_loss(
                        logits_teacher, logits_low,
                        temperature=float(getattr(args, "snr_consist_temp", 2.0)),
                        snr_db=snr_aux,
                        snr_lo=float(getattr(args, "snr_consist_snr_lo", -999.0)),
                        snr_hi=float(getattr(args, "snr_consist_snr_hi", 999.0)),
                        conf_thresh=float(getattr(args, "snr_consist_conf_thresh", 0.0)),
                        snr_new_db=snr_new,
                        snr_new_lo=float(getattr(args, "snr_consist_snr_new_lo", -999.0)),
                        snr_new_hi=float(getattr(args, "snr_consist_snr_new_hi", 999.0)),
                    )
                    consist_lambda = float(getattr(args, "snr_consist_lambda", 1.0))
                    ramp_epochs = int(getattr(args, "snr_consist_ramp", 0))
                    warmup_ep = int(getattr(args, "snr_consist_warmup", 5))
                    if ramp_epochs and ramp_epochs > 0:
                        ramp = (epoch - warmup_ep + 1) / float(max(1, ramp_epochs))
                        ramp = max(0.0, min(1.0, ramp))
                    else:
                        ramp = 1.0
                    loss = loss + consist_lambda * ramp * loss_consist

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            scheduler.step()
            global_step += 1
            if ema is not None and global_step >= args.ema_start and (global_step % max(1, args.ema_every) == 0):
                ema.update(model)

            logits_for_acc_batch = logits_for_acc if logits_for_acc is not None else logits
            batch_size = x_clean.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_loss_feat += float(loss_feat.detach().item()) * batch_size
            epoch_lfeat_active += float(lfeat_active_frac_batch.detach().item()) * batch_size
            epoch_loss_kd += float(loss_kd.detach().item()) * batch_size
            epoch_kd_active += float(kd_active_frac_batch.detach().item()) * batch_size
            epoch_loss_kd_hi += float(loss_kd_hi.detach().item()) * batch_size
            epoch_kd_hi_active += float(kd_hi_active_frac_batch.detach().item()) * batch_size
            epoch_loss_kd_denoise += float(loss_kd_denoise.detach().item()) * batch_size
            epoch_kd_denoise_active += float(kd_denoise_active_frac_batch.detach().item()) * batch_size
            epoch_loss_kd_feat += float(loss_kd_feat.detach().item()) * batch_size
            epoch_kd_feat_active += float(kd_feat_active_frac_batch.detach().item()) * batch_size
            epoch_correct += (logits_for_acc_batch.argmax(dim=1) == y).sum().item()
            epoch_total += batch_size
            if epoch_total > 0:
                progress.set_postfix(
                    train_loss=f"{epoch_loss / epoch_total:.4f}",
                    train_acc=f"{epoch_correct / epoch_total:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        train_loss = epoch_loss / max(1, epoch_total)
        train_loss_feat = epoch_loss_feat / max(1, epoch_total)
        train_lfeat_active_frac = epoch_lfeat_active / max(1, epoch_total)
        train_loss_kd = epoch_loss_kd / max(1, epoch_total)
        train_kd_active_frac = epoch_kd_active / max(1, epoch_total)
        train_loss_kd_hi = epoch_loss_kd_hi / max(1, epoch_total)
        train_kd_hi_active_frac = epoch_kd_hi_active / max(1, epoch_total)
        train_loss_kd_denoise = epoch_loss_kd_denoise / max(1, epoch_total)
        train_kd_denoise_active_frac = epoch_kd_denoise_active / max(1, epoch_total)
        train_loss_kd_feat = epoch_loss_kd_feat / max(1, epoch_total)
        train_kd_feat_active_frac = epoch_kd_feat_active / max(1, epoch_total)
        train_acc = epoch_correct / max(1, epoch_total)
        train_acc_clean = None
        if train_eval_loader is not None:
            train_acc_clean = evaluate_subset(
                model,
                train_eval_loader,
                device,
                t_eval=0,
                snr_mode=args.snr_mode,
                amp=args.amp,
                max_batches=args.train_eval_batches,
            )

        eval_bypass = bool(getattr(args, "cldnn_denoiser_bypass_eval", False))
        prev_eval_bypass = None
        if hasattr(model, "force_denoiser_bypass"):
            prev_eval_bypass = bool(getattr(model, "force_denoiser_bypass", False))
            model.force_denoiser_bypass = eval_bypass  # type: ignore[attr-defined]

        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
            val_acc, _, val_acc_by_snr, val_summary = evaluate(
                model,
                val_loader,
                device,
                args.t_eval,
                args.snr_mode,
                amp=args.amp,
                low_snr_lo=float(getattr(args, "report_low_snr_lo", -14.0)),
                low_snr_hi=float(getattr(args, "report_low_snr_hi", -6.0)),
            )
            ema.restore(model)
        else:
            val_acc, _, val_acc_by_snr, val_summary = evaluate(
                model,
                val_loader,
                device,
                args.t_eval,
                args.snr_mode,
                amp=args.amp,
                low_snr_lo=float(getattr(args, "report_low_snr_lo", -14.0)),
                low_snr_hi=float(getattr(args, "report_low_snr_hi", -6.0)),
            )

        noise_calib: Dict[str, object] = {}
        if args.arch == "cldnn" and (
            bool(getattr(args, "cldnn_noise_cond", False))
            or bool(getattr(args, "cldnn_denoiser", False))
        ):
            if ema is not None:
                ema.store(model)
                ema.copy_to(model)
                noise_calib = evaluate_eta_calibration(
                    model,
                    val_loader,
                    device,
                    args.t_eval,
                    args.snr_mode,
                    amp=args.amp,
                    rho_min=float(getattr(args, "noise_rho_min", 1e-4)),
                    rho_max=float(getattr(args, "noise_rho_max", 1.0 - 1e-4)),
                    eta_min=float(getattr(args, "noise_eta_min", -8.0)),
                    eta_max=float(getattr(args, "noise_eta_max", 5.5)),
                )
                ema.restore(model)
            else:
                noise_calib = evaluate_eta_calibration(
                    model,
                    val_loader,
                    device,
                    args.t_eval,
                    args.snr_mode,
                    amp=args.amp,
                    rho_min=float(getattr(args, "noise_rho_min", 1e-4)),
                    rho_max=float(getattr(args, "noise_rho_max", 1.0 - 1e-4)),
                    eta_min=float(getattr(args, "noise_eta_min", -8.0)),
                    eta_max=float(getattr(args, "noise_eta_max", 5.5)),
                )

        if prev_eval_bypass is not None:
            model.force_denoiser_bypass = prev_eval_bypass  # type: ignore[attr-defined]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_acc_clean": train_acc_clean,
            "val_acc": val_acc,
            "val_macro_acc": float(val_summary.get("macro_acc", 0.0)),
            "val_macro_f1": float(val_summary.get("macro_f1", 0.0)),
            "val_low_macro_acc": float(val_summary.get("low_macro_acc", 0.0)),
            "val_low_macro_f1": float(val_summary.get("low_macro_f1", 0.0)),
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": time.time() - start_time,
            "lambda_diff": lambda_diff,
            "lambda_noise": float(getattr(args, "lambda_noise", 0.0)),
            "lambda_dn": float(getattr(args, "lambda_dn", 0.0)),
            "lambda_id": float(getattr(args, "lambda_id", 0.0)),
            "lambda_feat": float(lambda_feat),
            "lambda_kd": float(lambda_kd_eff),
            "lambda_kd_denoise": float(lambda_kd_denoise_eff),
            "lambda_kd_feat": float(lambda_kd_feat_eff),
            "train_loss_feat": float(train_loss_feat),
            "train_lfeat_active_frac": float(train_lfeat_active_frac),
            "train_loss_kd": float(train_loss_kd),
            "train_kd_active_frac": float(train_kd_active_frac),
            "train_loss_kd_hi": float(train_loss_kd_hi),
            "train_kd_hi_active_frac": float(train_kd_hi_active_frac),
            "train_loss_kd_denoise": float(train_loss_kd_denoise),
            "train_kd_denoise_active_frac": float(train_kd_denoise_active_frac),
            "train_loss_kd_feat": float(train_loss_kd_feat),
            "train_kd_feat_active_frac": float(train_kd_feat_active_frac),
            "cls2dn_scale": float(cls2dn_scale),
            "p_clean": p_clean,
            "t_schedule": args.t_schedule,
            "snr_floor_db": args.snr_floor_db,
            "snr_cap_max_db": args.snr_cap_max_db,
            "low_snr_boost": args.low_snr_boost,
            "lr_decay_start_epoch": getattr(args, "lr_decay_start_epoch", 0),
            "curriculum_soft": bool(getattr(args, "curriculum_soft", False)),
            "curriculum_soft_low_weight": float(getattr(args, "curriculum_soft_low_weight", 0.1)),
            "mixup_snr_min": getattr(args, "mixup_snr_min", None),
            "mixup_cls_only": bool(getattr(args, "mixup_cls_only", True)),
            "cldnn_noise_cond": bool(getattr(args, "cldnn_noise_cond", False)),
            "cldnn_denoiser": bool(getattr(args, "cldnn_denoiser", False)),
        }
        if proxy_fit_info:
            record.update(proxy_fit_info)
        if noise_calib:
            record["eta_pearson"] = float(noise_calib.get("eta_pearson", 0.0))
            record["eta_spearman"] = float(noise_calib.get("eta_spearman", 0.0))
        write_jsonl(metrics_path, record)

        improved = val_acc > (best_val + float(args.early_stop_min_delta))
        if improved:
            best_val = val_acc
            best_epoch = epoch
            epochs_no_improve = 0
            save_checkpoint(
                os.path.join(args.out_dir, "best.pt"),
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                args,
                ema=ema,
            )
        else:
            epochs_no_improve += 1

        save_checkpoint(
            os.path.join(args.out_dir, "last.pt"),
            model,
            optimizer,
            scheduler,
            epoch,
            global_step,
            args,
            ema=ema,
        )

        if val_acc_by_snr:
            with open(os.path.join(args.out_dir, "val_acc_by_snr.json"), "w", encoding="utf-8") as f:
                json.dump(val_acc_by_snr, f, indent=2)
        if noise_calib and noise_calib.get("eta_by_snr"):
            with open(os.path.join(args.out_dir, "eta_calibration_by_snr.json"), "w", encoding="utf-8") as f:
                json.dump(noise_calib["eta_by_snr"], f, indent=2)

        tqdm.write(
            f"Epoch {epoch + 1}/{args.epochs} summary | "
            f"train_acc(noised)={train_acc:.4f} "
            f"train_acc_clean~={train_acc_clean if train_acc_clean is not None else 'NA'} "
            f"val_acc={val_acc:.4f}"
        )

        if args.early_stop_patience and args.early_stop_patience > 0:
            if epochs_no_improve >= int(args.early_stop_patience):
                tqdm.write(
                    f"Early stopping at epoch {epoch + 1}: no val_acc improvement for "
                    f"{epochs_no_improve} epochs (best={best_val:.4f} at epoch {best_epoch + 1 if best_epoch is not None else 'NA'})."
                )
                break

    # Evaluate best checkpoint on test set.
    best_path = os.path.join(args.out_dir, "best.pt")
    if os.path.exists(best_path):
        ckpt = load_checkpoint(best_path, model, ema=ema)
        if ema is not None and ckpt.get("ema") is not None:
            ema.copy_to(model)
    if hasattr(model, "force_denoiser_bypass"):
        model.force_denoiser_bypass = bool(getattr(args, "cldnn_denoiser_bypass_eval", False))  # type: ignore[attr-defined]
    test_acc, _, test_acc_by_snr, test_summary = evaluate(
        model,
        test_loader,
        device,
        args.t_eval,
        args.snr_mode,
        amp=args.amp,
        low_snr_lo=float(getattr(args, "report_low_snr_lo", -14.0)),
        low_snr_hi=float(getattr(args, "report_low_snr_hi", -6.0)),
    )
    dyn = None
    if args.dynamic_k_eval:
        dyn_k_max = int(args.dynamic_k_max) if args.dynamic_k_max is not None else int(args.k_max if args.k_max is not None else args.group_k)
        dyn_acc, _dyn_total, dyn_by_snr, avg_k = evaluate_dynamic_k(
            model,
            test_loader,
            device,
            args.t_eval,
            args.snr_mode,
            amp=args.amp,
            k_start=args.dynamic_k_start,
            k_step=args.dynamic_k_step,
            k_max=dyn_k_max,
            conf_thresh=args.dynamic_conf_thresh,
        )
        dyn = {"dynamic_test_acc": dyn_acc, "dynamic_avg_k": avg_k, "dynamic_test_acc_by_snr": dyn_by_snr}
    with open(os.path.join(args.out_dir, "test_acc_by_snr.json"), "w", encoding="utf-8") as f:
        json.dump(test_acc_by_snr, f, indent=2)
    with open(os.path.join(args.out_dir, "test_macro_summary.json"), "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)
    write_jsonl(
        metrics_path,
        {
            "epoch": best_epoch if best_epoch is not None else args.epochs,
            "test_acc": test_acc,
            "test_macro_acc": float(test_summary.get("macro_acc", 0.0)),
            "test_macro_f1": float(test_summary.get("macro_f1", 0.0)),
            "test_low_macro_acc": float(test_summary.get("low_macro_acc", 0.0)),
            "test_low_macro_f1": float(test_summary.get("low_macro_f1", 0.0)),
            "best_val_acc": best_val,
            **(dyn if dyn is not None else {}),
        },
    )


def run_eval(args: argparse.Namespace) -> None:
    apply_preset(args)
    if bool(getattr(args, "cldnn_snr_cond", False)) and bool(getattr(args, "cldnn_noise_cond", False)):
        raise ValueError("Use only one conditioning path: --cldnn-snr-cond OR --cldnn-noise-cond.")
    if args.arch != "cldnn" and (
        bool(getattr(args, "cldnn_noise_cond", False)) or bool(getattr(args, "cldnn_denoiser", False))
    ):
        raise ValueError("Noise/denoiser options are currently supported only for --arch cldnn.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, mods, snrs, seq_len = build_loaders(args, device)
    snr_min_db = float(min(snrs)) if snrs else -20.0
    snr_max_db = float(max(snrs)) if snrs else 18.0

    if args.arch == "dit":
        model = DiffusionAMC(
            num_classes=len(mods),
            seq_len=seq_len,
            patch_size=args.patch_size,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_ratio=args.mlp_ratio,
            dropout=args.dropout,
            snr_scale=args.snr_scale,
            stem_channels=args.stem_channels,
            stem_layers=args.stem_layers,
            group_pool=args.group_pool,
        ).to(device)
    elif args.arch == "multiview":
        model = MultiViewCLDNNAMC(
            num_classes=len(mods),
            seq_len=seq_len,
            conv_channels=int(args.cldnn_conv_ch),
            merge_channels=int(args.cldnn_merge_ch),
            lstm_hidden=int(args.cldnn_lstm_hidden),
            lstm_layers=int(args.cldnn_lstm_layers),
            bidirectional=bool(args.cldnn_bidir),
            dropout=float(args.dropout),
            pool=str(args.cldnn_pool),
            snr_cond=bool(args.cldnn_snr_cond),
            snr_loss_detach_backbone=bool(getattr(args, "snr_loss_detach_backbone", False)),
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            cls_hidden=int(getattr(args, 'cldnn_cls_hidden', 0)),
            stft_nfft=int(getattr(args, 'stft_nfft', 64)),
            stft_hop=int(getattr(args, 'stft_hop', 8)),
            stft_channels=int(getattr(args, 'stft_channels', 64)),
            cross_view_heads=int(getattr(args, 'cross_view_heads', 4)),
            snr_gate=bool(getattr(args, 'snr_gate', False)),
        ).to(device)
    else:
        model = CLDNNAMC(
            num_classes=len(mods),
            seq_len=seq_len,
            conv_channels=int(args.cldnn_conv_ch),
            merge_channels=int(args.cldnn_merge_ch),
            lstm_hidden=int(args.cldnn_lstm_hidden),
            lstm_layers=int(args.cldnn_lstm_layers),
            bidirectional=bool(args.cldnn_bidir),
            dropout=float(args.dropout),
            pool=str(args.cldnn_pool),
            snr_cond=bool(args.cldnn_snr_cond),
            noise_cond=bool(getattr(args, "cldnn_noise_cond", False)),
            snr_loss_detach_backbone=bool(getattr(args, "snr_loss_detach_backbone", False)),
            snr_min_db=snr_min_db,
            snr_max_db=snr_max_db,
            noise_eta_min=float(getattr(args, "noise_eta_min", -8.0)),
            noise_eta_max=float(getattr(args, "noise_eta_max", 5.5)),
            denoiser=bool(getattr(args, "cldnn_denoiser", False)),
            denoiser_dual_path=bool(getattr(args, "cldnn_denoiser_dual_path", False)),
            denoiser_base_channels=int(getattr(args, "cldnn_denoiser_base_ch", 32)),
            denoiser_dropout=float(getattr(args, "cldnn_denoiser_dropout", 0.0)),
            denoiser_soft_high_snr_blend=bool(getattr(args, "cldnn_denoiser_soft_hi_blend", False)),
            noise_head_hidden=int(getattr(args, "noise_head_hidden", 32)),
            expert_features=bool(getattr(args, 'cldnn_expert_features', False)),
            expert_channels=int(getattr(args, 'cldnn_expert_ch', 64)),
            expert_stacf_window=int(getattr(args, "cldnn_expert_stacf_win", 0)),
            expert_v2=bool(getattr(args, "cldnn_expert_v2", False)),
            expert_corr_norm_eps=float(getattr(args, "cldnn_expert_corr_eps", 1e-6)),
            expert_eta_gate=bool(getattr(args, "cldnn_expert_eta_gate", False)),
            expert_eta_gate_center=float(getattr(args, "cldnn_expert_eta_gate_center", 0.8)),
            expert_eta_gate_tau=float(getattr(args, "cldnn_expert_eta_gate_tau", 0.7)),
            expert_eta_gate_min=float(getattr(args, "cldnn_expert_eta_gate_min", 0.0)),
            expert_eta_gate_max=float(getattr(args, "cldnn_expert_eta_gate_max", 1.0)),
            expert_use_cyclo_stats=bool(getattr(args, "cldnn_cyclo_stats", True)),
            raw_low_snr_drop_prob=float(getattr(args, "cldnn_raw_low_snr_drop_prob", 0.0)),
            raw_low_snr_drop_gate=str(getattr(args, "cldnn_raw_low_snr_drop_gate", "auto")),
            raw_low_snr_drop_eta_thresh=float(getattr(args, "cldnn_raw_low_snr_drop_eta_thresh", 1.0)),
            raw_low_snr_drop_snr_thresh=float(getattr(args, "cldnn_raw_low_snr_drop_snr_thresh", -6.0)),
            raw_low_snr_drop_min_scale=float(getattr(args, "cldnn_raw_low_snr_drop_min_scale", 0.0)),
            raw_low_snr_drop_max_scale=float(getattr(args, "cldnn_raw_low_snr_drop_max_scale", 0.0)),
            raw_low_snr_drop_prob_lo=float(getattr(args, "cldnn_raw_low_snr_drop_prob_lo", -1.0)),
            raw_low_snr_drop_prob_mid=float(getattr(args, "cldnn_raw_low_snr_drop_prob_mid", -1.0)),
            raw_low_snr_drop_prob_hi=float(getattr(args, "cldnn_raw_low_snr_drop_prob_hi", -1.0)),
            raw_low_snr_drop_snr_lo=float(getattr(args, "cldnn_raw_low_snr_drop_snr_lo", -10.0)),
            raw_low_snr_drop_snr_mid=float(getattr(args, "cldnn_raw_low_snr_drop_snr_mid", -6.0)),
            cls_hidden=int(getattr(args, 'cldnn_cls_hidden', 0)),
            supcon_proj_dim=int(getattr(args, 'supcon_proj_dim', 0)) if getattr(args, 'supcon', False) else 0,
        ).to(device)

    if args.ckpt is None:
        raise ValueError("Provide --ckpt for eval-only mode.")
    ema = EMA.create(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    ckpt = load_checkpoint(args.ckpt, model, ema=ema)
    if ema is not None and ckpt.get("ema") is not None:
        ema.copy_to(model)
    if hasattr(model, "force_denoiser_bypass"):
        model.force_denoiser_bypass = bool(getattr(args, "cldnn_denoiser_bypass_eval", False))  # type: ignore[attr-defined]

    test_acc, _, test_acc_by_snr, test_summary = evaluate(
        model,
        test_loader,
        device,
        args.t_eval,
        args.snr_mode,
        amp=args.amp,
        low_snr_lo=float(getattr(args, "report_low_snr_lo", -14.0)),
        low_snr_hi=float(getattr(args, "report_low_snr_hi", -6.0)),
    )
    print(f"Test acc: {test_acc:.4f}")
    print("Test acc by SNR:", test_acc_by_snr)
    print(
        "Test macro summary:",
        {
            "macro_acc": float(test_summary.get("macro_acc", 0.0)),
            "macro_f1": float(test_summary.get("macro_f1", 0.0)),
            "low_macro_acc": float(test_summary.get("low_macro_acc", 0.0)),
            "low_macro_f1": float(test_summary.get("low_macro_f1", 0.0)),
        },
    )


def _resolve_dataset_defaults(args: argparse.Namespace) -> None:
    """Fill in None-valued args with dataset-aware defaults."""
    is_2018 = str(getattr(args, "dataset", "rml2016a")) == "rml2018a"

    # Per-bucket split sizes: RML2016.10a has ~1000/bucket, RML2018.01A has 4096/bucket.
    if args.train_per is None:
        args.train_per = 3200 if is_2018 else 600
    if args.val_per is None:
        args.val_per = 500 if is_2018 else 200

    # SNR cap: default to dataset max (18 for RML2016.10a, 30 for RML2018.01A).
    if args.snr_cap_max_db is None:
        args.snr_cap_max_db = 30.0 if is_2018 else 18.0


def main() -> None:
    args = parse_args()
    _resolve_dataset_defaults(args)
    if args.lr is None:
        args.lr = PRESETS[args.preset]["lr"]
    if args.eval_only:
        run_eval(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
