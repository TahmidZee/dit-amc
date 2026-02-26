import argparse
import copy
import json
import math
import os
import random
import time
from contextlib import nullcontext
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
    RML2016aContrastivePairDataset,
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


def get_lambda_dn_cls(epoch: int, args: argparse.Namespace) -> float:
    """
    Optional warmup/ramp for task-aware dn-diff classification loss.
    """
    target = float(getattr(args, "lambda_dn_cls", 0.0))
    if target <= 0.0:
        return 0.0
    warmup = int(max(0, getattr(args, "dn_diff_cls_warmup", 0)))
    if epoch < warmup:
        return 0.0
    ramp_epochs = int(max(0, getattr(args, "dn_diff_cls_ramp", 0)))
    if ramp_epochs <= 0:
        return target
    progress = (epoch - warmup + 1) / float(max(1, ramp_epochs))
    progress = min(1.0, max(0.0, progress))
    return target * progress


def get_lambda_dn_diff(epoch: int, args: argparse.Namespace) -> float:
    """
    Optional warmup + decay schedule for dn-diff reconstruction objective.

    Before warmup: keep base lambda.
    After warmup: linearly move base lambda toward base*final_scale over ramp epochs.
    """
    base = float(getattr(args, "lambda_dn_diff", 1.0))
    if base <= 0.0:
        return 0.0
    warmup = int(max(0, getattr(args, "dn_diff_diff_warmup", 0)))
    final_scale = float(getattr(args, "dn_diff_diff_final_scale", 1.0))
    if epoch < warmup:
        return base
    ramp_epochs = int(max(0, getattr(args, "dn_diff_diff_ramp", 0)))
    if ramp_epochs <= 0:
        return base * final_scale
    progress = (epoch - warmup + 1) / float(max(1, ramp_epochs))
    progress = min(1.0, max(0.0, progress))
    scale = 1.0 + (final_scale - 1.0) * progress
    return base * scale


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


def snr_weighted_ce_weights(
    snr_db: torch.Tensor,
    snr_min_db: float,
    snr_max_db: float,
    scale: float,
    max_weight: float,
) -> torch.Tensor:
    """
    Build capped, normalized per-sample CE weights from SNR.

    - Lower SNR => larger weight.
    - Weights are capped for stability.
    - Batch mean is normalized to ~1.0 to avoid implicit LR changes.
    """
    snr_f = snr_db.float()
    s_min = float(min(snr_min_db, snr_max_db))
    s_max = float(max(snr_min_db, snr_max_db))
    denom = max(1e-6, s_max - s_min)
    snr_clamped = torch.clamp(snr_f, min=s_min, max=s_max)
    low_frac = torch.clamp((s_max - snr_clamped) / denom, min=0.0, max=1.0)
    w = 1.0 + float(scale) * low_frac
    w = torch.clamp(w, min=0.0, max=float(max_weight))
    w_mean = torch.clamp(w.mean(), min=1e-6)
    return w / w_mean


def compute_moe_regularizers(
    gate: Optional[torch.Tensor],
    balance_lambda: float,
    specialize_lambda: float,
    entropy_warm_lambda: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    MoE regularizers from gate probabilities:
      - balance: variance of per-expert average load (minimize for balanced usage)
      - specialize: entropy of per-sample gate distribution (minimize for peaked routing)
    """
    if gate is None or gate.ndim != 2:
        z = torch.tensor(0.0)
        return z, z, z, None
    g = gate.float()
    load = g.mean(dim=0)  # (E,)
    loss_bal = torch.tensor(0.0, device=g.device, dtype=g.dtype)
    loss_spec = torch.tensor(0.0, device=g.device, dtype=g.dtype)
    loss_ent_warm = torch.tensor(0.0, device=g.device, dtype=g.dtype)
    ent = -(torch.clamp(g, min=1e-8) * torch.log(torch.clamp(g, min=1e-8))).sum(dim=1).mean()
    if float(balance_lambda) > 0.0:
        loss_bal = float(balance_lambda) * torch.var(load, unbiased=False)
    if float(specialize_lambda) > 0.0:
        # Minimize entropy => more peaked routing.
        loss_spec = float(specialize_lambda) * ent
    if float(entropy_warm_lambda) > 0.0:
        # Early anti-collapse guard: maximize entropy (uniformer routing) before specialization.
        loss_ent_warm = -float(entropy_warm_lambda) * ent
    return loss_bal, loss_spec, loss_ent_warm, load.detach()


def _mixup_or_standard_ce(
    logits: torch.Tensor,
    y: torch.Tensor,
    use_mixup: bool,
    y_a: Optional[torch.Tensor],
    y_b: Optional[torch.Tensor],
    lam: Union[float, torch.Tensor],
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    if use_mixup and y_a is not None and y_b is not None:
        ce_a = F.cross_entropy(logits, y_a, reduction="none", label_smoothing=label_smoothing)
        ce_b = F.cross_entropy(logits, y_b, reduction="none", label_smoothing=label_smoothing)
        if isinstance(lam, torch.Tensor):
            lam_t = lam.to(device=logits.device, dtype=logits.dtype).view(-1)
        else:
            lam_t = torch.full(
                (logits.shape[0],),
                float(lam),
                device=logits.device,
                dtype=logits.dtype,
            )
        return lam_t * ce_a + (1.0 - lam_t) * ce_b
    return F.cross_entropy(logits, y, reduction="none", label_smoothing=label_smoothing)


def compute_moe_head_specialization_losses(
    logits_experts: Optional[torch.Tensor],
    y: torch.Tensor,
    snr_db: torch.Tensor,
    curriculum_mask: torch.Tensor,
    use_mixup: bool,
    y_a: Optional[torch.Tensor],
    y_b: Optional[torch.Tensor],
    lam: Union[float, torch.Tensor],
    low_head_idx: int,
    high_head_idx: int,
    low_lambda: float,
    high_lambda: float,
    low_snr_lo: float,
    low_snr_hi: float,
    high_snr_lo: float,
    high_snr_hi: float,
    label_smoothing: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if logits_experts is None or logits_experts.ndim != 3:
        z = torch.tensor(0.0, device=snr_db.device)
        return z, z, z, z
    n_experts = int(logits_experts.shape[1])
    if n_experts < 2:
        z = torch.tensor(0.0, device=logits_experts.device, dtype=logits_experts.dtype)
        return z, z, z, z
    if low_head_idx < 0 or high_head_idx < 0 or low_head_idx >= n_experts or high_head_idx >= n_experts:
        z = torch.tensor(0.0, device=logits_experts.device, dtype=logits_experts.dtype)
        return z, z, z, z

    snr_f = snr_db.float()
    curr = curriculum_mask.float()
    ce_low = _mixup_or_standard_ce(
        logits_experts[:, low_head_idx, :],
        y=y,
        use_mixup=use_mixup,
        y_a=y_a,
        y_b=y_b,
        lam=lam,
        label_smoothing=label_smoothing,
    )
    ce_high = _mixup_or_standard_ce(
        logits_experts[:, high_head_idx, :],
        y=y,
        use_mixup=use_mixup,
        y_a=y_a,
        y_b=y_b,
        lam=lam,
        label_smoothing=label_smoothing,
    )

    low_mask = ((snr_f >= float(low_snr_lo)) & (snr_f <= float(low_snr_hi))).float() * curr
    high_mask = ((snr_f >= float(high_snr_lo)) & (snr_f <= float(high_snr_hi))).float() * curr
    low_denom = torch.clamp(low_mask.sum(), min=1.0)
    high_denom = torch.clamp(high_mask.sum(), min=1.0)
    loss_low = torch.tensor(0.0, device=logits_experts.device, dtype=logits_experts.dtype)
    loss_high = torch.tensor(0.0, device=logits_experts.device, dtype=logits_experts.dtype)
    if float(low_lambda) > 0.0:
        loss_low = float(low_lambda) * (ce_low * low_mask).sum() / low_denom
    if float(high_lambda) > 0.0:
        loss_high = float(high_lambda) * (ce_high * high_mask).sum() / high_denom
    low_frac = low_mask.mean()
    high_frac = high_mask.mean()
    return loss_low, loss_high, low_frac, high_frac


def compute_moe_diversity_loss(
    logits_experts: Optional[torch.Tensor],
    diversity_lambda: float,
) -> torch.Tensor:
    if logits_experts is None or logits_experts.ndim != 3:
        return torch.tensor(0.0)
    if float(diversity_lambda) <= 0.0:
        return torch.tensor(0.0, device=logits_experts.device, dtype=logits_experts.dtype)
    n_experts = int(logits_experts.shape[1])
    if n_experts < 2:
        return torch.tensor(0.0, device=logits_experts.device, dtype=logits_experts.dtype)
    z = F.normalize(logits_experts.float(), dim=-1)
    sims: List[torch.Tensor] = []
    for i in range(n_experts):
        for j in range(i + 1, n_experts):
            sims.append(F.cosine_similarity(z[:, i, :], z[:, j, :], dim=1).mean())
    if not sims:
        return torch.tensor(0.0, device=logits_experts.device, dtype=logits_experts.dtype)
    return float(diversity_lambda) * torch.stack(sims).mean().to(dtype=logits_experts.dtype)


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


class MoCoProjectionMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(in_dim), int(hidden_dim)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def moco_momentum_update(query_module: torch.nn.Module, key_module: torch.nn.Module, momentum: float) -> None:
    m = float(momentum)
    for p_q, p_k in zip(query_module.parameters(), key_module.parameters()):
        p_k.data.mul_(m).add_(p_q.data, alpha=1.0 - m)
    # Keep BN/LN buffers aligned to avoid stale key statistics in single-process mode.
    for b_q, b_k in zip(query_module.buffers(), key_module.buffers()):
        b_k.copy_(b_q)


@torch.no_grad()
def moco_enqueue(queue: torch.Tensor, ptr: int, keys: torch.Tensor) -> int:
    if queue.ndim != 2 or keys.ndim != 2:
        raise ValueError("moco_enqueue expects 2D queue and keys tensors.")
    if queue.shape[1] != keys.shape[1]:
        raise ValueError(f"Queue/key dim mismatch: queue={tuple(queue.shape)} keys={tuple(keys.shape)}")

    qsz = int(queue.shape[0])
    if qsz <= 0:
        raise ValueError("MoCo queue size must be > 0.")

    k = F.normalize(keys.detach(), dim=1)
    bsz = int(k.shape[0])
    if bsz <= 0:
        return int(ptr) % qsz

    if bsz >= qsz:
        queue.copy_(k[-qsz:])
        return 0

    ptr = int(ptr) % qsz
    end = ptr + bsz
    if end <= qsz:
        queue[ptr:end] = k
    else:
        first = qsz - ptr
        queue[ptr:] = k[:first]
        queue[: end - qsz] = k[first:]
    return (ptr + bsz) % qsz


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
        help="Model architecture. dit=DiffusionAMC. cldnn=CNN+temporal backbone (LSTM/TCN/ResNet1D). multiview=IQ+STFT dual-branch with cross-view attention.",
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
    parser.add_argument(
        "--cldnn-backbone",
        type=str,
        choices=["lstm", "tcn", "resnet1d"],
        default="lstm",
        help="Temporal backbone for CLDNN classifier path.",
    )
    parser.add_argument("--cldnn-tcn-levels", type=int, default=6, help="Number of dilated residual blocks in TCN backbone.")
    parser.add_argument("--cldnn-tcn-channels", type=int, default=128, help="Hidden channels in TCN backbone.")
    parser.add_argument("--cldnn-tcn-kernel", type=int, default=3, help="Kernel size for TCN residual convolutions (odd).")
    parser.add_argument("--cldnn-tcn-dilation-base", type=int, default=2, help="Dilation growth base for TCN levels.")
    parser.add_argument("--cldnn-tcn-dropout", type=float, default=0.15, help="Dropout used inside TCN residual blocks.")
    parser.add_argument("--cldnn-resnet-blocks", type=int, default=8, help="Number of residual blocks in ResNet1D backbone.")
    parser.add_argument("--cldnn-resnet-channels", type=int, default=128, help="Hidden channels in ResNet1D backbone.")
    parser.add_argument("--cldnn-resnet-kernel", type=int, default=5, help="Kernel size for ResNet1D convolutions (odd).")
    parser.add_argument("--cldnn-resnet-dilation-cycle", type=int, default=4, help="Dilation cycle length for ResNet1D blocks.")
    parser.add_argument("--cldnn-resnet-dropout", type=float, default=0.15, help="Dropout used inside ResNet1D residual blocks.")
    parser.add_argument("--cldnn-pool", type=str, default="attn", choices=["attn", "last", "mean"], help="Temporal pooling over backbone sequence outputs.")
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
    parser.add_argument("--moe-n-experts", type=int, default=1, help="Number of classifier experts (1 disables MoE).")
    parser.add_argument(
        "--moe-gate-type",
        type=str,
        choices=["eta-sigmoid", "learned", "hard"],
        default="eta-sigmoid",
        help="MoE router type: eta-sigmoid (interpretable), learned (MLP), or hard (top-1).",
    )
    parser.add_argument("--moe-gate-center", type=float, default=0.5, help="Eta center for MoE eta-sigmoid gating.")
    parser.add_argument("--moe-gate-tau", type=float, default=0.3, help="Temperature for MoE eta-sigmoid gating.")
    parser.add_argument(
        "--moe-gate-use-feat",
        action="store_true",
        help="For learned MoE router, use detached pooled features in addition to detached eta input.",
    )
    parser.add_argument(
        "--moe-balance-lambda",
        type=float,
        default=0.01,
        help="Weight for MoE load-balancing regularizer (variance of per-expert load).",
    )
    parser.add_argument(
        "--moe-specialize-lambda",
        type=float,
        default=0.0,
        help="Weight for MoE specialization regularizer (entropy minimization of gate distribution).",
    )
    parser.add_argument(
        "--moe-specialize-start-epoch",
        type=int,
        default=0,
        help="Epoch to start applying --moe-specialize-lambda (use >0 to guard against early collapse).",
    )
    parser.add_argument(
        "--moe-entropy-warmup-lambda",
        type=float,
        default=0.0,
        help="Early anti-collapse entropy-maximization weight for gate distributions (decays over --moe-entropy-warmup-epochs).",
    )
    parser.add_argument(
        "--moe-entropy-warmup-epochs",
        type=int,
        default=0,
        help="Epochs for entropy warmup decay; 0 keeps --moe-entropy-warmup-lambda constant when enabled.",
    )
    parser.add_argument(
        "--moe-gate-tau-start",
        type=float,
        default=-1.0,
        help="If >0 with --moe-gate-tau-anneal-epochs>0, linearly anneal gate tau from this value to --moe-gate-tau.",
    )
    parser.add_argument(
        "--moe-gate-tau-anneal-epochs",
        type=int,
        default=0,
        help="Epochs to anneal gate tau from --moe-gate-tau-start to --moe-gate-tau (0 disables anneal).",
    )
    parser.add_argument(
        "--moe-oracle-gate-train",
        action="store_true",
        help="Train-time diagnostic: route MoE gate using true SNR (oracle) while keeping the student path unchanged.",
    )
    parser.add_argument(
        "--moe-oracle-gate-eval",
        action="store_true",
        help="Eval-time diagnostic: route MoE gate using true SNR (oracle).",
    )
    parser.add_argument("--moe-low-head-idx", type=int, default=0, help="Expert index used as low-SNR head for head-specific CE.")
    parser.add_argument("--moe-high-head-idx", type=int, default=1, help="Expert index used as high-SNR head for head-specific CE.")
    parser.add_argument("--moe-head-low-lambda", type=float, default=0.0, help="Auxiliary CE weight for low-SNR head specialization.")
    parser.add_argument("--moe-head-high-lambda", type=float, default=0.0, help="Auxiliary CE weight for high-SNR head specialization.")
    parser.add_argument("--moe-head-low-snr-lo", type=float, default=-14.0, help="Low-head CE mask lower SNR bound (dB).")
    parser.add_argument("--moe-head-low-snr-hi", type=float, default=2.0, help="Low-head CE mask upper SNR bound (dB).")
    parser.add_argument("--moe-head-high-snr-lo", type=float, default=-6.0, help="High-head CE mask lower SNR bound (dB).")
    parser.add_argument("--moe-head-high-snr-hi", type=float, default=18.0, help="High-head CE mask upper SNR bound (dB).")
    parser.add_argument(
        "--moe-head-ce-warmup",
        type=int,
        default=0,
        help="Epochs to keep head-specific CE disabled before ramping it in.",
    )
    parser.add_argument(
        "--moe-head-ce-ramp",
        type=int,
        default=0,
        help="Linear ramp epochs for head-specific CE after warmup.",
    )
    parser.add_argument(
        "--moe-head-ce-detach-trunk",
        action="store_true",
        help="Compute MoE head-specific CE on detached classifier features (head-only gradients).",
    )
    parser.add_argument(
        "--moe-head-ce-source",
        type=str,
        default="clean",
        choices=["clean", "cls"],
        help="Source forward path for head-specific CE logits: clean aux forward or cls forward.",
    )
    parser.add_argument(
        "--moe-transition-snr-lo",
        type=float,
        default=-8.0,
        help="Lower SNR bound (dB) for transition-band gate diagnostics.",
    )
    parser.add_argument(
        "--moe-transition-snr-hi",
        type=float,
        default=-2.0,
        help="Upper SNR bound (dB) for transition-band gate diagnostics.",
    )
    parser.add_argument(
        "--moe-diversity-lambda",
        type=float,
        default=0.0,
        help="Penalty weight on cosine similarity between expert logits (small values like 1e-3..1e-2).",
    )

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
    parser.add_argument("--moco-pretrain-epochs", type=int, default=0, help="Number of epochs for MoCo-v2 pre-training (0 = disabled).")
    parser.add_argument("--moco-temp", type=float, default=0.20, help="Temperature for MoCo InfoNCE logits.")
    parser.add_argument("--moco-momentum", type=float, default=0.999, help="Momentum coefficient for MoCo key encoder EMA update.")
    parser.add_argument("--moco-queue-size", type=int, default=16384, help="Number of negative keys in MoCo queue.")
    parser.add_argument("--moco-proj-dim", type=int, default=128, help="Output embedding dimension for MoCo projection heads.")
    parser.add_argument("--moco-hidden-dim", type=int, default=512, help="Hidden dimension for MoCo projection MLP.")
    parser.add_argument("--moco-lr", type=float, default=None, help="Learning rate for MoCo pre-training (defaults to --lr).")
    parser.add_argument("--moco-weight-decay", type=float, default=None, help="Weight decay for MoCo pre-training (defaults to --weight-decay).")

    # Extra SSL augmentations for MoCo pair views.
    parser.add_argument("--ssl-aug-awgn-prob", type=float, default=0.0, help="Probability of AWGN augmentation per SSL view.")
    parser.add_argument("--ssl-aug-awgn-snr-min-db", type=float, default=6.0, help="Minimum target SNR (dB) for SSL AWGN augmentation.")
    parser.add_argument("--ssl-aug-awgn-snr-max-db", type=float, default=20.0, help="Maximum target SNR (dB) for SSL AWGN augmentation.")
    parser.add_argument("--ssl-aug-time-mask-prob", type=float, default=0.0, help="Probability of temporal masking augmentation per SSL view.")
    parser.add_argument("--ssl-aug-time-mask-max-frac", type=float, default=0.12, help="Maximum masked time fraction for SSL temporal masking.")
    parser.add_argument("--ssl-aug-iq-drop-prob", type=float, default=0.0, help="Probability of dropping one IQ channel per SSL view.")

    # Supervised Contrastive Learning (SupCon)  –  Khosla et al., NeurIPS 2020
    parser.add_argument("--supcon", action="store_true", help="Enable supervised contrastive loss (SupCon) alongside CE.")
    parser.add_argument("--supcon-lambda", type=float, default=0.1, help="Weight for SupCon loss (relative to CE). Start with 0.1.")
    parser.add_argument("--supcon-temp", type=float, default=0.07, help="Temperature τ for cosine similarities (0.07 is standard for L2-normed embeddings).")
    parser.add_argument("--supcon-proj-dim", type=int, default=128, help="Output dimension of the MLP projection head for SupCon.")
    parser.add_argument("--supcon-warmup", type=int, default=0, help="Number of epochs before SupCon loss kicks in (0 = from the start).")
    parser.add_argument(
        "--supcon-clean-branch-on-mixup-cls-only",
        action="store_true",
        help="When mixup_cls_only is active, compute SupCon from clean-branch features/labels instead of skipping SupCon.",
    )

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
        "--early-stop-start-epoch",
        type=int,
        default=0,
        help="Apply early stopping only at/after this 1-based epoch index (0 disables the start gate).",
    )
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
    parser.add_argument(
        "--snr-weight-ce",
        action="store_true",
        help="Enable SNR-weighted CE (upweight low-SNR samples with capped, normalized per-batch weights).",
    )
    parser.add_argument(
        "--snr-weight-ce-scale",
        type=float,
        default=2.0,
        help="Scale factor for SNR-weighted CE: w=1+scale*normalized_low_snr.",
    )
    parser.add_argument(
        "--snr-weight-ce-max",
        type=float,
        default=3.0,
        help="Maximum per-sample CE weight for SNR-weighted CE.",
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
    parser.add_argument("--dn-diff-enable", action="store_true", help="Enable waveform diffusion denoiser front-end for CLDNN.")
    parser.add_argument("--dn-diff-target", type=str, choices=["v", "eps"], default="v", help="Prediction target for diffusion denoiser (v or eps).")
    parser.add_argument("--dn-diff-train-timesteps", type=int, default=100, help="Diffusion timestep count for waveform denoiser schedule.")
    parser.add_argument("--dn-diff-beta-start", type=float, default=1e-4, help="Beta schedule start value for diffusion denoiser.")
    parser.add_argument("--dn-diff-beta-end", type=float, default=2e-2, help="Beta schedule end value for diffusion denoiser. Increase to cover lower SNR (e.g. 0.10 for -22 dB).")
    parser.add_argument(
        "--dn-diff-train-t-start-source",
        type=str,
        choices=["snr_pred", "fixed"],
        default="snr_pred",
        help="Training t-start source for diffusion objective.",
    )
    parser.add_argument(
        "--dn-diff-train-forward-mode",
        type=str,
        choices=["raw", "onestep"],
        default="onestep",
        help="Classifier-path denoiser behavior during training: raw (legacy bypass) or onestep reconstruction.",
    )
    parser.add_argument(
        "--dn-diff-eval-t-start-source",
        type=str,
        choices=["snr_pred", "snr_true", "fixed"],
        default="snr_pred",
        help="Eval t-start source for diffusion denoising.",
    )
    parser.add_argument("--dn-diff-fixed-t-start", type=int, default=30, help="Fixed t-start when train/eval t-source is fixed.")
    parser.add_argument("--dn-diff-snr2t-scale", type=float, default=1.0, help="Scale factor applied to snr_to_t mapping.")
    parser.add_argument("--dn-diff-snr2t-bias", type=float, default=0.0, help="Bias applied after snr_to_t mapping.")
    parser.add_argument(
        "--dn-diff-detach-eta-cond",
        dest="dn_diff_detach_eta_cond",
        action="store_true",
        help="Detach predicted eta before conditioning the diffusion denoiser (default).",
    )
    parser.add_argument(
        "--no-dn-diff-detach-eta-cond",
        dest="dn_diff_detach_eta_cond",
        action="store_false",
        help="Allow diffusion losses to backprop through eta conditioning path.",
    )
    parser.set_defaults(dn_diff_detach_eta_cond=True)
    parser.add_argument("--dn-diff-eval-mode", type=str, choices=["ddim", "onestep"], default="onestep", help="Diffusion eval reconstruction mode.")
    parser.add_argument("--dn-diff-eval-steps", type=int, default=8, help="DDIM steps for eval-time denoising.")
    parser.add_argument("--dn-diff-ddim-eta", type=float, default=0.0, help="DDIM eta (0 = deterministic).")
    parser.add_argument("--dn-diff-multisample", type=int, default=1, help="Number of denoised samples to average at eval.")
    parser.add_argument(
        "--dn-diff-allow-eval-ddim-mismatch",
        action="store_true",
        help=(
            "Allow objective mismatch where training classifier path uses one-step denoising "
            "but eval/deploy uses DDIM."
        ),
    )
    parser.add_argument("--dn-diff-low-snr-thresh", type=float, default=-6.0, help="Low-SNR denoising threshold (dB).")
    parser.add_argument("--dn-diff-high-snr-margin", type=float, default=2.0, help="High-SNR bypass margin above low-SNR threshold.")
    parser.add_argument(
        "--dn-diff-hard-bypass-high-snr",
        dest="dn_diff_hard_bypass_high_snr",
        action="store_true",
        help="Hard-bypass diffusion denoiser at high SNR.",
    )
    parser.add_argument(
        "--no-dn-diff-hard-bypass-high-snr",
        dest="dn_diff_hard_bypass_high_snr",
        action="store_false",
        help="Disable hard high-SNR diffusion bypass.",
    )
    parser.set_defaults(dn_diff_hard_bypass_high_snr=True)
    parser.add_argument(
        "--dn-diff-apply-lowband-only-train",
        dest="dn_diff_apply_lowband_only_train",
        action="store_true",
        help="Apply diffusion losses only inside configured low-SNR training band.",
    )
    parser.add_argument(
        "--no-dn-diff-apply-lowband-only-train",
        dest="dn_diff_apply_lowband_only_train",
        action="store_false",
        help="Disable low-band-only masking for diffusion losses.",
    )
    parser.set_defaults(dn_diff_apply_lowband_only_train=True)
    parser.add_argument("--dn-diff-loss-snr-lo", type=float, default=-14.0, help="Lower SNR bound for diffusion training loss mask.")
    parser.add_argument("--dn-diff-loss-snr-hi", type=float, default=-6.0, help="Upper SNR bound for diffusion training loss mask.")
    parser.add_argument(
        "--dn-diff-loss-cond-source",
        type=str,
        choices=["raw", "degraded"],
        default="raw",
        help=(
            "Condition source for dn-diff training loss. "
            "raw matches eval-time conditioning; degraded keeps legacy extra-degraded conditioning."
        ),
    )
    parser.add_argument(
        "--dn-diff-freeze-classifier",
        dest="dn_diff_freeze_classifier",
        action="store_true",
        help="Freeze classifier stack; default resolves to true when dn_diff is enabled, else false.",
    )
    parser.add_argument(
        "--no-dn-diff-freeze-classifier",
        dest="dn_diff_freeze_classifier",
        action="store_false",
        help="Disable classifier freezing during diffusion training.",
    )
    parser.set_defaults(dn_diff_freeze_classifier=None)
    parser.add_argument(
        "--allow-random-frozen-classifier",
        action="store_true",
        help="Allow freezing classifier without loading pretrained weights (debug only).",
    )
    parser.add_argument(
        "--init-ckpt",
        type=str,
        default=None,
        help="Initialize model weights from checkpoint without resuming optimizer/scheduler state.",
    )
    parser.add_argument(
        "--init-ckpt-source",
        type=str,
        default="auto",
        choices=["auto", "model", "ema"],
        help=(
            "Warm-start source for --init-ckpt. "
            "auto uses EMA for frozen-classifier runs when available; otherwise model weights."
        ),
    )
    parser.add_argument("--lambda-dn-diff", type=float, default=1.0, help="Weight for diffusion denoiser target loss.")
    parser.add_argument("--lambda-dn-recon", type=float, default=0.0, help="Weight for one-step reconstruction loss from diffusion x0 prediction.")
    parser.add_argument(
        "--lambda-dn-cls",
        type=float,
        default=0.0,
        help="Task-aware CE on denoised outputs (updates denoiser path; classifier can remain frozen).",
    )
    parser.add_argument(
        "--dn-diff-cls-warmup",
        type=int,
        default=0,
        help="Epochs to keep dn-diff classification loss disabled before ramping.",
    )
    parser.add_argument(
        "--dn-diff-cls-ramp",
        type=int,
        default=0,
        help="Linear ramp epochs for dn-diff classification loss after warmup.",
    )
    parser.add_argument(
        "--dn-diff-diff-warmup",
        type=int,
        default=0,
        help="Epochs to keep full dn-diff reconstruction weight before optional decay.",
    )
    parser.add_argument(
        "--dn-diff-diff-ramp",
        type=int,
        default=0,
        help="Linear ramp epochs for dn-diff reconstruction weight toward --dn-diff-diff-final-scale.",
    )
    parser.add_argument(
        "--dn-diff-diff-final-scale",
        type=float,
        default=1.0,
        help="Final multiplicative scale applied to --lambda-dn-diff after warmup/ramp.",
    )
    parser.add_argument("--lambda-dn-feat-align", type=float, default=0.0, help="Weight for diffusion feature-alignment loss.")
    parser.add_argument("--lambda-dn-logit-align", type=float, default=0.0, help="Weight for diffusion logit-alignment loss.")
    parser.add_argument(
        "--dn-diff-require-noise-supervision",
        dest="dn_diff_require_noise_supervision",
        action="store_true",
        help=(
            "Require lambda_noise>0 when dn-diff control path uses predicted SNR "
            "(t-start and/or high-SNR bypass in snr_mode=predict)."
        ),
    )
    parser.add_argument(
        "--no-dn-diff-require-noise-supervision",
        dest="dn_diff_require_noise_supervision",
        action="store_false",
        help="Allow predicted-SNR dn-diff controls with lambda_noise<=0 (debug only).",
    )
    parser.set_defaults(dn_diff_require_noise_supervision=True)
    parser.add_argument(
        "--dn-diff-align-teacher",
        type=str,
        choices=["none", "frozen", "ema"],
        default="frozen",
        help="Teacher mode for diffusion alignment losses.",
    )
    parser.add_argument("--dn-diff-feat-align-start-epoch", type=int, default=20, help="Epoch to start diffusion feature alignment.")
    parser.add_argument("--dn-diff-logit-align-start-epoch", type=int, default=30, help="Epoch to start diffusion logit alignment.")
    parser.add_argument(
        "--dn-diff-cond-diagnostic",
        type=str,
        choices=["none", "zero", "shuffle"],
        default="none",
        help="Conditioning diagnostic mode for diffusion eval.",
    )
    parser.add_argument(
        "--dn-diff-force-deterministic-multisample",
        action="store_true",
        help="Keep DDIM deterministic for multisample eval (debug only).",
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
            cldnn_backbone=str(_cfg_get(cfg, fallback, "cldnn_backbone", "lstm")),
            cldnn_tcn_levels=int(_cfg_get(cfg, fallback, "cldnn_tcn_levels", 6)),
            cldnn_tcn_channels=int(_cfg_get(cfg, fallback, "cldnn_tcn_channels", 128)),
            cldnn_tcn_kernel=int(_cfg_get(cfg, fallback, "cldnn_tcn_kernel", 3)),
            cldnn_tcn_dilation_base=int(_cfg_get(cfg, fallback, "cldnn_tcn_dilation_base", 2)),
            cldnn_tcn_dropout=float(_cfg_get(cfg, fallback, "cldnn_tcn_dropout", 0.15)),
            cldnn_resnet_blocks=int(_cfg_get(cfg, fallback, "cldnn_resnet_blocks", 8)),
            cldnn_resnet_channels=int(_cfg_get(cfg, fallback, "cldnn_resnet_channels", 128)),
            cldnn_resnet_kernel=int(_cfg_get(cfg, fallback, "cldnn_resnet_kernel", 5)),
            cldnn_resnet_dilation_cycle=int(_cfg_get(cfg, fallback, "cldnn_resnet_dilation_cycle", 4)),
            cldnn_resnet_dropout=float(_cfg_get(cfg, fallback, "cldnn_resnet_dropout", 0.15)),
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
            moe_n_experts=int(cfg.get("moe_n_experts", 1) if isinstance(cfg, dict) else getattr(fallback, "moe_n_experts", 1)),
            moe_gate_type=str(cfg.get("moe_gate_type", "eta-sigmoid") if isinstance(cfg, dict) else getattr(fallback, "moe_gate_type", "eta-sigmoid")),
            moe_gate_center=float(cfg.get("moe_gate_center", 0.5) if isinstance(cfg, dict) else getattr(fallback, "moe_gate_center", 0.5)),
            moe_gate_tau=float(cfg.get("moe_gate_tau", 0.3) if isinstance(cfg, dict) else getattr(fallback, "moe_gate_tau", 0.3)),
            moe_gate_use_feat=bool(cfg.get("moe_gate_use_feat", False) if isinstance(cfg, dict) else getattr(fallback, "moe_gate_use_feat", False)),
            supcon_proj_dim=int(_cfg_get(cfg, fallback, "supcon_proj_dim", 0)) if bool(_cfg_get(cfg, fallback, "supcon", False)) else 0,
            dn_diff_enable=bool(_cfg_get(cfg, fallback, "dn_diff_enable", False)),
            dn_diff_target=str(_cfg_get(cfg, fallback, "dn_diff_target", "v")),
            dn_diff_train_timesteps=int(_cfg_get(cfg, fallback, "dn_diff_train_timesteps", 100)),
            dn_diff_beta_start=float(_cfg_get(cfg, fallback, "dn_diff_beta_start", 1e-4)),
            dn_diff_beta_end=float(_cfg_get(cfg, fallback, "dn_diff_beta_end", 2e-2)),
            dn_diff_train_t_start_source=str(_cfg_get(cfg, fallback, "dn_diff_train_t_start_source", "snr_pred")),
            dn_diff_train_forward_mode=str(_cfg_get(cfg, fallback, "dn_diff_train_forward_mode", "onestep")),
            dn_diff_eval_mode=str(_cfg_get(cfg, fallback, "dn_diff_eval_mode", "onestep")),
            dn_diff_eval_steps=int(_cfg_get(cfg, fallback, "dn_diff_eval_steps", 8)),
            dn_diff_ddim_eta=float(_cfg_get(cfg, fallback, "dn_diff_ddim_eta", 0.0)),
            dn_diff_multisample=int(_cfg_get(cfg, fallback, "dn_diff_multisample", 1)),
            dn_diff_eval_t_start_source=str(_cfg_get(cfg, fallback, "dn_diff_eval_t_start_source", "snr_pred")),
            dn_diff_fixed_t_start=int(_cfg_get(cfg, fallback, "dn_diff_fixed_t_start", 30)),
            dn_diff_snr2t_scale=float(_cfg_get(cfg, fallback, "dn_diff_snr2t_scale", 1.0)),
            dn_diff_snr2t_bias=float(_cfg_get(cfg, fallback, "dn_diff_snr2t_bias", 0.0)),
            dn_diff_detach_eta_cond=bool(_cfg_get(cfg, fallback, "dn_diff_detach_eta_cond", True)),
            dn_diff_low_snr_thresh=float(_cfg_get(cfg, fallback, "dn_diff_low_snr_thresh", -6.0)),
            dn_diff_high_snr_margin=float(_cfg_get(cfg, fallback, "dn_diff_high_snr_margin", 2.0)),
            dn_diff_hard_bypass_high_snr=bool(_cfg_get(cfg, fallback, "dn_diff_hard_bypass_high_snr", True)),
            dn_diff_cond_diagnostic=str(_cfg_get(cfg, fallback, "dn_diff_cond_diagnostic", "none")),
            dn_diff_force_deterministic_multisample=bool(
                _cfg_get(cfg, fallback, "dn_diff_force_deterministic_multisample", False)
            ),
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


def _resolve_eval_snr_input(model: torch.nn.Module, snr: torch.Tensor, snr_mode: str) -> Optional[torch.Tensor]:
    """
    Decide whether to pass true SNR into model forward during evaluation helpers.
    Normal behavior is to pass SNR only when snr_mode == 'known'.
    For dn-diff oracle diagnostics, if eval t-start source is 'snr_true', pass SNR
    even when snr_mode == 'predict' so the oracle path is actually exercised.
    """
    if snr_mode == "known":
        return snr
    if bool(getattr(model, "dn_diff_enabled", False)):
        src = str(getattr(model, "dn_diff_eval_t_start_source", "snr_pred")).strip().lower()
        if src == "snr_true":
            return snr
    return None


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    t_eval: int,
    snr_mode: str,
    amp: bool = False,
    low_snr_lo: float = -14.0,
    low_snr_hi: float = -6.0,
    moe_oracle_gate_eval: bool = False,
    moe_transition_snr_lo: float = -8.0,
    moe_transition_snr_hi: float = -2.0,
) -> Tuple[float, float, Dict[int, float], Dict[str, float]]:
    model.eval()
    total_correct = 0
    total = 0
    snr_correct: Dict[int, int] = {}
    snr_total: Dict[int, int] = {}
    conf_all: Optional[np.ndarray] = None
    conf_low: Optional[np.ndarray] = None
    moe_expert_correct: Optional[np.ndarray] = None
    moe_expert_low_correct: Optional[np.ndarray] = None
    moe_expert_total = 0
    moe_expert_low_total = 0
    moe_gate_load_sum: Optional[np.ndarray] = None
    moe_gate_transition_load_sum: Optional[np.ndarray] = None
    moe_gate_count = 0
    moe_gate_transition_count = 0
    moe_gate_entropy_sum = 0.0
    dn_diff_t_sum = 0.0
    dn_diff_t_sq_sum = 0.0
    dn_diff_t_count = 0
    dn_diff_active_low_sum = 0.0
    dn_diff_active_mid_sum = 0.0
    dn_diff_active_high_sum = 0.0
    dn_diff_active_low_count = 0
    dn_diff_active_mid_count = 0
    dn_diff_active_high_count = 0

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
            snr_in = _resolve_eval_snr_input(model, snr, snr_mode)
            moe_kwargs_eval: Dict[str, object] = {}
            if bool(moe_oracle_gate_eval) and getattr(model, "moe_head", None) is not None:
                moe_kwargs_eval["moe_use_oracle_gate"] = True
                moe_kwargs_eval["moe_oracle_snr"] = snr
            if amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=_cuda_amp_dtype(), enabled=True):
                    logits, _, _ = model(
                        x,
                        t,
                        snr=snr_in,
                        snr_mode=snr_mode,
                        group_mask=mask,
                        **moe_kwargs_eval,
                    )
            else:
                logits, _, _ = model(
                    x,
                    t,
                    snr=snr_in,
                    snr_mode=snr_mode,
                    group_mask=mask,
                    **moe_kwargs_eval,
                )

            preds = logits.argmax(dim=1)
            correct = (preds == y).sum().item()
            total_correct += correct
            total += y.shape[0]
            low_mask_dev = (snr.float() >= float(low_snr_lo)) & (snr.float() <= float(low_snr_hi))
            trans_mask_dev = (snr.float() >= float(moe_transition_snr_lo)) & (snr.float() <= float(moe_transition_snr_hi))

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
            low_mask_t = low_mask_dev.detach().cpu()
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

            logits_experts = getattr(model, "_moe_logits_experts", None)
            if logits_experts is not None and isinstance(logits_experts, torch.Tensor) and logits_experts.ndim == 3:
                n_exp = int(logits_experts.shape[1])
                if moe_expert_correct is None or moe_expert_correct.shape[0] != n_exp:
                    moe_expert_correct = np.zeros((n_exp,), dtype=np.float64)
                    moe_expert_low_correct = np.zeros((n_exp,), dtype=np.float64)
                preds_experts = logits_experts.argmax(dim=2)  # (B,E)
                corr_experts = (preds_experts == y.unsqueeze(1))
                moe_expert_correct += corr_experts.sum(dim=0).detach().cpu().numpy()
                moe_expert_total += int(y.shape[0])
                if bool(torch.any(low_mask_dev)):
                    corr_low = corr_experts[low_mask_dev]
                    moe_expert_low_correct += corr_low.sum(dim=0).detach().cpu().numpy()
                    moe_expert_low_total += int(corr_low.shape[0])

            gate = getattr(model, "_moe_gate", None)
            if gate is not None and isinstance(gate, torch.Tensor) and gate.ndim == 2:
                g = gate.float()
                n_exp_gate = int(g.shape[1])
                if moe_gate_load_sum is None or moe_gate_load_sum.shape[0] != n_exp_gate:
                    moe_gate_load_sum = np.zeros((n_exp_gate,), dtype=np.float64)
                    moe_gate_transition_load_sum = np.zeros((n_exp_gate,), dtype=np.float64)
                moe_gate_load_sum += g.sum(dim=0).detach().cpu().numpy()
                moe_gate_count += int(g.shape[0])
                ent = -(torch.clamp(g, min=1e-8) * torch.log(torch.clamp(g, min=1e-8))).sum(dim=1)
                moe_gate_entropy_sum += float(ent.sum().item())
                if bool(torch.any(trans_mask_dev)):
                    g_trans = g[trans_mask_dev]
                    moe_gate_transition_load_sum += g_trans.sum(dim=0).detach().cpu().numpy()
                    moe_gate_transition_count += int(g_trans.shape[0])

            if bool(getattr(model, "dn_diff_enabled", False)):
                t_start_eval = getattr(model, "_dn_diff_t_start", None)
                if isinstance(t_start_eval, torch.Tensor):
                    t_eval_f = t_start_eval.detach().float().view(-1)
                    dn_diff_t_sum += float(t_eval_f.sum().item())
                    dn_diff_t_sq_sum += float((t_eval_f * t_eval_f).sum().item())
                    dn_diff_t_count += int(t_eval_f.numel())

                active_eval = getattr(model, "_dn_diff_active_mask", None)
                if isinstance(active_eval, torch.Tensor):
                    a = active_eval.detach().float().view(-1)
                    if x.ndim == 4 and int(a.numel()) == int(x.shape[0] * x.shape[1]):
                        a = a.view(x.shape[0], x.shape[1]).mean(dim=1)
                    elif int(a.numel()) != int(y.shape[0]):
                        a = None
                    if a is not None:
                        mid_mask_dev = (~low_mask_dev) & (snr.float() < 6.0)
                        high_mask_dev = snr.float() >= 6.0
                        if bool(torch.any(low_mask_dev)):
                            dn_diff_active_low_sum += float(a[low_mask_dev].sum().item())
                            dn_diff_active_low_count += int(low_mask_dev.sum().item())
                        if bool(torch.any(mid_mask_dev)):
                            dn_diff_active_mid_sum += float(a[mid_mask_dev].sum().item())
                            dn_diff_active_mid_count += int(mid_mask_dev.sum().item())
                        if bool(torch.any(high_mask_dev)):
                            dn_diff_active_high_sum += float(a[high_mask_dev].sum().item())
                            dn_diff_active_high_count += int(high_mask_dev.sum().item())

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
        "overall_acc": float(acc),
        "macro_acc": float(macro_all.get("macro_acc", 0.0)),
        "macro_f1": float(macro_all.get("macro_f1", 0.0)),
        "low_macro_acc": float(macro_low.get("macro_acc", 0.0)),
        "low_macro_f1": float(macro_low.get("macro_f1", 0.0)),
        "low_snr_lo": float(low_snr_lo),
        "low_snr_hi": float(low_snr_hi),
    }
    if moe_expert_correct is not None and moe_expert_total > 0:
        for i in range(int(moe_expert_correct.shape[0])):
            summary[f"moe_expert_{i}_acc"] = float(moe_expert_correct[i] / float(moe_expert_total))
            if moe_expert_low_correct is not None and moe_expert_low_total > 0:
                summary[f"moe_expert_{i}_low_acc"] = float(moe_expert_low_correct[i] / float(moe_expert_low_total))
            else:
                summary[f"moe_expert_{i}_low_acc"] = 0.0
    if moe_gate_load_sum is not None and moe_gate_count > 0:
        summary["moe_gate_entropy"] = float(moe_gate_entropy_sum / float(moe_gate_count))
        summary["moe_transition_snr_lo"] = float(moe_transition_snr_lo)
        summary["moe_transition_snr_hi"] = float(moe_transition_snr_hi)
        for i in range(int(moe_gate_load_sum.shape[0])):
            summary[f"moe_gate_expert_{i}_load"] = float(moe_gate_load_sum[i] / float(moe_gate_count))
            if moe_gate_transition_load_sum is not None and moe_gate_transition_count > 0:
                summary[f"moe_gate_expert_{i}_load_transition"] = float(
                    moe_gate_transition_load_sum[i] / float(moe_gate_transition_count)
                )
            else:
                summary[f"moe_gate_expert_{i}_load_transition"] = 0.0
    if dn_diff_t_count > 0:
        t_mean = dn_diff_t_sum / float(dn_diff_t_count)
        t_var = max(0.0, dn_diff_t_sq_sum / float(dn_diff_t_count) - t_mean * t_mean)
        summary["dn_diff_t_start_mean"] = float(t_mean)
        summary["dn_diff_t_start_std"] = float(math.sqrt(t_var))
    else:
        summary["dn_diff_t_start_mean"] = 0.0
        summary["dn_diff_t_start_std"] = 0.0
    summary["dn_diff_active_frac_low"] = (
        float(dn_diff_active_low_sum / float(dn_diff_active_low_count))
        if dn_diff_active_low_count > 0
        else 0.0
    )
    summary["dn_diff_active_frac_mid"] = (
        float(dn_diff_active_mid_sum / float(dn_diff_active_mid_count))
        if dn_diff_active_mid_count > 0
        else 0.0
    )
    summary["dn_diff_active_frac_high"] = (
        float(dn_diff_active_high_sum / float(dn_diff_active_high_count))
        if dn_diff_active_high_count > 0
        else 0.0
    )
    summary["dn_diff_eval_steps"] = float(getattr(model, "dn_diff_eval_steps", 0))
    summary["dn_diff_ddim_eta"] = float(getattr(model, "dn_diff_ddim_eta_used", getattr(model, "dn_diff_ddim_eta", 0.0)))
    summary["dn_diff_multisample"] = float(getattr(model, "dn_diff_multisample", 1))
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
            snr_in = _resolve_eval_snr_input(model, snr, snr_mode)

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
            snr_in = _resolve_eval_snr_input(model, snr, snr_mode)
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
            snr_in = _resolve_eval_snr_input(model, snr, snr_mode)
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
    if bool(getattr(args, "dn_diff_enable", False)) and args.arch != "cldnn":
        raise ValueError("dn_diff_enable is currently supported only for --arch cldnn.")
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
    if args.arch == "cldnn":
        cldnn_backbone = str(getattr(args, "cldnn_backbone", "lstm")).strip().lower()
        if cldnn_backbone not in {"lstm", "tcn", "resnet1d"}:
            raise ValueError("cldnn_backbone must be one of: lstm|tcn|resnet1d.")
        if int(getattr(args, "cldnn_tcn_levels", 6)) <= 0:
            raise ValueError("cldnn_tcn_levels must be > 0.")
        if int(getattr(args, "cldnn_tcn_channels", 128)) <= 0:
            raise ValueError("cldnn_tcn_channels must be > 0.")
        tcn_k = int(getattr(args, "cldnn_tcn_kernel", 3))
        if tcn_k <= 0 or (tcn_k % 2) == 0:
            raise ValueError("cldnn_tcn_kernel must be odd and > 0.")
        if int(getattr(args, "cldnn_tcn_dilation_base", 2)) <= 0:
            raise ValueError("cldnn_tcn_dilation_base must be > 0.")
        if float(getattr(args, "cldnn_tcn_dropout", 0.15)) < 0.0:
            raise ValueError("cldnn_tcn_dropout must be >= 0.")
        if int(getattr(args, "cldnn_resnet_blocks", 8)) <= 0:
            raise ValueError("cldnn_resnet_blocks must be > 0.")
        if int(getattr(args, "cldnn_resnet_channels", 128)) <= 0:
            raise ValueError("cldnn_resnet_channels must be > 0.")
        res_k = int(getattr(args, "cldnn_resnet_kernel", 5))
        if res_k <= 0 or (res_k % 2) == 0:
            raise ValueError("cldnn_resnet_kernel must be odd and > 0.")
        if int(getattr(args, "cldnn_resnet_dilation_cycle", 4)) <= 0:
            raise ValueError("cldnn_resnet_dilation_cycle must be > 0.")
        if float(getattr(args, "cldnn_resnet_dropout", 0.15)) < 0.0:
            raise ValueError("cldnn_resnet_dropout must be >= 0.")
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
    init_ckpt_path = str(getattr(args, "init_ckpt", "") or "").strip()
    if init_ckpt_path and (args.ckpt is not None and args.resume):
        raise ValueError("Use either --init-ckpt or --ckpt with --resume, not both.")
    if init_ckpt_path and not os.path.exists(init_ckpt_path):
        raise FileNotFoundError(f"init_ckpt not found: {init_ckpt_path}")
    if int(getattr(args, "early_stop_patience", 0)) < 0:
        raise ValueError("early_stop_patience must be >= 0.")
    if int(getattr(args, "early_stop_start_epoch", 0)) < 0:
        raise ValueError("early_stop_start_epoch must be >= 0.")
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
    if bool(getattr(args, "dn_diff_enable", False)):
        if args.arch != "cldnn":
            raise ValueError("dn_diff_enable requires --arch cldnn.")
        if int(getattr(args, "dn_diff_train_timesteps", 100)) <= 1:
            raise ValueError("dn_diff_train_timesteps must be > 1.")
        beta_start = float(getattr(args, "dn_diff_beta_start", 1e-4))
        beta_end = float(getattr(args, "dn_diff_beta_end", 2e-2))
        if beta_start <= 0.0 or beta_start >= 1.0:
            raise ValueError("dn_diff_beta_start must be in (0,1).")
        if beta_end <= 0.0 or beta_end >= 1.0:
            raise ValueError("dn_diff_beta_end must be in (0,1).")
        if beta_end <= beta_start:
            raise ValueError("dn_diff_beta_end must be > dn_diff_beta_start.")
        if int(getattr(args, "dn_diff_eval_steps", 8)) <= 0:
            raise ValueError("dn_diff_eval_steps must be > 0.")
        if int(getattr(args, "dn_diff_multisample", 1)) <= 0:
            raise ValueError("dn_diff_multisample must be > 0.")
        if float(getattr(args, "dn_diff_loss_snr_lo", -14.0)) > float(getattr(args, "dn_diff_loss_snr_hi", -6.0)):
            raise ValueError("dn_diff_loss_snr_lo must be <= dn_diff_loss_snr_hi.")
        if int(getattr(args, "dn_diff_fixed_t_start", 30)) < 0:
            raise ValueError("dn_diff_fixed_t_start must be >= 0.")
        if float(getattr(args, "lambda_dn_diff", 1.0)) < 0.0:
            raise ValueError("lambda_dn_diff must be >= 0.")
        if float(getattr(args, "lambda_dn_recon", 0.0)) < 0.0:
            raise ValueError("lambda_dn_recon must be >= 0.")
        if float(getattr(args, "lambda_dn_cls", 0.0)) < 0.0:
            raise ValueError("lambda_dn_cls must be >= 0.")
        if int(getattr(args, "dn_diff_cls_warmup", 0)) < 0:
            raise ValueError("dn_diff_cls_warmup must be >= 0.")
        if int(getattr(args, "dn_diff_cls_ramp", 0)) < 0:
            raise ValueError("dn_diff_cls_ramp must be >= 0.")
        if int(getattr(args, "dn_diff_diff_warmup", 0)) < 0:
            raise ValueError("dn_diff_diff_warmup must be >= 0.")
        if int(getattr(args, "dn_diff_diff_ramp", 0)) < 0:
            raise ValueError("dn_diff_diff_ramp must be >= 0.")
        if float(getattr(args, "dn_diff_diff_final_scale", 1.0)) < 0.0:
            raise ValueError("dn_diff_diff_final_scale must be >= 0.")
        if float(getattr(args, "lambda_dn_feat_align", 0.0)) < 0.0:
            raise ValueError("lambda_dn_feat_align must be >= 0.")
        if float(getattr(args, "lambda_dn_logit_align", 0.0)) < 0.0:
            raise ValueError("lambda_dn_logit_align must be >= 0.")
        if int(getattr(args, "dn_diff_feat_align_start_epoch", 20)) < 0:
            raise ValueError("dn_diff_feat_align_start_epoch must be >= 0.")
        if int(getattr(args, "dn_diff_logit_align_start_epoch", 30)) < 0:
            raise ValueError("dn_diff_logit_align_start_epoch must be >= 0.")
        train_t_src = str(getattr(args, "dn_diff_train_t_start_source", "snr_pred")).strip().lower()
        eval_t_src = str(getattr(args, "dn_diff_eval_t_start_source", "snr_pred")).strip().lower()
        if train_t_src == "snr_true":
            raise ValueError("dn_diff_train_t_start_source cannot be snr_true.")
        if (
            str(getattr(args, "dn_diff_train_forward_mode", "onestep")).strip().lower() == "onestep"
            and str(getattr(args, "dn_diff_eval_mode", "onestep")).strip().lower() == "ddim"
            and not bool(getattr(args, "dn_diff_allow_eval_ddim_mismatch", False))
        ):
            raise ValueError(
                "dn_diff eval/training objective mismatch is blocked by default: "
                "train_forward_mode=onestep with eval_mode=ddim. "
                "Use --dn-diff-eval-mode onestep, or explicitly allow via "
                "--dn-diff-allow-eval-ddim-mismatch."
            )
        if (
            bool(getattr(args, "dn_diff_require_noise_supervision", True))
            and str(getattr(args, "snr_mode", "predict")).strip().lower() == "predict"
        ):
            pred_control_active = (
                train_t_src == "snr_pred"
                or eval_t_src == "snr_pred"
                or bool(getattr(args, "dn_diff_hard_bypass_high_snr", True))
            )
            if pred_control_active and float(getattr(args, "lambda_noise", 0.0)) <= 0.0:
                raise ValueError(
                    "dn-diff control uses predicted SNR but lambda_noise<=0; this leaves control "
                    "signals unsupervised. Set --lambda-noise > 0, disable predicted controls, or "
                    "override with --no-dn-diff-require-noise-supervision."
                )
        if (
            (float(getattr(args, "lambda_dn_feat_align", 0.0)) > 0.0
             or float(getattr(args, "lambda_dn_logit_align", 0.0)) > 0.0)
            and str(getattr(args, "dn_diff_align_teacher", "frozen")).strip().lower() == "none"
        ):
            raise ValueError(
                "Diffusion alignment losses require a stable teacher target "
                "(use --dn-diff-align-teacher frozen|ema)."
            )
        freeze_cfg_eval = getattr(args, "dn_diff_freeze_classifier", None)
        freeze_classifier_eval = bool(getattr(args, "dn_diff_enable", False)) if freeze_cfg_eval is None else bool(freeze_cfg_eval)
        if freeze_classifier_eval:
            has_task_signal = (
                float(getattr(args, "lambda_dn_cls", 0.0)) > 0.0
                or float(getattr(args, "lambda_dn_feat_align", 0.0)) > 0.0
                or float(getattr(args, "lambda_dn_logit_align", 0.0)) > 0.0
            )
            if not has_task_signal:
                raise ValueError(
                    "Frozen-classifier dn-diff run requires task-aware denoiser supervision. "
                    "Set --lambda-dn-cls > 0 and/or enable alignment losses."
                )
        if (
            int(getattr(args, "dn_diff_multisample", 1)) > 1
            and abs(float(getattr(args, "dn_diff_ddim_eta", 0.0))) < 1e-12
            and not bool(getattr(args, "dn_diff_force_deterministic_multisample", False))
        ):
            print("[dn_diff] multisample with eta=0 detected; eval eta will be auto-promoted to 0.2.")
    elif float(getattr(args, "lambda_dn_cls", 0.0)) > 0.0:
        raise ValueError("lambda_dn_cls > 0 requires --dn-diff-enable.")
    if float(getattr(args, "snr_consist_low_delta_min", 2.0)) < 0.0 or float(getattr(args, "snr_consist_low_delta_max", 4.0)) < 0.0:
        raise ValueError("snr_consist_low_delta_min/max must be >= 0.")
    if float(getattr(args, "snr_consist_low_delta_max", 4.0)) < float(getattr(args, "snr_consist_low_delta_min", 2.0)):
        raise ValueError("snr_consist_low_delta_max must be >= snr_consist_low_delta_min.")
    if float(getattr(args, "snr_weight_ce_scale", 2.0)) < 0.0:
        raise ValueError("snr_weight_ce_scale must be >= 0.")
    if float(getattr(args, "snr_weight_ce_max", 3.0)) <= 0.0:
        raise ValueError("snr_weight_ce_max must be > 0.")
    if int(getattr(args, "moe_n_experts", 1)) < 1:
        raise ValueError("moe_n_experts must be >= 1.")
    if float(getattr(args, "moe_gate_tau", 0.3)) <= 0.0:
        raise ValueError("moe_gate_tau must be > 0.")
    if float(getattr(args, "moe_balance_lambda", 0.01)) < 0.0:
        raise ValueError("moe_balance_lambda must be >= 0.")
    if float(getattr(args, "moe_specialize_lambda", 0.0)) < 0.0:
        raise ValueError("moe_specialize_lambda must be >= 0.")
    if int(getattr(args, "moe_specialize_start_epoch", 0)) < 0:
        raise ValueError("moe_specialize_start_epoch must be >= 0.")
    if float(getattr(args, "moe_entropy_warmup_lambda", 0.0)) < 0.0:
        raise ValueError("moe_entropy_warmup_lambda must be >= 0.")
    if int(getattr(args, "moe_entropy_warmup_epochs", 0)) < 0:
        raise ValueError("moe_entropy_warmup_epochs must be >= 0.")
    moe_tau_start = float(getattr(args, "moe_gate_tau_start", -1.0))
    if moe_tau_start != -1.0 and moe_tau_start <= 0.0:
        raise ValueError("moe_gate_tau_start must be > 0 (or -1 to disable).")
    if int(getattr(args, "moe_gate_tau_anneal_epochs", 0)) < 0:
        raise ValueError("moe_gate_tau_anneal_epochs must be >= 0.")
    if float(getattr(args, "moe_head_low_lambda", 0.0)) < 0.0:
        raise ValueError("moe_head_low_lambda must be >= 0.")
    if float(getattr(args, "moe_head_high_lambda", 0.0)) < 0.0:
        raise ValueError("moe_head_high_lambda must be >= 0.")
    if int(getattr(args, "moe_head_ce_warmup", 0)) < 0:
        raise ValueError("moe_head_ce_warmup must be >= 0.")
    if int(getattr(args, "moe_head_ce_ramp", 0)) < 0:
        raise ValueError("moe_head_ce_ramp must be >= 0.")
    if float(getattr(args, "moe_diversity_lambda", 0.0)) < 0.0:
        raise ValueError("moe_diversity_lambda must be >= 0.")
    if float(getattr(args, "moe_head_low_snr_lo", -14.0)) > float(getattr(args, "moe_head_low_snr_hi", 2.0)):
        raise ValueError("moe_head_low_snr_lo must be <= moe_head_low_snr_hi.")
    if float(getattr(args, "moe_head_high_snr_lo", -6.0)) > float(getattr(args, "moe_head_high_snr_hi", 18.0)):
        raise ValueError("moe_head_high_snr_lo must be <= moe_head_high_snr_hi.")
    if float(getattr(args, "moe_transition_snr_lo", -8.0)) > float(getattr(args, "moe_transition_snr_hi", -2.0)):
        raise ValueError("moe_transition_snr_lo must be <= moe_transition_snr_hi.")
    if int(getattr(args, "moe_low_head_idx", 0)) == int(getattr(args, "moe_high_head_idx", 1)):
        raise ValueError("moe_low_head_idx and moe_high_head_idx must be different.")
    if int(getattr(args, "moe_n_experts", 1)) > 1 and args.arch != "cldnn":
        raise ValueError("moe_n_experts > 1 is currently supported only for --arch cldnn.")
    if (
        int(getattr(args, "moe_n_experts", 1)) <= 1
        and (
            float(getattr(args, "moe_head_low_lambda", 0.0)) > 0.0
            or float(getattr(args, "moe_head_high_lambda", 0.0)) > 0.0
            or float(getattr(args, "moe_diversity_lambda", 0.0)) > 0.0
            or bool(getattr(args, "moe_oracle_gate_train", False))
            or bool(getattr(args, "moe_oracle_gate_eval", False))
            or bool(getattr(args, "moe_head_ce_detach_trunk", False))
        )
    ):
        raise ValueError("MoE-specific training knobs require --moe-n-experts > 1.")
    if (
        int(getattr(args, "moe_gate_tau_anneal_epochs", 0)) > 0
        and float(getattr(args, "moe_gate_tau_start", -1.0)) <= 0.0
    ):
        raise ValueError("moe_gate_tau_anneal_epochs > 0 requires --moe-gate-tau-start > 0.")
    if (
        float(getattr(args, "moe_head_low_lambda", 0.0)) > 0.0
        or float(getattr(args, "moe_head_high_lambda", 0.0)) > 0.0
    ) and int(getattr(args, "moe_n_experts", 1)) < 2:
        raise ValueError("Head-specific MoE CE requires at least 2 experts.")
    if str(getattr(args, "moe_head_ce_source", "clean")).strip().lower() not in {"clean", "cls"}:
        raise ValueError("moe_head_ce_source must be one of: clean | cls.")
    if int(getattr(args, "contrastive_pretrain_epochs", 0)) > 0 and int(getattr(args, "moco_pretrain_epochs", 0)) > 0:
        raise ValueError("Use only one SSL pretrain mode: contrastive_pretrain_epochs OR moco_pretrain_epochs.")
    if int(getattr(args, "moco_pretrain_epochs", 0)) < 0:
        raise ValueError("moco_pretrain_epochs must be >= 0.")
    if int(getattr(args, "moco_pretrain_epochs", 0)) > 0 and args.arch not in ("cldnn", "multiview"):
        raise ValueError("MoCo pretraining currently supports only --arch cldnn or --arch multiview.")
    if float(getattr(args, "moco_temp", 0.20)) <= 0.0:
        raise ValueError("moco_temp must be > 0.")
    moco_m = float(getattr(args, "moco_momentum", 0.999))
    if moco_m < 0.0 or moco_m >= 1.0:
        raise ValueError("moco_momentum must be in [0, 1).")
    if int(getattr(args, "moco_queue_size", 16384)) <= 0:
        raise ValueError("moco_queue_size must be > 0.")
    if int(getattr(args, "moco_proj_dim", 128)) <= 0:
        raise ValueError("moco_proj_dim must be > 0.")
    if int(getattr(args, "moco_hidden_dim", 512)) <= 0:
        raise ValueError("moco_hidden_dim must be > 0.")
    if float(getattr(args, "ssl_aug_awgn_prob", 0.0)) < 0.0 or float(getattr(args, "ssl_aug_awgn_prob", 0.0)) > 1.0:
        raise ValueError("ssl_aug_awgn_prob must be in [0,1].")
    if float(getattr(args, "ssl_aug_time_mask_prob", 0.0)) < 0.0 or float(getattr(args, "ssl_aug_time_mask_prob", 0.0)) > 1.0:
        raise ValueError("ssl_aug_time_mask_prob must be in [0,1].")
    if float(getattr(args, "ssl_aug_iq_drop_prob", 0.0)) < 0.0 or float(getattr(args, "ssl_aug_iq_drop_prob", 0.0)) > 1.0:
        raise ValueError("ssl_aug_iq_drop_prob must be in [0,1].")
    if float(getattr(args, "ssl_aug_time_mask_max_frac", 0.12)) < 0.0 or float(getattr(args, "ssl_aug_time_mask_max_frac", 0.12)) > 1.0:
        raise ValueError("ssl_aug_time_mask_max_frac must be in [0,1].")
    if float(getattr(args, "ssl_aug_awgn_snr_min_db", 6.0)) > float(getattr(args, "ssl_aug_awgn_snr_max_db", 20.0)):
        raise ValueError("ssl_aug_awgn_snr_min_db must be <= ssl_aug_awgn_snr_max_db.")
    if float(getattr(args, "focal_gamma", 0.0)) > 0.0 and float(getattr(args, "label_smoothing", 0.0)) > 0.0:
        print("[warn] focal_gamma > 0 with label_smoothing > 0 can conflict; consider label_smoothing=0.0 for focal runs.")
    if bool(getattr(args, "snr_weight_ce", False)) and float(getattr(args, "low_snr_boost", 0.0)) > 0.0:
        print("[warn] Both --snr-weight-ce and --low-snr-boost are enabled; weights will multiply.")
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
            cldnn_backbone=str(getattr(args, "cldnn_backbone", "lstm")),
            cldnn_tcn_levels=int(getattr(args, "cldnn_tcn_levels", 6)),
            cldnn_tcn_channels=int(getattr(args, "cldnn_tcn_channels", 128)),
            cldnn_tcn_kernel=int(getattr(args, "cldnn_tcn_kernel", 3)),
            cldnn_tcn_dilation_base=int(getattr(args, "cldnn_tcn_dilation_base", 2)),
            cldnn_tcn_dropout=float(getattr(args, "cldnn_tcn_dropout", 0.15)),
            cldnn_resnet_blocks=int(getattr(args, "cldnn_resnet_blocks", 8)),
            cldnn_resnet_channels=int(getattr(args, "cldnn_resnet_channels", 128)),
            cldnn_resnet_kernel=int(getattr(args, "cldnn_resnet_kernel", 5)),
            cldnn_resnet_dilation_cycle=int(getattr(args, "cldnn_resnet_dilation_cycle", 4)),
            cldnn_resnet_dropout=float(getattr(args, "cldnn_resnet_dropout", 0.15)),
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
            moe_n_experts=int(getattr(args, "moe_n_experts", 1)),
            moe_gate_type=str(getattr(args, "moe_gate_type", "eta-sigmoid")),
            moe_gate_center=float(getattr(args, "moe_gate_center", 0.5)),
            moe_gate_tau=float(getattr(args, "moe_gate_tau", 0.3)),
            moe_gate_use_feat=bool(getattr(args, "moe_gate_use_feat", False)),
            supcon_proj_dim=int(getattr(args, 'supcon_proj_dim', 0)) if getattr(args, 'supcon', False) else 0,
            dn_diff_enable=bool(getattr(args, "dn_diff_enable", False)),
            dn_diff_target=str(getattr(args, "dn_diff_target", "v")),
            dn_diff_train_timesteps=int(getattr(args, "dn_diff_train_timesteps", 100)),
            dn_diff_beta_start=float(getattr(args, "dn_diff_beta_start", 1e-4)),
            dn_diff_beta_end=float(getattr(args, "dn_diff_beta_end", 2e-2)),
            dn_diff_train_t_start_source=str(getattr(args, "dn_diff_train_t_start_source", "snr_pred")),
            dn_diff_train_forward_mode=str(getattr(args, "dn_diff_train_forward_mode", "onestep")),
            dn_diff_eval_mode=str(getattr(args, "dn_diff_eval_mode", "onestep")),
            dn_diff_eval_steps=int(getattr(args, "dn_diff_eval_steps", 8)),
            dn_diff_ddim_eta=float(getattr(args, "dn_diff_ddim_eta", 0.0)),
            dn_diff_multisample=int(getattr(args, "dn_diff_multisample", 1)),
            dn_diff_eval_t_start_source=str(getattr(args, "dn_diff_eval_t_start_source", "snr_pred")),
            dn_diff_fixed_t_start=int(getattr(args, "dn_diff_fixed_t_start", 30)),
            dn_diff_snr2t_scale=float(getattr(args, "dn_diff_snr2t_scale", 1.0)),
            dn_diff_snr2t_bias=float(getattr(args, "dn_diff_snr2t_bias", 0.0)),
            dn_diff_detach_eta_cond=bool(getattr(args, "dn_diff_detach_eta_cond", True)),
            dn_diff_low_snr_thresh=float(getattr(args, "dn_diff_low_snr_thresh", -6.0)),
            dn_diff_high_snr_margin=float(getattr(args, "dn_diff_high_snr_margin", 2.0)),
            dn_diff_hard_bypass_high_snr=bool(getattr(args, "dn_diff_hard_bypass_high_snr", True)),
            dn_diff_cond_diagnostic=str(getattr(args, "dn_diff_cond_diagnostic", "none")),
            dn_diff_force_deterministic_multisample=bool(
                getattr(args, "dn_diff_force_deterministic_multisample", False)
            ),
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

    freeze_cfg = getattr(args, "dn_diff_freeze_classifier", None)
    if freeze_cfg is None:
        dn_diff_freeze_classifier_eff = bool(getattr(args, "dn_diff_enable", False))
    else:
        dn_diff_freeze_classifier_eff = bool(freeze_cfg)

    init_ckpt_path = str(getattr(args, "init_ckpt", "") or "").strip()
    has_init_weights = bool((args.ckpt is not None and args.resume) or init_ckpt_path)
    if dn_diff_freeze_classifier_eff and not (
        bool(getattr(args, "dn_diff_enable", False))
        or bool(getattr(args, "cldnn_denoiser", False))
    ):
        raise ValueError("--dn-diff-freeze-classifier requires dn_diff_enable or cldnn_denoiser path.")
    if (
        dn_diff_freeze_classifier_eff
        and not has_init_weights
        and not bool(getattr(args, "allow_random_frozen_classifier", False))
    ):
        raise ValueError(
            "Frozen classifier mode requires pretrained initialization "
            "(use --init-ckpt PATH or --ckpt PATH --resume). "
            "Use --allow-random-frozen-classifier only for debug."
        )

    freeze_cls_for_dn = dn_diff_freeze_classifier_eff
    if args.arch == "cldnn" and freeze_cls_for_dn:
        if bool(getattr(args, "dn_diff_enable", False)) and hasattr(model, "set_dn_diff_train_freeze"):
            model.set_dn_diff_train_freeze(True)
            if float(getattr(args, "lambda_noise", 0.0)) <= 0.0:
                noise_head = getattr(model, "noise_fraction_net", None)
                if noise_head is not None:
                    for p in noise_head.parameters():
                        p.requires_grad_(False)
        elif bool(getattr(args, "cldnn_denoiser", False)):
            # Matched frozen-control mode: keep denoiser/noise head trainable, freeze classifier stack.
            for p in model.parameters():
                p.requires_grad_(False)
            for name, p in model.named_parameters():
                if name.startswith("denoiser") or name.startswith("noise_fraction_net"):
                    p.requires_grad_(True)

    def _current_trainable_params() -> List[torch.nn.Parameter]:
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable model parameters are enabled.")
        return params

    optimizer = torch.optim.AdamW(_current_trainable_params(), lr=args.lr, weight_decay=args.weight_decay)
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
    init_ckpt_source_used = "none"

    if args.ckpt is not None and args.resume:
        ckpt = load_checkpoint(args.ckpt, model, optimizer=optimizer, scheduler=scheduler, ema=ema)
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("step", 0)
        init_ckpt_source_used = "resume"
    elif getattr(args, "init_ckpt", None):
        init_ckpt = str(getattr(args, "init_ckpt"))
        if not os.path.exists(init_ckpt):
            raise FileNotFoundError(f"init_ckpt not found: {init_ckpt}")
        try:
            init_blob = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            init_blob = torch.load(init_ckpt, map_location="cpu")
        init_pref = str(getattr(args, "init_ckpt_source", "auto")).strip().lower()
        init_state: Dict[str, torch.Tensor] | torch.Tensor
        if isinstance(init_blob, dict):
            ema_state = init_blob.get("ema")
            use_ema_init = False
            if init_pref == "ema":
                if not (isinstance(ema_state, dict) and len(ema_state) > 0):
                    raise ValueError("--init-ckpt-source ema requested, but checkpoint has no EMA state.")
                use_ema_init = True
            elif init_pref == "auto":
                use_ema_init = bool(dn_diff_freeze_classifier_eff) and isinstance(ema_state, dict) and len(ema_state) > 0
            if use_ema_init:
                init_state = ema_state
                init_ckpt_source_used = "ema"
            else:
                init_state = init_blob.get("model", init_blob)
                init_ckpt_source_used = "model"
        else:
            init_state = init_blob
            init_ckpt_source_used = "model"
        init_stats = load_state_dict_flexible(model, init_state, prefix="init-ckpt")
        print(
            f"[init-ckpt] loaded from {init_ckpt} "
            f"source={init_ckpt_source_used} "
            f"(loaded={init_stats.get('loaded', 0)}, skipped_shape={init_stats.get('skipped_shape', 0)})."
        )
        # Keep EMA consistent with warm-started weights.
        # Without this, eval can read stale/random EMA shadows for many steps.
        if ema is not None:
            ema = EMA.create(model, decay=args.ema_decay)
            print("[init-ckpt] EMA shadow reinitialized from loaded model state.")

    # --- Diffusion schedule coverage diagnostic ---
    if args.arch == "cldnn" and bool(getattr(args, "dn_diff_enable", False)):
        _dn_sched = getattr(model, "dn_diff_schedule", None)
        if _dn_sched is not None:
            import math as _math
            _ab_min = float(_dn_sched.alpha_bars[-1])
            _ab_max = float(_dn_sched.alpha_bars[0])
            # SNR (dB) at which snr_to_t() saturates: alpha_bar_min = SNR_lin / (SNR_lin + 1)
            # => SNR_lin = alpha_bar_min / (1 - alpha_bar_min)
            _snr_lin_min = _ab_min / max(1.0 - _ab_min, 1e-12)
            _snr_db_min = 10.0 * _math.log10(max(_snr_lin_min, 1e-12))
            _loss_lo = float(getattr(args, "dn_diff_loss_snr_lo", -14.0))
            _loss_hi = float(getattr(args, "dn_diff_loss_snr_hi", -6.0))
            _covered = _snr_db_min <= _loss_lo
            print(
                f"[dn_diff] schedule coverage: T={_dn_sched.timesteps}, "
                f"beta=[{model.dn_diff_beta_start:.1e}, {model.dn_diff_beta_end:.1e}], "
                f"alpha_bar=[{_ab_min:.4f}, {_ab_max:.4f}], "
                f"SNR_min_covered={_snr_db_min:.1f} dB"
            )
            if not _covered:
                print(
                    f"[dn_diff] WARNING: schedule alpha_bar_min={_ab_min:.4f} "
                    f"only covers SNR >= {_snr_db_min:.1f} dB, "
                    f"but loss range is [{_loss_lo}, {_loss_hi}] dB. "
                    f"All samples below {_snr_db_min:.1f} dB will clamp to t={_dn_sched.timesteps - 1}. "
                    f"Consider --dn-diff-beta-end 0.10 or --dn-diff-train-timesteps 500+."
                )

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
                cldnn_backbone=str(getattr(args, "cldnn_backbone", "lstm")),
                cldnn_tcn_levels=int(getattr(args, "cldnn_tcn_levels", 6)),
                cldnn_tcn_channels=int(getattr(args, "cldnn_tcn_channels", 128)),
                cldnn_tcn_kernel=int(getattr(args, "cldnn_tcn_kernel", 3)),
                cldnn_tcn_dilation_base=int(getattr(args, "cldnn_tcn_dilation_base", 2)),
                cldnn_tcn_dropout=float(getattr(args, "cldnn_tcn_dropout", 0.15)),
                cldnn_resnet_blocks=int(getattr(args, "cldnn_resnet_blocks", 8)),
                cldnn_resnet_channels=int(getattr(args, "cldnn_resnet_channels", 128)),
                cldnn_resnet_kernel=int(getattr(args, "cldnn_resnet_kernel", 5)),
                cldnn_resnet_dilation_cycle=int(getattr(args, "cldnn_resnet_dilation_cycle", 4)),
                cldnn_resnet_dropout=float(getattr(args, "cldnn_resnet_dropout", 0.15)),
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
                moe_n_experts=int(getattr(args, "moe_n_experts", 1)),
                moe_gate_type=str(getattr(args, "moe_gate_type", "eta-sigmoid")),
                moe_gate_center=float(getattr(args, "moe_gate_center", 0.5)),
                moe_gate_tau=float(getattr(args, "moe_gate_tau", 0.3)),
                moe_gate_use_feat=bool(getattr(args, "moe_gate_use_feat", False)),
                supcon_proj_dim=int(getattr(args, "supcon_proj_dim", 0)) if getattr(args, "supcon", False) else 0,
                dn_diff_enable=bool(getattr(args, "dn_diff_enable", False)),
                dn_diff_target=str(getattr(args, "dn_diff_target", "v")),
                dn_diff_train_timesteps=int(getattr(args, "dn_diff_train_timesteps", 100)),
                dn_diff_beta_start=float(getattr(args, "dn_diff_beta_start", 1e-4)),
                dn_diff_beta_end=float(getattr(args, "dn_diff_beta_end", 2e-2)),
                dn_diff_train_t_start_source=str(getattr(args, "dn_diff_train_t_start_source", "snr_pred")),
                dn_diff_train_forward_mode=str(getattr(args, "dn_diff_train_forward_mode", "onestep")),
                dn_diff_eval_mode=str(getattr(args, "dn_diff_eval_mode", "onestep")),
                dn_diff_eval_steps=int(getattr(args, "dn_diff_eval_steps", 8)),
                dn_diff_ddim_eta=float(getattr(args, "dn_diff_ddim_eta", 0.0)),
                dn_diff_multisample=int(getattr(args, "dn_diff_multisample", 1)),
                dn_diff_eval_t_start_source=str(getattr(args, "dn_diff_eval_t_start_source", "snr_pred")),
                dn_diff_fixed_t_start=int(getattr(args, "dn_diff_fixed_t_start", 30)),
                dn_diff_snr2t_scale=float(getattr(args, "dn_diff_snr2t_scale", 1.0)),
                dn_diff_snr2t_bias=float(getattr(args, "dn_diff_snr2t_bias", 0.0)),
                dn_diff_detach_eta_cond=bool(getattr(args, "dn_diff_detach_eta_cond", True)),
                dn_diff_low_snr_thresh=float(getattr(args, "dn_diff_low_snr_thresh", -6.0)),
                dn_diff_high_snr_margin=float(getattr(args, "dn_diff_high_snr_margin", 2.0)),
                dn_diff_hard_bypass_high_snr=bool(getattr(args, "dn_diff_hard_bypass_high_snr", True)),
                dn_diff_cond_diagnostic=str(getattr(args, "dn_diff_cond_diagnostic", "none")),
                dn_diff_force_deterministic_multisample=bool(
                    getattr(args, "dn_diff_force_deterministic_multisample", False)
                ),
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
    early_stop_start_epoch = int(getattr(args, "early_stop_start_epoch", 0))

    # =========================================================================
    # MOCO-V2 PRE-TRAINING PHASE (if enabled)
    # =========================================================================
    if args.moco_pretrain_epochs > 0 and args.arch in ("cldnn", "multiview"):
        ssl_pretrain_path = os.path.join(args.out_dir, "ssl_pretrain.jsonl")
        if os.path.exists(ssl_pretrain_path):
            os.remove(ssl_pretrain_path)

        print(f"\n{'='*60}")
        print(
            "MOCO PRE-TRAINING: "
            f"{args.moco_pretrain_epochs} epochs, T={float(args.moco_temp):.3f}, "
            f"m={float(args.moco_momentum):.4f}, queue={int(args.moco_queue_size)}"
        )
        print(f"{'='*60}\n")

        if str(getattr(args, "dataset", "rml2016a")) == "rml2018a":
            X_cl, y_cl, snr_cl, _mods_cl, _snrs_cl, train_idx_cl, _val_idx_cl, _test_idx_cl = load_rml2018a_hdf5(
                args.data_path, seed=args.seed, train_per=args.train_per, val_per=args.val_per
            )
        else:
            X_cl, y_cl, snr_cl, _mods_cl, _snrs_cl, train_idx_cl, _val_idx_cl, _test_idx_cl = load_rml2016a(
                args.data_path, seed=args.seed, train_per=args.train_per, val_per=args.val_per
            )
        train_snrs_cl = parse_snrs(getattr(args, "train_snrs", None))
        train_idx_cl = filter_indices_by_snrs(train_idx_cl, snr_cl, train_snrs_cl)
        X_t_cl, y_t_cl, snr_t_cl = build_tensors(X_cl, y_cl, snr_cl)
        moco_dataset = RML2016aContrastivePairDataset(
            X_t_cl,
            y_t_cl,
            snr_t_cl,
            train_idx_cl,
            normalize=args.normalize,
            aug_phase=args.aug_phase,
            aug_shift=args.aug_shift,
            aug_gain=args.aug_gain,
            aug_cfo=args.aug_cfo,
            ssl_aug_awgn_prob=float(getattr(args, "ssl_aug_awgn_prob", 0.0)),
            ssl_aug_awgn_snr_min_db=float(getattr(args, "ssl_aug_awgn_snr_min_db", 6.0)),
            ssl_aug_awgn_snr_max_db=float(getattr(args, "ssl_aug_awgn_snr_max_db", 20.0)),
            ssl_aug_time_mask_prob=float(getattr(args, "ssl_aug_time_mask_prob", 0.0)),
            ssl_aug_time_mask_max_frac=float(getattr(args, "ssl_aug_time_mask_max_frac", 0.12)),
            ssl_aug_iq_drop_prob=float(getattr(args, "ssl_aug_iq_drop_prob", 0.0)),
        )
        moco_loader = DataLoader(
            moco_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            worker_init_fn=_seed_worker,
        )
        if len(moco_loader) <= 0:
            raise RuntimeError("MoCo pretraining loader is empty. Reduce batch size or adjust split settings.")

        # Probe feature dimension.
        probe_batch = next(iter(moco_loader))
        x_probe = probe_batch[0].to(device)
        snr_probe = probe_batch[3].to(device)
        snr_probe_in = snr_probe if args.snr_mode == "known" else None
        with torch.no_grad():
            feat_probe = model.forward_features(x_probe, snr=snr_probe_in, snr_mode=args.snr_mode)
        feat_dim = int(feat_probe.shape[1])
        del probe_batch, x_probe, snr_probe, snr_probe_in, feat_probe

        moco_q_proj = MoCoProjectionMLP(
            in_dim=feat_dim,
            hidden_dim=int(getattr(args, "moco_hidden_dim", 512)),
            out_dim=int(getattr(args, "moco_proj_dim", 128)),
        ).to(device)
        moco_k_proj = MoCoProjectionMLP(
            in_dim=feat_dim,
            hidden_dim=int(getattr(args, "moco_hidden_dim", 512)),
            out_dim=int(getattr(args, "moco_proj_dim", 128)),
        ).to(device)
        moco_k_proj.load_state_dict(moco_q_proj.state_dict())
        for p in moco_k_proj.parameters():
            p.requires_grad_(False)
        moco_k_proj.eval()

        moco_key_encoder = copy.deepcopy(model).to(device)
        moco_key_encoder.load_state_dict(model.state_dict())
        for p in moco_key_encoder.parameters():
            p.requires_grad_(False)
        moco_key_encoder.eval()

        queue_size = int(getattr(args, "moco_queue_size", 16384))
        proj_dim = int(getattr(args, "moco_proj_dim", 128))
        moco_queue = F.normalize(torch.randn(queue_size, proj_dim, device=device), dim=1)
        moco_queue_ptr = 0

        moco_lr = float(args.moco_lr) if args.moco_lr is not None else float(args.lr)
        moco_wd = float(args.moco_weight_decay) if args.moco_weight_decay is not None else float(args.weight_decay)
        moco_optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(moco_q_proj.parameters()),
            lr=moco_lr,
            weight_decay=moco_wd,
        )
        moco_steps = int(args.moco_pretrain_epochs) * len(moco_loader)
        moco_scheduler = build_scheduler(
            moco_optimizer,
            args.warmup_steps,
            moco_steps,
            args.min_lr,
            decay_start_step=0,
        )

        for moco_epoch in range(int(args.moco_pretrain_epochs)):
            model.train()
            moco_q_proj.train()
            moco_key_encoder.eval()
            moco_k_proj.eval()
            moco_epoch_loss = 0.0
            moco_epoch_pos = 0.0
            moco_epoch_neg = 0.0
            moco_epoch_total = 0
            moco_start = time.time()

            progress = tqdm(
                moco_loader,
                desc=f"MoCo {moco_epoch + 1}/{int(args.moco_pretrain_epochs)}",
                unit="batch",
                dynamic_ncols=True,
            )
            for batch in progress:
                x_q, x_k, _y_ssl, snr_ssl = batch[:4]
                x_q = x_q.to(device)
                x_k = x_k.to(device)
                snr_ssl = snr_ssl.to(device)
                snr_ssl_in = snr_ssl if args.snr_mode == "known" else None

                moco_optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    moco_momentum_update(model, moco_key_encoder, float(getattr(args, "moco_momentum", 0.999)))
                    moco_momentum_update(moco_q_proj, moco_k_proj, float(getattr(args, "moco_momentum", 0.999)))

                if amp_enabled:
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                        q_feat = model.forward_features(x_q, snr=snr_ssl_in, snr_mode=args.snr_mode)
                        q = F.normalize(moco_q_proj(q_feat), dim=1)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True), torch.no_grad():
                        k_feat = moco_key_encoder.forward_features(x_k, snr=snr_ssl_in, snr_mode=args.snr_mode)
                        k = F.normalize(moco_k_proj(k_feat), dim=1)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                        l_pos = torch.sum(q * k, dim=1, keepdim=True)
                        l_neg = torch.matmul(q, moco_queue.detach().T)
                        logits_moco = torch.cat([l_pos, l_neg], dim=1) / max(1e-6, float(getattr(args, "moco_temp", 0.20)))
                        target_moco = torch.zeros(logits_moco.shape[0], dtype=torch.long, device=device)
                        loss_moco = F.cross_entropy(logits_moco, target_moco)
                else:
                    q_feat = model.forward_features(x_q, snr=snr_ssl_in, snr_mode=args.snr_mode)
                    q = F.normalize(moco_q_proj(q_feat), dim=1)
                    with torch.no_grad():
                        k_feat = moco_key_encoder.forward_features(x_k, snr=snr_ssl_in, snr_mode=args.snr_mode)
                        k = F.normalize(moco_k_proj(k_feat), dim=1)
                    l_pos = torch.sum(q * k, dim=1, keepdim=True)
                    l_neg = torch.matmul(q, moco_queue.detach().T)
                    logits_moco = torch.cat([l_pos, l_neg], dim=1) / max(1e-6, float(getattr(args, "moco_temp", 0.20)))
                    target_moco = torch.zeros(logits_moco.shape[0], dtype=torch.long, device=device)
                    loss_moco = F.cross_entropy(logits_moco, target_moco)

                if scaler.is_enabled():
                    scaler.scale(loss_moco).backward()
                    scaler.unscale_(moco_optimizer)
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(moco_q_proj.parameters()), args.grad_clip)
                    scaler.step(moco_optimizer)
                    scaler.update()
                else:
                    loss_moco.backward()
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(moco_q_proj.parameters()), args.grad_clip)
                    moco_optimizer.step()

                moco_scheduler.step()

                with torch.no_grad():
                    moco_queue_ptr = moco_enqueue(moco_queue, moco_queue_ptr, k)

                bsz = int(x_q.shape[0])
                moco_epoch_loss += float(loss_moco.detach().item()) * bsz
                moco_epoch_pos += float(l_pos.detach().mean().item()) * bsz
                moco_epoch_neg += float(l_neg.detach().mean().item()) * bsz
                moco_epoch_total += bsz
                progress.set_postfix(
                    loss=f"{moco_epoch_loss / max(1, moco_epoch_total):.4f}",
                    lr=f"{moco_optimizer.param_groups[0]['lr']:.2e}",
                    qptr=int(moco_queue_ptr),
                )

            moco_epoch_loss_mean = moco_epoch_loss / max(1, moco_epoch_total)
            moco_epoch_pos_mean = moco_epoch_pos / max(1, moco_epoch_total)
            moco_epoch_neg_mean = moco_epoch_neg / max(1, moco_epoch_total)
            write_jsonl(
                ssl_pretrain_path,
                {
                    "phase": "moco",
                    "epoch": int(moco_epoch),
                    "train_loss": float(moco_epoch_loss_mean),
                    "mean_pos_logit": float(moco_epoch_pos_mean),
                    "mean_neg_logit": float(moco_epoch_neg_mean),
                    "lr": float(moco_optimizer.param_groups[0]["lr"]),
                    "queue_ptr": int(moco_queue_ptr),
                    "moco_temp": float(getattr(args, "moco_temp", 0.20)),
                    "moco_momentum": float(getattr(args, "moco_momentum", 0.999)),
                    "moco_queue_size": int(getattr(args, "moco_queue_size", 16384)),
                    "moco_proj_dim": int(getattr(args, "moco_proj_dim", 128)),
                    "moco_hidden_dim": int(getattr(args, "moco_hidden_dim", 512)),
                    "time_sec": float(time.time() - moco_start),
                },
            )
            tqdm.write(
                "MoCo epoch "
                f"{moco_epoch + 1}: loss={moco_epoch_loss_mean:.4f}, "
                f"pos={moco_epoch_pos_mean:.4f}, neg={moco_epoch_neg_mean:.4f}"
            )

        del moco_key_encoder, moco_q_proj, moco_k_proj, moco_queue

        print(f"\n{'='*60}")
        print("MOCO PRE-TRAINING COMPLETE. Starting fine-tuning...")
        print(f"{'='*60}\n")

        # Reset optimizer for fine-tuning phase
        optimizer = torch.optim.AdamW(_current_trainable_params(), lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = max(1, len(train_loader))
        total_steps = args.epochs * steps_per_epoch
        decay_start_step = int(getattr(args, "lr_decay_start_epoch", 0)) * steps_per_epoch
        scheduler = build_scheduler(
            optimizer, args.warmup_steps, total_steps, args.min_lr, decay_start_step=decay_start_step
        )

    # =========================================================================
    # LEGACY CONTRASTIVE PRE-TRAINING PHASE (if enabled)
    # =========================================================================
    if args.contrastive_pretrain_epochs > 0 and args.arch in ("cldnn", "multiview"):
        print(f"\n{'='*60}")
        print(f"CONTRASTIVE PRE-TRAINING: {args.contrastive_pretrain_epochs} epochs, K={args.contrastive_k}")
        print(f"{'='*60}\n")

        if str(getattr(args, "dataset", "rml2016a")) == "rml2018a":
            X_cl, y_cl, snr_cl, _mods_cl, _snrs_cl, train_idx_cl, _val_idx_cl, _test_idx_cl = load_rml2018a_hdf5(
                args.data_path, seed=args.seed, train_per=args.train_per, val_per=args.val_per
            )
        else:
            X_cl, y_cl, snr_cl, _mods_cl, _snrs_cl, train_idx_cl, _val_idx_cl, _test_idx_cl = load_rml2016a(
                args.data_path, seed=args.seed, train_per=args.train_per, val_per=args.val_per
            )
        train_snrs_cl = parse_snrs(getattr(args, "train_snrs", None))
        train_idx_cl = filter_indices_by_snrs(train_idx_cl, snr_cl, train_snrs_cl)
        X_t_cl, y_t_cl, snr_t_cl = build_tensors(X_cl, y_cl, snr_cl)
        contrastive_dataset = RML2016aGroupedDataset(
            X_t_cl,
            y_t_cl,
            snr_t_cl,
            train_idx_cl,
            group_k=args.contrastive_k,
            normalize=args.normalize,
            aug_phase=args.aug_phase,
            aug_shift=args.aug_shift,
            aug_gain=args.aug_gain,
            aug_cfo=args.aug_cfo,
        )
        contrastive_loader = DataLoader(
            contrastive_dataset,
            batch_size=max(1, args.batch_size // max(1, args.contrastive_k)),
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            worker_init_fn=_seed_worker,
        )
        if len(contrastive_loader) <= 0:
            raise RuntimeError("Contrastive pretraining loader is empty. Reduce batch size or adjust split settings.")

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
                x = x.to(device)  # (B, K, 2, L)
                y = y.to(device)
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
                epoch_loss += float(loss.detach().item()) * x.shape[0]
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
        optimizer = torch.optim.AdamW(_current_trainable_params(), lr=args.lr, weight_decay=args.weight_decay)
        steps_per_epoch = max(1, len(train_loader))
        total_steps = args.epochs * steps_per_epoch
        decay_start_step = int(getattr(args, "lr_decay_start_epoch", 0)) * steps_per_epoch
        scheduler = build_scheduler(
            optimizer, args.warmup_steps, total_steps, args.min_lr, decay_start_step=decay_start_step
        )

    dn_diff_align_teacher_mode = str(getattr(args, "dn_diff_align_teacher", "frozen")).strip().lower()
    dn_diff_align_teacher: Optional[torch.nn.Module] = None
    dn_diff_align_teacher_momentum = float(getattr(args, "ema_decay", 0.9996))
    dn_diff_align_teacher_needed = (
        bool(getattr(args, "dn_diff_enable", False))
        and dn_diff_align_teacher_mode in {"frozen", "ema"}
        and (
            float(getattr(args, "lambda_dn_feat_align", 0.0)) > 0.0
            or float(getattr(args, "lambda_dn_logit_align", 0.0)) > 0.0
        )
    )
    if dn_diff_align_teacher_needed:
        dn_diff_align_teacher = copy.deepcopy(model).to(device)
        dn_diff_align_teacher.eval()
        for p in dn_diff_align_teacher.parameters():
            p.requires_grad_(False)
        print(f"[dn_diff] alignment teacher initialized ({dn_diff_align_teacher_mode}).")

    for epoch in range(start_epoch, args.epochs):
        phase1 = epoch < args.phase1_epochs
        lambda_diff = args.phase1_lambda_diff if phase1 else args.lambda_diff
        p_clean = args.phase1_p_clean if phase1 else args.p_clean
        cls2dn_scale = get_cls2dn_scale(epoch, args)
        lambda_dn_diff_eff_cfg = get_lambda_dn_diff(epoch, args)
        lambda_dn_cls_eff_cfg = get_lambda_dn_cls(epoch, args)
        cls_loss_mult = 1.0
        if bool(getattr(args, "cldnn_denoiser", False)) and bool(getattr(args, "stage_a_no_cls", False)):
            if epoch < int(getattr(args, "stage_a_epochs", 0)):
                cls_loss_mult = 0.0
        if bool(dn_diff_freeze_classifier_eff):
            cls_loss_mult = 0.0
        lambda_snr_eff = 0.0 if bool(getattr(args, "dn_diff_enable", False)) else float(args.lambda_snr)
        lambda_noise_eff = float(getattr(args, "lambda_noise", 0.0))
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
        if bool(dn_diff_freeze_classifier_eff):
            if hasattr(model, "set_frozen_classifier_train_mode"):
                model.set_frozen_classifier_train_mode(True)
            elif bool(getattr(args, "dn_diff_enable", False)) and hasattr(model, "set_dn_diff_train_mode"):
                model.set_dn_diff_train_mode(True)
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
        epoch_loss_moe_balance = 0.0
        epoch_loss_moe_specialize = 0.0
        epoch_loss_moe_entropy_warm = 0.0
        epoch_loss_moe_head_low = 0.0
        epoch_loss_moe_head_high = 0.0
        epoch_loss_moe_diversity = 0.0
        epoch_loss_supcon = 0.0
        epoch_supcon_active = 0.0
        epoch_loss_dn_diff = 0.0
        epoch_loss_dn_recon = 0.0
        epoch_loss_dn_cls = 0.0
        epoch_loss_dn_feat_align = 0.0
        epoch_loss_dn_logit_align = 0.0
        epoch_dn_t_start_mean = 0.0
        epoch_dn_t_start_std = 0.0
        epoch_dn_active_low = 0.0
        epoch_dn_active_mid = 0.0
        epoch_dn_active_high = 0.0
        epoch_moe_low_active = 0.0
        epoch_moe_high_active = 0.0
        moe_n_experts_cfg = int(getattr(args, "moe_n_experts", 1))
        epoch_moe_expert_load = [0.0 for _ in range(moe_n_experts_cfg)] if moe_n_experts_cfg > 1 else []
        moe_gate_tau_base = float(getattr(args, "moe_gate_tau", 0.3))
        moe_gate_tau_curr = moe_gate_tau_base
        moe_gate_tau_start = float(getattr(args, "moe_gate_tau_start", -1.0))
        moe_gate_tau_anneal_epochs = int(getattr(args, "moe_gate_tau_anneal_epochs", 0))
        if (
            args.arch == "cldnn"
            and moe_n_experts_cfg > 1
            and moe_gate_tau_start > 0.0
            and moe_gate_tau_anneal_epochs > 0
        ):
            tau_prog = min(1.0, float(epoch) / float(max(1, moe_gate_tau_anneal_epochs)))
            moe_gate_tau_curr = moe_gate_tau_start + tau_prog * (moe_gate_tau_base - moe_gate_tau_start)
        moe_gate_tau_scale = moe_gate_tau_curr / max(1e-6, moe_gate_tau_base)
        moe_oracle_gate_train = bool(getattr(args, "moe_oracle_gate_train", False))
        moe_head_ce_detach_trunk = bool(getattr(args, "moe_head_ce_detach_trunk", False))
        moe_head_ce_source = str(getattr(args, "moe_head_ce_source", "clean")).strip().lower()
        moe_head_ce_warmup = int(getattr(args, "moe_head_ce_warmup", 0))
        moe_head_ce_ramp = int(getattr(args, "moe_head_ce_ramp", 0))
        if epoch < moe_head_ce_warmup:
            moe_head_ce_scale = 0.0
        elif moe_head_ce_ramp > 0:
            ramp_prog = float(epoch - moe_head_ce_warmup + 1) / float(max(1, moe_head_ce_ramp))
            moe_head_ce_scale = max(0.0, min(1.0, ramp_prog))
        else:
            moe_head_ce_scale = 1.0
        moe_head_low_lambda_eff = float(getattr(args, "moe_head_low_lambda", 0.0)) * moe_head_ce_scale
        moe_head_high_lambda_eff = float(getattr(args, "moe_head_high_lambda", 0.0)) * moe_head_ce_scale
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
            curriculum_mask_aux = curriculum_mask_base if (not use_mixup or mixup_cls_only) else curriculum_mask_cls

            optimizer.zero_grad(set_to_none=True)
            snr_in_aux = snr_aux if args.snr_mode == "known" else None
            snr_in_cls = snr_cls if args.snr_mode == "known" else None
            snr_in = snr_in_aux
            extra_cls2dn = {"cls_to_denoiser_scale": cls2dn_scale} if args.arch == "cldnn" else {}
            def _moe_kwargs(snr_for_gate: Optional[torch.Tensor]) -> Dict[str, object]:
                if args.arch != "cldnn" or moe_n_experts_cfg <= 1:
                    return {}
                kw: Dict[str, object] = {
                    "moe_gate_tau_scale": float(moe_gate_tau_scale),
                }
                if moe_oracle_gate_train:
                    kw["moe_use_oracle_gate"] = True
                    kw["moe_oracle_snr"] = snr_for_gate
                if moe_head_ce_detach_trunk:
                    kw["moe_head_ce_detach_trunk"] = True
                return kw

            def _moe_logits_for_aux(model_obj: torch.nn.Module) -> Optional[torch.Tensor]:
                if moe_head_ce_detach_trunk:
                    logits_aux = getattr(model_obj, "_moe_logits_experts_aux", None)
                    if logits_aux is not None:
                        return logits_aux
                return getattr(model_obj, "_moe_logits_experts", None)

            def _capture_dn_diff_cache(model_obj: torch.nn.Module) -> Optional[Dict[str, torch.Tensor]]:
                pred_c = getattr(model_obj, "_dn_diff_train_pred_flat", None)
                target_c = getattr(model_obj, "_dn_diff_train_target_flat", None)
                x0_c = getattr(model_obj, "_dn_diff_train_x0_flat", None)
                t_c = getattr(model_obj, "_dn_diff_t_start", None)
                if (
                    isinstance(pred_c, torch.Tensor)
                    and isinstance(target_c, torch.Tensor)
                    and isinstance(x0_c, torch.Tensor)
                    and isinstance(t_c, torch.Tensor)
                ):
                    return {
                        "pred": pred_c,
                        "target": target_c,
                        "x0": x0_c,
                        "t": t_c,
                    }
                return None

            moe_kwargs_aux = _moe_kwargs(snr_aux)
            moe_kwargs_cls = _moe_kwargs(snr_cls)
            dn_diff_cache_for_loss: Optional[Dict[str, torch.Tensor]] = None
            logits_for_acc = None
            logits_teacher_base = None
            moe_logits_experts_clean: Optional[torch.Tensor] = None
            moe_logits_experts_cls: Optional[torch.Tensor] = None
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
            loss_moe_balance = torch.tensor(0.0, device=device)
            loss_moe_specialize = torch.tensor(0.0, device=device)
            loss_moe_entropy_warm = torch.tensor(0.0, device=device)
            loss_moe_head_low = torch.tensor(0.0, device=device)
            loss_moe_head_high = torch.tensor(0.0, device=device)
            loss_moe_diversity = torch.tensor(0.0, device=device)
            loss_supcon = torch.tensor(0.0, device=device)
            supcon_active_frac_batch = torch.tensor(0.0, device=device)
            loss_dn_diff = torch.tensor(0.0, device=device)
            loss_dn_recon = torch.tensor(0.0, device=device)
            loss_dn_cls = torch.tensor(0.0, device=device)
            loss_dn_feat_align = torch.tensor(0.0, device=device)
            loss_dn_logit_align = torch.tensor(0.0, device=device)
            dn_t_start_mean_batch = torch.tensor(0.0, device=device)
            dn_t_start_std_batch = torch.tensor(0.0, device=device)
            dn_active_low_batch = torch.tensor(0.0, device=device)
            dn_active_mid_batch = torch.tensor(0.0, device=device)
            dn_active_high_batch = torch.tensor(0.0, device=device)
            moe_low_active_frac_batch = torch.tensor(0.0, device=device)
            moe_high_active_frac_batch = torch.tensor(0.0, device=device)
            moe_load_batch: Optional[torch.Tensor] = None
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
                                **moe_kwargs_aux,
                            )
                            if bool(getattr(args, "dn_diff_enable", False)):
                                dn_diff_cache_for_loss = _capture_dn_diff_cache(model)
                            moe_logits_experts_clean = _moe_logits_for_aux(model)
                            student_prefilm_kd = getattr(model, "_pooled_pre_film", None)
                            eta_pred_for_noise = getattr(model, "_eta_pred", None)
                            logits, _x0_pred_mix, _snr_pred_mix = model(
                                x_cls,
                                t,
                                snr=snr_in_cls,
                                snr_mode=args.snr_mode,
                                group_mask=mask,
                                **extra_cls2dn,
                                **moe_kwargs_cls,
                            )
                            moe_logits_experts_cls = _moe_logits_for_aux(model)
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
                                **moe_kwargs_cls,
                            )
                            if bool(getattr(args, "dn_diff_enable", False)):
                                dn_diff_cache_for_loss = _capture_dn_diff_cache(model)
                            moe_logits_experts_cls = _moe_logits_for_aux(model)
                            moe_logits_experts_clean = moe_logits_experts_cls
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
                    if bool(getattr(args, "snr_weight_ce", False)):
                        ce = ce * snr_weighted_ce_weights(
                            snr_cls,
                            snr_min_db=snr_min_db,
                            snr_max_db=snr_max_db,
                            scale=float(getattr(args, "snr_weight_ce_scale", 2.0)),
                            max_weight=float(getattr(args, "snr_weight_ce_max", 3.0)),
                        )

                    # Apply curriculum mask (zero out samples below SNR threshold)
                    ce = ce * curriculum_mask_cls
                    denom_curriculum = torch.clamp(curriculum_mask_cls.sum(), min=1.0)
                    loss_cls = ce.sum() / denom_curriculum

                    loss_snr = F.smooth_l1_loss(snr_pred, snr_aux) if args.lambda_snr > 0 else 0.0
                    loss_noise = 0.0
                    if float(lambda_noise_eff) > 0.0:
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
                        + float(lambda_snr_eff) * loss_snr
                        + float(lambda_noise_eff) * loss_noise
                    )
                    if args.arch == "cldnn" and moe_n_experts_cfg > 1:
                        moe_logits_experts_batch = (
                            moe_logits_experts_clean if moe_head_ce_source == "clean" else moe_logits_experts_cls
                        )
                        if moe_logits_experts_batch is None:
                            moe_logits_experts_batch = (
                                moe_logits_experts_cls if moe_head_ce_source == "clean" else moe_logits_experts_clean
                            )
                        (
                            loss_moe_head_low,
                            loss_moe_head_high,
                            moe_low_active_frac_batch,
                            moe_high_active_frac_batch,
                        ) = compute_moe_head_specialization_losses(
                            moe_logits_experts_batch,
                            y=y,
                            snr_db=snr_cls,
                            curriculum_mask=curriculum_mask_cls,
                            use_mixup=use_mixup,
                            y_a=y_a,
                            y_b=y_b,
                            lam=lam,
                            low_head_idx=int(getattr(args, "moe_low_head_idx", 0)),
                            high_head_idx=int(getattr(args, "moe_high_head_idx", 1)),
                            low_lambda=float(moe_head_low_lambda_eff),
                            high_lambda=float(moe_head_high_lambda_eff),
                            low_snr_lo=float(getattr(args, "moe_head_low_snr_lo", -14.0)),
                            low_snr_hi=float(getattr(args, "moe_head_low_snr_hi", 2.0)),
                            high_snr_lo=float(getattr(args, "moe_head_high_snr_lo", -6.0)),
                            high_snr_hi=float(getattr(args, "moe_head_high_snr_hi", 18.0)),
                            label_smoothing=float(args.label_smoothing),
                        )
                        if moe_logits_experts_batch is not None:
                            loss_moe_diversity = compute_moe_diversity_loss(
                                moe_logits_experts_batch,
                                diversity_lambda=float(getattr(args, "moe_diversity_lambda", 0.0)),
                            )
                        loss = loss + loss_moe_head_low + loss_moe_head_high + loss_moe_diversity
                    if (
                        bool(getattr(args, "cldnn_denoiser", False))
                        and args.arch == "cldnn"
                        and not bool(getattr(args, "dn_diff_enable", False))
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
                    # Optional clean-branch path under mixup_cls_only.
                    supcon_clean_on_mixup = bool(getattr(args, "supcon_clean_branch_on_mixup_cls_only", False))
                    supcon_mixup_ok = (not use_mixup) or (supcon_clean_on_mixup and mixup_cls_only)
                    if (
                        getattr(args, "supcon", False)
                        and args.arch != "dit"
                        and epoch >= int(getattr(args, "supcon_warmup", 0))
                        and supcon_mixup_ok
                        and hasattr(model, "proj_head")
                        and model.proj_head is not None
                    ):
                        if use_mixup and mixup_cls_only and supcon_clean_on_mixup:
                            supcon_feat = student_prefilm_kd
                        else:
                            supcon_feat = getattr(model, "_pooled_pre_film", None)
                        if supcon_feat is not None:
                            z_proj = model.proj_head(supcon_feat)
                            z_proj = F.normalize(z_proj, dim=1)
                            loss_supcon = supervised_contrastive_loss(
                                z_proj,
                                y,
                                temperature=float(getattr(args, "supcon_temp", 0.07)),
                            )
                            loss = loss + float(getattr(args, "supcon_lambda", 0.1)) * loss_supcon
                            supcon_active_frac_batch = torch.tensor(1.0, device=device)

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
                                    **moe_kwargs_aux,
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
                            **_moe_kwargs(snr_new),
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
                            **moe_kwargs_aux,
                        )
                        if bool(getattr(args, "dn_diff_enable", False)):
                            dn_diff_cache_for_loss = _capture_dn_diff_cache(model)
                        moe_logits_experts_clean = _moe_logits_for_aux(model)
                        student_prefilm_kd = getattr(model, "_pooled_pre_film", None)
                        eta_pred_for_noise = getattr(model, "_eta_pred", None)
                        logits, _x0_pred_mix, _snr_pred_mix = model(
                            x_cls,
                            t,
                            snr=snr_in_cls,
                            snr_mode=args.snr_mode,
                            group_mask=mask,
                            **extra_cls2dn,
                            **moe_kwargs_cls,
                        )
                        moe_logits_experts_cls = _moe_logits_for_aux(model)
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
                            **moe_kwargs_cls,
                        )
                        if bool(getattr(args, "dn_diff_enable", False)):
                            dn_diff_cache_for_loss = _capture_dn_diff_cache(model)
                        moe_logits_experts_cls = _moe_logits_for_aux(model)
                        moe_logits_experts_clean = moe_logits_experts_cls
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
                if bool(getattr(args, "snr_weight_ce", False)):
                    ce = ce * snr_weighted_ce_weights(
                        snr_cls,
                        snr_min_db=snr_min_db,
                        snr_max_db=snr_max_db,
                        scale=float(getattr(args, "snr_weight_ce_scale", 2.0)),
                        max_weight=float(getattr(args, "snr_weight_ce_max", 3.0)),
                    )

                # Apply curriculum mask (zero out samples below SNR threshold)
                ce = ce * curriculum_mask_cls
                denom_curriculum = torch.clamp(curriculum_mask_cls.sum(), min=1.0)
                loss_cls = ce.sum() / denom_curriculum

                loss_snr = F.smooth_l1_loss(snr_pred, snr_aux) if args.lambda_snr > 0 else 0.0
                loss_noise = 0.0
                if float(lambda_noise_eff) > 0.0:
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
                    + float(lambda_snr_eff) * loss_snr
                    + float(lambda_noise_eff) * loss_noise
                )
                if args.arch == "cldnn" and moe_n_experts_cfg > 1:
                    moe_logits_experts_batch = (
                        moe_logits_experts_clean if moe_head_ce_source == "clean" else moe_logits_experts_cls
                    )
                    if moe_logits_experts_batch is None:
                        moe_logits_experts_batch = (
                            moe_logits_experts_cls if moe_head_ce_source == "clean" else moe_logits_experts_clean
                        )
                    (
                        loss_moe_head_low,
                        loss_moe_head_high,
                        moe_low_active_frac_batch,
                        moe_high_active_frac_batch,
                    ) = compute_moe_head_specialization_losses(
                        moe_logits_experts_batch,
                        y=y,
                        snr_db=snr_cls,
                        curriculum_mask=curriculum_mask_cls,
                        use_mixup=use_mixup,
                        y_a=y_a,
                        y_b=y_b,
                        lam=lam,
                        low_head_idx=int(getattr(args, "moe_low_head_idx", 0)),
                        high_head_idx=int(getattr(args, "moe_high_head_idx", 1)),
                        low_lambda=float(moe_head_low_lambda_eff),
                        high_lambda=float(moe_head_high_lambda_eff),
                        low_snr_lo=float(getattr(args, "moe_head_low_snr_lo", -14.0)),
                        low_snr_hi=float(getattr(args, "moe_head_low_snr_hi", 2.0)),
                        high_snr_lo=float(getattr(args, "moe_head_high_snr_lo", -6.0)),
                        high_snr_hi=float(getattr(args, "moe_head_high_snr_hi", 18.0)),
                        label_smoothing=float(args.label_smoothing),
                    )
                    if moe_logits_experts_batch is not None:
                        loss_moe_diversity = compute_moe_diversity_loss(
                            moe_logits_experts_batch,
                            diversity_lambda=float(getattr(args, "moe_diversity_lambda", 0.0)),
                        )
                    loss = loss + loss_moe_head_low + loss_moe_head_high + loss_moe_diversity
                if (
                    bool(getattr(args, "cldnn_denoiser", False))
                    and args.arch == "cldnn"
                    and not bool(getattr(args, "dn_diff_enable", False))
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
                # Optional clean-branch path under mixup_cls_only.
                supcon_clean_on_mixup = bool(getattr(args, "supcon_clean_branch_on_mixup_cls_only", False))
                supcon_mixup_ok = (not use_mixup) or (supcon_clean_on_mixup and mixup_cls_only)
                if (
                    getattr(args, "supcon", False)
                    and args.arch != "dit"
                    and epoch >= int(getattr(args, "supcon_warmup", 0))
                    and supcon_mixup_ok
                    and hasattr(model, "proj_head")
                    and model.proj_head is not None
                ):
                    if use_mixup and mixup_cls_only and supcon_clean_on_mixup:
                        supcon_feat = student_prefilm_kd
                    else:
                        supcon_feat = getattr(model, "_pooled_pre_film", None)
                    if supcon_feat is not None:
                        z_proj = model.proj_head(supcon_feat)
                        z_proj = F.normalize(z_proj, dim=1)
                        loss_supcon = supervised_contrastive_loss(
                            z_proj,
                            y,
                            temperature=float(getattr(args, "supcon_temp", 0.07)),
                        )
                        loss = loss + float(getattr(args, "supcon_lambda", 0.1)) * loss_supcon
                        supcon_active_frac_batch = torch.tensor(1.0, device=device)

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
                                **moe_kwargs_aux,
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
                        **_moe_kwargs(snr_new),
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

            if bool(getattr(args, "dn_diff_enable", False)) and args.arch == "cldnn":
                if x_aux.ndim == 4:
                    g_dn = x_aux.shape[1]
                    x_dn_ref = x_aux.reshape(-1, x_aux.shape[2], x_aux.shape[3])
                    snr_dn_ref = snr_aux.repeat_interleave(g_dn)
                else:
                    x_dn_ref = x_aux
                    snr_dn_ref = snr_aux

                dn_loss_cond_source = str(getattr(args, "dn_diff_loss_cond_source", "raw")).strip().lower()
                if dn_loss_cond_source == "degraded":
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
                    x_cond_dn, _delta_dn = snr_path_degrade(
                        x_dn_ref,
                        snr_dn_ref,
                        delta_min=dn_pair_delta_min,
                        delta_max=dn_pair_delta_max,
                        snr_floor=dn_pair_snr_floor,
                        snr_target_min=dn_pair_snr_new_lo,
                        snr_target_max=dn_pair_snr_new_hi,
                    )
                else:
                    x_cond_dn = x_dn_ref

                pred_dn: Optional[torch.Tensor] = None
                target_dn: Optional[torch.Tensor] = None
                x0_dn: Optional[torch.Tensor] = None
                t_dn: Optional[torch.Tensor] = None
                # Reuse denoiser tensors from the already-computed classifier forward
                # to avoid a decoupled second denoiser pass when possible.
                if dn_diff_cache_for_loss is not None:
                    pred_c = dn_diff_cache_for_loss.get("pred")
                    target_c = dn_diff_cache_for_loss.get("target")
                    x0_c = dn_diff_cache_for_loss.get("x0")
                    t_c = dn_diff_cache_for_loss.get("t")
                    if (
                        isinstance(pred_c, torch.Tensor)
                        and isinstance(target_c, torch.Tensor)
                        and isinstance(x0_c, torch.Tensor)
                        and isinstance(t_c, torch.Tensor)
                        and int(pred_c.shape[0]) == int(x_dn_ref.shape[0])
                        and dn_loss_cond_source == "raw"
                    ):
                        pred_dn = pred_c
                        target_dn = target_c
                        x0_dn = x0_c
                        t_dn = t_c

                if pred_dn is None or target_dn is None or x0_dn is None or t_dn is None:
                    if hasattr(model, "_dn_diff_train_t_start"):
                        t_dn = model._dn_diff_train_t_start(x_raw=x_dn_ref, snr_flat=snr_dn_ref)  # type: ignore[attr-defined]
                    else:
                        train_t_source = str(getattr(args, "dn_diff_train_t_start_source", "snr_pred")).strip().lower()
                        if train_t_source == "fixed":
                            t_dn = torch.full(
                                (x_dn_ref.shape[0],),
                                int(getattr(args, "dn_diff_fixed_t_start", 30)),
                                device=device,
                                dtype=torch.long,
                            )
                            if hasattr(model, "dn_diff_schedule") and getattr(model, "dn_diff_schedule", None) is not None:
                                t_dn = torch.clamp(t_dn, min=0, max=int(model.dn_diff_schedule.timesteps) - 1)
                        else:
                            snr_est_dn: Optional[torch.Tensor] = None
                            if getattr(model, "noise_fraction_net", None) is not None:
                                eta_est_dn, _aux_eta = model.noise_fraction_net(x_dn_ref)  # type: ignore[attr-defined]
                                snr_est_dn = model._snr_from_eta(eta_est_dn.detach())  # type: ignore[attr-defined]
                            if snr_est_dn is None:
                                snr_est_dn = snr_dn_ref.float()
                            t_dn = model._dn_diff_map_snr_to_t(snr_est_dn)  # type: ignore[attr-defined]

                    if getattr(model, "dn_diff_schedule", None) is None:
                        raise RuntimeError("dn_diff_enable requires model.dn_diff_schedule.")
                    dn_schedule = model.dn_diff_schedule.to(device)  # type: ignore[attr-defined]
                    noise_dn = torch.randn_like(x_dn_ref)
                    x_t_dn = dn_schedule.q_sample(x_dn_ref, t_dn, noise_dn)
                    snr_dn_in = snr_dn_ref if args.snr_mode == "known" else None
                    pred_dn, _eta_pred_dn, _eta_cond_dn = model.dn_diff_predict(  # type: ignore[attr-defined]
                        x_t=x_t_dn,
                        x_cond=x_cond_dn,
                        t=t_dn,
                        snr=snr_dn_in,
                        snr_mode=args.snr_mode,
                        cond_diagnostic="none",
                    )

                    dn_target_type = str(getattr(args, "dn_diff_target", "v")).strip().lower()
                    if dn_target_type == "v":
                        alpha_bar_dn = dn_schedule.alpha_bars.gather(0, t_dn).view(-1, 1, 1)
                        target_dn = torch.sqrt(torch.clamp(alpha_bar_dn, min=0.0)) * noise_dn - torch.sqrt(
                            torch.clamp(1.0 - alpha_bar_dn, min=0.0)
                        ) * x_dn_ref
                        x0_dn = dn_schedule.predict_x0_from_v(x_t_dn, t_dn, pred_dn)
                    else:
                        target_dn = noise_dn
                        x0_dn = dn_schedule.predict_x0_from_eps(x_t_dn, t_dn, pred_dn)

                diff_vec = torch.mean((pred_dn.float() - target_dn.float()) ** 2, dim=(1, 2))
                recon_vec = torch.mean(torch.abs(x0_dn.float() - x_dn_ref.float()), dim=(1, 2))
                if bool(getattr(args, "dn_diff_apply_lowband_only_train", True)):
                    dn_loss_mask = (
                        (snr_dn_ref.float() >= float(getattr(args, "dn_diff_loss_snr_lo", -14.0)))
                        & (snr_dn_ref.float() <= float(getattr(args, "dn_diff_loss_snr_hi", -6.0)))
                    ).float()
                else:
                    dn_loss_mask = torch.ones_like(snr_dn_ref.float())
                dn_denom = torch.clamp(dn_loss_mask.sum(), min=1.0)
                loss_dn_diff = (diff_vec * dn_loss_mask).sum() / dn_denom
                loss_dn_recon = (recon_vec * dn_loss_mask).sum() / dn_denom
                loss = loss + float(lambda_dn_diff_eff_cfg) * loss_dn_diff
                loss = loss + float(getattr(args, "lambda_dn_recon", 0.0)) * loss_dn_recon

                # -- Shared classifier forward on denoised x0 --
                # Used by both dn-cls loss and logit alignment; computed once
                # to avoid a redundant classifier pass.
                _need_dn_cls = float(lambda_dn_cls_eff_cfg) > 0.0
                _need_dn_logit_align = (
                    float(getattr(args, "lambda_dn_logit_align", 0.0)) > 0.0
                    and epoch >= int(getattr(args, "dn_diff_logit_align_start_epoch", 30))
                )
                logits_dn_shared = None
                if _need_dn_cls or _need_dn_logit_align:
                    if x_aux.ndim == 4:
                        x0_dn_cls = x0_dn.view(x_aux.shape[0], x_aux.shape[1], x_aux.shape[2], x_aux.shape[3])
                    else:
                        x0_dn_cls = x0_dn
                    t_dn_cls = torch.zeros((x_aux.shape[0],), device=device, dtype=torch.long)
                    _dn_cls_ctx = nullcontext()
                    if bool(dn_diff_freeze_classifier_eff):
                        model_dn_cls = model.module if hasattr(model, "module") else model
                        backbone_name = str(getattr(model_dn_cls, "cldnn_backbone", "")).strip().lower()
                        if backbone_name == "lstm":
                            # Frozen-classifier mode keeps the stack in eval(); disable cuDNN
                            # for this auxiliary pass so RNN input-gradient backprop is valid.
                            _dn_cls_ctx = torch.backends.cudnn.flags(enabled=False)
                    _amp_ctx_cls = (
                        torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
                        if amp_enabled
                        else nullcontext()
                    )
                    with _dn_cls_ctx, _amp_ctx_cls:
                        logits_dn_shared, _, _ = model(
                            x0_dn_cls,
                            t_dn_cls,
                            snr=snr_in_aux,
                            snr_mode=args.snr_mode,
                            group_mask=mask,
                            denoiser_bypass=True,
                            **moe_kwargs_aux,
                        )

                if _need_dn_cls:
                    if use_mixup and (not mixup_cls_only):
                        ce_dn_a = focal_cross_entropy(
                            logits_dn_shared,
                            y_a,
                            gamma=focal_gamma,
                            label_smoothing=float(args.label_smoothing),
                        )
                        ce_dn_b = focal_cross_entropy(
                            logits_dn_shared,
                            y_b,
                            gamma=focal_gamma,
                            label_smoothing=float(args.label_smoothing),
                        )
                        ce_dn = lam * ce_dn_a + (1.0 - lam) * ce_dn_b
                    else:
                        ce_dn = focal_cross_entropy(
                            logits_dn_shared,
                            y,
                            gamma=focal_gamma,
                            label_smoothing=float(args.label_smoothing),
                        )
                    ce_dn = ce_dn * curriculum_mask_aux
                    denom_dn_cls = torch.clamp(curriculum_mask_aux.sum(), min=1.0)
                    loss_dn_cls = ce_dn.sum() / denom_dn_cls
                    loss = loss + float(lambda_dn_cls_eff_cfg) * loss_dn_cls

                if (
                    float(getattr(args, "lambda_dn_feat_align", 0.0)) > 0.0
                    and epoch >= int(getattr(args, "dn_diff_feat_align_start_epoch", 20))
                    and hasattr(model, "early_features")
                ):
                    if bool(getattr(model, "has_feat_encoder", False)) is False and hasattr(model, "build_feat_encoder"):
                        model.build_feat_encoder()
                    feat_in_pred = x0_dn.float()
                    feat_in_tgt = x_dn_ref.float()
                    if bool(getattr(model, "denoiser_dual_path", False)):
                        feat_in_pred = torch.cat([x_cond_dn.float(), x0_dn.float()], dim=1)
                        feat_in_tgt = torch.cat([x_cond_dn.float(), x_dn_ref.float()], dim=1)
                    feat_pred_dn = model.early_features(feat_in_pred)
                    if dn_diff_align_teacher is not None and hasattr(dn_diff_align_teacher, "early_features"):
                        if (
                            hasattr(dn_diff_align_teacher, "build_feat_encoder")
                            and bool(getattr(dn_diff_align_teacher, "has_feat_encoder", False)) is False
                        ):
                            dn_diff_align_teacher.build_feat_encoder()
                        with torch.no_grad():
                            feat_tgt_dn = dn_diff_align_teacher.early_features(feat_in_tgt)
                    else:
                        feat_tgt_dn = model.early_features(feat_in_tgt).detach()
                    feat_vec_dn = torch.mean(torch.abs(feat_pred_dn - feat_tgt_dn), dim=1)
                    loss_dn_feat_align = (feat_vec_dn * dn_loss_mask).sum() / dn_denom
                    loss = loss + float(getattr(args, "lambda_dn_feat_align", 0.0)) * loss_dn_feat_align

                if _need_dn_logit_align:
                    align_teacher_model = dn_diff_align_teacher if dn_diff_align_teacher is not None else model
                    with torch.no_grad():
                        logits_dn_tgt, _, _ = align_teacher_model(
                            x_aux,
                            t_dn_cls,
                            snr=snr_in_aux,
                            snr_mode=args.snr_mode,
                            group_mask=mask,
                            denoiser_bypass=True,
                        )
                    logit_vec_dn = torch.mean((logits_dn_shared.float() - logits_dn_tgt.detach().float()) ** 2, dim=1)
                    if bool(getattr(args, "dn_diff_apply_lowband_only_train", True)):
                        dn_logit_mask = (
                            (snr_aux.float() >= float(getattr(args, "dn_diff_loss_snr_lo", -14.0)))
                            & (snr_aux.float() <= float(getattr(args, "dn_diff_loss_snr_hi", -6.0)))
                        ).float()
                    else:
                        dn_logit_mask = torch.ones_like(snr_aux.float())
                    dn_logit_denom = torch.clamp(dn_logit_mask.sum(), min=1.0)
                    loss_dn_logit_align = (logit_vec_dn * dn_logit_mask).sum() / dn_logit_denom
                    loss = loss + float(getattr(args, "lambda_dn_logit_align", 0.0)) * loss_dn_logit_align

                dn_t_start_mean_batch = t_dn.float().mean()
                dn_t_start_std_batch = t_dn.float().std(unbiased=False)
                low_mask_dn = (snr_dn_ref.float() >= -14.0) & (snr_dn_ref.float() <= -6.0)
                mid_mask_dn = (snr_dn_ref.float() > -6.0) & (snr_dn_ref.float() < 6.0)
                high_mask_dn = snr_dn_ref.float() >= 6.0
                dn_active_vec = dn_loss_mask
                if bool(torch.any(low_mask_dn)):
                    dn_active_low_batch = dn_active_vec[low_mask_dn].mean()
                if bool(torch.any(mid_mask_dn)):
                    dn_active_mid_batch = dn_active_vec[mid_mask_dn].mean()
                if bool(torch.any(high_mask_dn)):
                    dn_active_high_batch = dn_active_vec[high_mask_dn].mean()

            if args.arch == "cldnn" and int(getattr(args, "moe_n_experts", 1)) > 1:
                gate_batch = getattr(model, "_moe_gate", None)
                moe_spec_lambda_cfg = float(getattr(args, "moe_specialize_lambda", 0.0))
                moe_spec_start = int(getattr(args, "moe_specialize_start_epoch", 0))
                moe_spec_lambda_eff = moe_spec_lambda_cfg if epoch >= moe_spec_start else 0.0
                moe_entropy_warm_lambda_cfg = float(getattr(args, "moe_entropy_warmup_lambda", 0.0))
                moe_entropy_warm_epochs = int(getattr(args, "moe_entropy_warmup_epochs", 0))
                moe_entropy_warm_eff = 0.0
                if moe_entropy_warm_lambda_cfg > 0.0:
                    if moe_entropy_warm_epochs > 0:
                        warm_frac = 1.0 - (float(epoch) / float(max(1, moe_entropy_warm_epochs)))
                        warm_frac = max(0.0, min(1.0, warm_frac))
                        moe_entropy_warm_eff = moe_entropy_warm_lambda_cfg * warm_frac
                    else:
                        moe_entropy_warm_eff = moe_entropy_warm_lambda_cfg
                (
                    loss_moe_balance,
                    loss_moe_specialize,
                    loss_moe_entropy_warm,
                    moe_load_batch,
                ) = compute_moe_regularizers(
                    gate_batch,
                    balance_lambda=float(getattr(args, "moe_balance_lambda", 0.01)),
                    specialize_lambda=moe_spec_lambda_eff,
                    entropy_warm_lambda=moe_entropy_warm_eff,
                )
                if gate_batch is not None:
                    loss = loss + loss_moe_balance + loss_moe_specialize + loss_moe_entropy_warm

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
            if dn_diff_align_teacher is not None and dn_diff_align_teacher_mode == "ema":
                moco_momentum_update(
                    model,
                    dn_diff_align_teacher,
                    momentum=dn_diff_align_teacher_momentum,
                )

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
            epoch_loss_moe_balance += float(loss_moe_balance.detach().item()) * batch_size
            epoch_loss_moe_specialize += float(loss_moe_specialize.detach().item()) * batch_size
            epoch_loss_moe_entropy_warm += float(loss_moe_entropy_warm.detach().item()) * batch_size
            epoch_loss_moe_head_low += float(loss_moe_head_low.detach().item()) * batch_size
            epoch_loss_moe_head_high += float(loss_moe_head_high.detach().item()) * batch_size
            epoch_loss_moe_diversity += float(loss_moe_diversity.detach().item()) * batch_size
            epoch_loss_supcon += float(loss_supcon.detach().item()) * batch_size
            epoch_supcon_active += float(supcon_active_frac_batch.detach().item()) * batch_size
            epoch_loss_dn_diff += float(loss_dn_diff.detach().item()) * batch_size
            epoch_loss_dn_recon += float(loss_dn_recon.detach().item()) * batch_size
            epoch_loss_dn_cls += float(loss_dn_cls.detach().item()) * batch_size
            epoch_loss_dn_feat_align += float(loss_dn_feat_align.detach().item()) * batch_size
            epoch_loss_dn_logit_align += float(loss_dn_logit_align.detach().item()) * batch_size
            epoch_dn_t_start_mean += float(dn_t_start_mean_batch.detach().item()) * batch_size
            epoch_dn_t_start_std += float(dn_t_start_std_batch.detach().item()) * batch_size
            epoch_dn_active_low += float(dn_active_low_batch.detach().item()) * batch_size
            epoch_dn_active_mid += float(dn_active_mid_batch.detach().item()) * batch_size
            epoch_dn_active_high += float(dn_active_high_batch.detach().item()) * batch_size
            epoch_moe_low_active += float(moe_low_active_frac_batch.detach().item()) * batch_size
            epoch_moe_high_active += float(moe_high_active_frac_batch.detach().item()) * batch_size
            if moe_load_batch is not None and len(epoch_moe_expert_load) == int(moe_load_batch.numel()):
                for i in range(len(epoch_moe_expert_load)):
                    epoch_moe_expert_load[i] += float(moe_load_batch[i].detach().item()) * batch_size
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
        train_loss_moe_balance = epoch_loss_moe_balance / max(1, epoch_total)
        train_loss_moe_specialize = epoch_loss_moe_specialize / max(1, epoch_total)
        train_loss_moe_entropy_warm = epoch_loss_moe_entropy_warm / max(1, epoch_total)
        train_loss_moe_head_low = epoch_loss_moe_head_low / max(1, epoch_total)
        train_loss_moe_head_high = epoch_loss_moe_head_high / max(1, epoch_total)
        train_loss_moe_diversity = epoch_loss_moe_diversity / max(1, epoch_total)
        train_loss_supcon = epoch_loss_supcon / max(1, epoch_total)
        train_supcon_active_frac = epoch_supcon_active / max(1, epoch_total)
        train_loss_dn_diff = epoch_loss_dn_diff / max(1, epoch_total)
        train_loss_dn_recon = epoch_loss_dn_recon / max(1, epoch_total)
        train_loss_dn_cls = epoch_loss_dn_cls / max(1, epoch_total)
        train_loss_dn_feat_align = epoch_loss_dn_feat_align / max(1, epoch_total)
        train_loss_dn_logit_align = epoch_loss_dn_logit_align / max(1, epoch_total)
        train_dn_t_start_mean = epoch_dn_t_start_mean / max(1, epoch_total)
        train_dn_t_start_std = epoch_dn_t_start_std / max(1, epoch_total)
        train_dn_active_frac_low = epoch_dn_active_low / max(1, epoch_total)
        train_dn_active_frac_mid = epoch_dn_active_mid / max(1, epoch_total)
        train_dn_active_frac_high = epoch_dn_active_high / max(1, epoch_total)
        train_moe_low_active_frac = epoch_moe_low_active / max(1, epoch_total)
        train_moe_high_active_frac = epoch_moe_high_active / max(1, epoch_total)
        train_moe_expert_load = [v / max(1, epoch_total) for v in epoch_moe_expert_load]
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

        use_ema_eval = ema is not None and global_step >= int(args.ema_start)
        if use_ema_eval:
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
                moe_oracle_gate_eval=bool(getattr(args, "moe_oracle_gate_eval", False)),
                moe_transition_snr_lo=float(getattr(args, "moe_transition_snr_lo", -8.0)),
                moe_transition_snr_hi=float(getattr(args, "moe_transition_snr_hi", -2.0)),
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
                moe_oracle_gate_eval=bool(getattr(args, "moe_oracle_gate_eval", False)),
                moe_transition_snr_lo=float(getattr(args, "moe_transition_snr_lo", -8.0)),
                moe_transition_snr_hi=float(getattr(args, "moe_transition_snr_hi", -2.0)),
            )

        noise_calib: Dict[str, object] = {}
        if args.arch == "cldnn" and (
            bool(getattr(args, "cldnn_noise_cond", False))
            or bool(getattr(args, "cldnn_denoiser", False))
        ):
            if use_ema_eval:
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
            "lambda_snr_eff": float(lambda_snr_eff),
            "lambda_noise": float(getattr(args, "lambda_noise", 0.0)),
            "lambda_noise_eff": float(lambda_noise_eff),
            "lambda_dn": float(getattr(args, "lambda_dn", 0.0)),
            "lambda_id": float(getattr(args, "lambda_id", 0.0)),
            "lambda_feat": float(lambda_feat),
            "lambda_dn_diff": float(getattr(args, "lambda_dn_diff", 1.0)),
            "lambda_dn_diff_eff": float(lambda_dn_diff_eff_cfg),
            "lambda_dn_recon": float(getattr(args, "lambda_dn_recon", 0.0)),
            "lambda_dn_cls": float(getattr(args, "lambda_dn_cls", 0.0)),
            "lambda_dn_cls_eff": float(lambda_dn_cls_eff_cfg),
            "lambda_dn_feat_align": float(getattr(args, "lambda_dn_feat_align", 0.0)),
            "lambda_dn_logit_align": float(getattr(args, "lambda_dn_logit_align", 0.0)),
            "lambda_kd": float(lambda_kd_eff),
            "lambda_kd_denoise": float(lambda_kd_denoise_eff),
            "lambda_kd_feat": float(lambda_kd_feat_eff),
            "train_loss_feat": float(train_loss_feat),
            "train_lfeat_active_frac": float(train_lfeat_active_frac),
            "train_loss_dn_diff": float(train_loss_dn_diff),
            "train_loss_dn_recon": float(train_loss_dn_recon),
            "train_loss_dn_cls": float(train_loss_dn_cls),
            "train_loss_dn_feat_align": float(train_loss_dn_feat_align),
            "train_loss_dn_logit_align": float(train_loss_dn_logit_align),
            "dn_diff_t_start_mean": float(train_dn_t_start_mean),
            "dn_diff_t_start_std": float(train_dn_t_start_std),
            "dn_diff_active_frac_low": float(train_dn_active_frac_low),
            "dn_diff_active_frac_mid": float(train_dn_active_frac_mid),
            "dn_diff_active_frac_high": float(train_dn_active_frac_high),
            "train_loss_kd": float(train_loss_kd),
            "train_kd_active_frac": float(train_kd_active_frac),
            "train_loss_kd_hi": float(train_loss_kd_hi),
            "train_kd_hi_active_frac": float(train_kd_hi_active_frac),
            "train_loss_kd_denoise": float(train_loss_kd_denoise),
            "train_kd_denoise_active_frac": float(train_kd_denoise_active_frac),
            "train_loss_kd_feat": float(train_loss_kd_feat),
            "train_kd_feat_active_frac": float(train_kd_feat_active_frac),
            "train_loss_moe_balance": float(train_loss_moe_balance),
            "train_loss_moe_specialize": float(train_loss_moe_specialize),
            "train_loss_moe_entropy_warm": float(train_loss_moe_entropy_warm),
            "train_loss_moe_head_low": float(train_loss_moe_head_low),
            "train_loss_moe_head_high": float(train_loss_moe_head_high),
            "train_loss_moe_diversity": float(train_loss_moe_diversity),
            "train_loss_supcon": float(train_loss_supcon),
            "train_supcon_active_frac": float(train_supcon_active_frac),
            "train_moe_low_active_frac": float(train_moe_low_active_frac),
            "train_moe_high_active_frac": float(train_moe_high_active_frac),
            "cls2dn_scale": float(cls2dn_scale),
            "p_clean": p_clean,
            "t_schedule": args.t_schedule,
            "snr_floor_db": args.snr_floor_db,
            "snr_cap_max_db": args.snr_cap_max_db,
            "low_snr_boost": args.low_snr_boost,
            "snr_weight_ce": bool(getattr(args, "snr_weight_ce", False)),
            "snr_weight_ce_scale": float(getattr(args, "snr_weight_ce_scale", 2.0)),
            "snr_weight_ce_max": float(getattr(args, "snr_weight_ce_max", 3.0)),
            "moco_pretrain_epochs": int(getattr(args, "moco_pretrain_epochs", 0)),
            "moco_temp": float(getattr(args, "moco_temp", 0.20)),
            "moco_momentum": float(getattr(args, "moco_momentum", 0.999)),
            "moco_queue_size": int(getattr(args, "moco_queue_size", 16384)),
            "moco_proj_dim": int(getattr(args, "moco_proj_dim", 128)),
            "moe_n_experts": int(getattr(args, "moe_n_experts", 1)),
            "moe_gate_type": str(getattr(args, "moe_gate_type", "eta-sigmoid")),
            "moe_gate_tau_train": float(moe_gate_tau_curr),
            "moe_balance_lambda": float(getattr(args, "moe_balance_lambda", 0.01)),
            "moe_specialize_lambda": float(getattr(args, "moe_specialize_lambda", 0.0)),
            "moe_specialize_start_epoch": int(getattr(args, "moe_specialize_start_epoch", 0)),
            "moe_entropy_warmup_lambda": float(getattr(args, "moe_entropy_warmup_lambda", 0.0)),
            "moe_entropy_warmup_epochs": int(getattr(args, "moe_entropy_warmup_epochs", 0)),
            "moe_gate_tau_start": float(getattr(args, "moe_gate_tau_start", -1.0)),
            "moe_gate_tau_anneal_epochs": int(getattr(args, "moe_gate_tau_anneal_epochs", 0)),
            "moe_oracle_gate_train": bool(getattr(args, "moe_oracle_gate_train", False)),
            "moe_oracle_gate_eval": bool(getattr(args, "moe_oracle_gate_eval", False)),
            "moe_low_head_idx": int(getattr(args, "moe_low_head_idx", 0)),
            "moe_high_head_idx": int(getattr(args, "moe_high_head_idx", 1)),
            "moe_head_low_lambda": float(getattr(args, "moe_head_low_lambda", 0.0)),
            "moe_head_high_lambda": float(getattr(args, "moe_head_high_lambda", 0.0)),
            "moe_head_low_lambda_eff": float(moe_head_low_lambda_eff),
            "moe_head_high_lambda_eff": float(moe_head_high_lambda_eff),
            "moe_head_ce_scale": float(moe_head_ce_scale),
            "moe_head_ce_warmup": int(getattr(args, "moe_head_ce_warmup", 0)),
            "moe_head_ce_ramp": int(getattr(args, "moe_head_ce_ramp", 0)),
            "moe_head_low_snr_lo": float(getattr(args, "moe_head_low_snr_lo", -14.0)),
            "moe_head_low_snr_hi": float(getattr(args, "moe_head_low_snr_hi", 2.0)),
            "moe_head_high_snr_lo": float(getattr(args, "moe_head_high_snr_lo", -6.0)),
            "moe_head_high_snr_hi": float(getattr(args, "moe_head_high_snr_hi", 18.0)),
            "moe_head_ce_detach_trunk": bool(getattr(args, "moe_head_ce_detach_trunk", False)),
            "moe_head_ce_source": str(getattr(args, "moe_head_ce_source", "clean")),
            "moe_transition_snr_lo": float(getattr(args, "moe_transition_snr_lo", -8.0)),
            "moe_transition_snr_hi": float(getattr(args, "moe_transition_snr_hi", -2.0)),
            "moe_diversity_lambda": float(getattr(args, "moe_diversity_lambda", 0.0)),
            "lr_decay_start_epoch": getattr(args, "lr_decay_start_epoch", 0),
            "curriculum_soft": bool(getattr(args, "curriculum_soft", False)),
            "curriculum_soft_low_weight": float(getattr(args, "curriculum_soft_low_weight", 0.1)),
            "mixup_snr_min": getattr(args, "mixup_snr_min", None),
            "mixup_cls_only": bool(getattr(args, "mixup_cls_only", True)),
            "early_stop_patience": int(getattr(args, "early_stop_patience", 0)),
            "early_stop_min_delta": float(getattr(args, "early_stop_min_delta", 0.0)),
            "early_stop_start_epoch": int(early_stop_start_epoch),
            "init_ckpt": str(getattr(args, "init_ckpt", "") or ""),
            "init_ckpt_source_used": str(init_ckpt_source_used),
            "allow_random_frozen_classifier": bool(getattr(args, "allow_random_frozen_classifier", False)),
            "cldnn_backbone": str(getattr(args, "cldnn_backbone", "lstm")),
            "cldnn_tcn_levels": int(getattr(args, "cldnn_tcn_levels", 6)),
            "cldnn_tcn_channels": int(getattr(args, "cldnn_tcn_channels", 128)),
            "cldnn_tcn_kernel": int(getattr(args, "cldnn_tcn_kernel", 3)),
            "cldnn_resnet_blocks": int(getattr(args, "cldnn_resnet_blocks", 8)),
            "cldnn_resnet_channels": int(getattr(args, "cldnn_resnet_channels", 128)),
            "cldnn_resnet_kernel": int(getattr(args, "cldnn_resnet_kernel", 5)),
            "cldnn_noise_cond": bool(getattr(args, "cldnn_noise_cond", False)),
            "cldnn_denoiser": bool(getattr(args, "cldnn_denoiser", False)),
            "dn_diff_enabled": bool(getattr(args, "dn_diff_enable", False)),
            "dn_diff_target": str(getattr(args, "dn_diff_target", "v")),
            "dn_diff_train_timesteps": int(getattr(args, "dn_diff_train_timesteps", 100)),
            "dn_diff_beta_start": float(getattr(args, "dn_diff_beta_start", 1e-4)),
            "dn_diff_beta_end": float(getattr(args, "dn_diff_beta_end", 2e-2)),
            "dn_diff_alpha_bar_min": float(getattr(model, "dn_diff_schedule", None).alpha_bars[-1]) if getattr(model, "dn_diff_schedule", None) is not None else 0.0,
            "dn_diff_train_t_source": str(getattr(args, "dn_diff_train_t_start_source", "snr_pred")),
            "dn_diff_train_forward_mode": str(getattr(args, "dn_diff_train_forward_mode", "onestep")),
            "dn_diff_eval_t_source": str(getattr(args, "dn_diff_eval_t_start_source", "snr_pred")),
            "dn_diff_eval_mode": str(getattr(args, "dn_diff_eval_mode", "onestep")),
            "dn_diff_eval_steps": int(getattr(args, "dn_diff_eval_steps", 8)),
            "dn_diff_ddim_eta": float(getattr(args, "dn_diff_ddim_eta", 0.0)),
            "dn_diff_multisample": int(getattr(args, "dn_diff_multisample", 1)),
            "dn_diff_allow_eval_ddim_mismatch": bool(getattr(args, "dn_diff_allow_eval_ddim_mismatch", False)),
            "dn_diff_require_noise_supervision": bool(getattr(args, "dn_diff_require_noise_supervision", True)),
            "dn_diff_snr2t_scale": float(getattr(args, "dn_diff_snr2t_scale", 1.0)),
            "dn_diff_snr2t_bias": float(getattr(args, "dn_diff_snr2t_bias", 0.0)),
            "dn_diff_detach_eta_cond": bool(getattr(args, "dn_diff_detach_eta_cond", True)),
            "dn_diff_low_snr_thresh": float(getattr(args, "dn_diff_low_snr_thresh", -6.0)),
            "dn_diff_high_snr_margin": float(getattr(args, "dn_diff_high_snr_margin", 2.0)),
            "dn_diff_hard_bypass_high_snr": bool(getattr(args, "dn_diff_hard_bypass_high_snr", True)),
            "dn_diff_apply_lowband_only_train": bool(getattr(args, "dn_diff_apply_lowband_only_train", True)),
            "dn_diff_loss_snr_lo": float(getattr(args, "dn_diff_loss_snr_lo", -14.0)),
            "dn_diff_loss_snr_hi": float(getattr(args, "dn_diff_loss_snr_hi", -6.0)),
            "dn_diff_loss_cond_source": str(getattr(args, "dn_diff_loss_cond_source", "raw")),
            "dn_diff_cls_warmup": int(getattr(args, "dn_diff_cls_warmup", 0)),
            "dn_diff_cls_ramp": int(getattr(args, "dn_diff_cls_ramp", 0)),
            "dn_diff_diff_warmup": int(getattr(args, "dn_diff_diff_warmup", 0)),
            "dn_diff_diff_ramp": int(getattr(args, "dn_diff_diff_ramp", 0)),
            "dn_diff_diff_final_scale": float(getattr(args, "dn_diff_diff_final_scale", 1.0)),
            "dn_diff_freeze_classifier": bool(dn_diff_freeze_classifier_eff),
            "dn_diff_align_teacher": str(getattr(args, "dn_diff_align_teacher", "frozen")),
            "dn_diff_feat_align_start_epoch": int(getattr(args, "dn_diff_feat_align_start_epoch", 20)),
            "dn_diff_logit_align_start_epoch": int(getattr(args, "dn_diff_logit_align_start_epoch", 30)),
            "dn_diff_cond_diagnostic": str(getattr(args, "dn_diff_cond_diagnostic", "none")),
        }
        for i, load_i in enumerate(train_moe_expert_load):
            record[f"train_moe_expert_{i}_load"] = float(load_i)
        if proxy_fit_info:
            record.update(proxy_fit_info)
        if noise_calib:
            record["eta_pearson"] = float(noise_calib.get("eta_pearson", 0.0))
            record["eta_spearman"] = float(noise_calib.get("eta_spearman", 0.0))
        for k, v in val_summary.items():
            if k.startswith("moe_") or k.startswith("dn_diff_"):
                record[f"val_{k}"] = float(v)
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
            # Persist epoch-by-epoch SNR diagnostics (append-only) for run-vs-run comparisons.
            val_acc_by_snr_hist = {str(k): float(v) for k, v in val_acc_by_snr.items()}
            write_jsonl(
                os.path.join(args.out_dir, "val_acc_by_snr_history.jsonl"),
                {
                    "epoch": int(epoch),
                    "val_acc": float(val_acc),
                    "val_macro_acc": float(val_summary.get("macro_acc", 0.0)),
                    "val_macro_f1": float(val_summary.get("macro_f1", 0.0)),
                    "val_low_macro_acc": float(val_summary.get("low_macro_acc", 0.0)),
                    "val_low_macro_f1": float(val_summary.get("low_macro_f1", 0.0)),
                    "val_acc_by_snr": val_acc_by_snr_hist,
                },
            )
        if noise_calib and noise_calib.get("eta_by_snr"):
            with open(os.path.join(args.out_dir, "eta_calibration_by_snr.json"), "w", encoding="utf-8") as f:
                json.dump(noise_calib["eta_by_snr"], f, indent=2)

        tqdm.write(
            f"Epoch {epoch + 1}/{args.epochs} summary | "
            f"train_acc(noised)={train_acc:.4f} "
            f"train_acc_clean~={train_acc_clean if train_acc_clean is not None else 'NA'} "
            f"val_acc={val_acc:.4f}"
        )

        if (
            args.early_stop_patience
            and args.early_stop_patience > 0
            and (early_stop_start_epoch <= 0 or (epoch + 1) >= early_stop_start_epoch)
        ):
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
        moe_oracle_gate_eval=bool(getattr(args, "moe_oracle_gate_eval", False)),
        moe_transition_snr_lo=float(getattr(args, "moe_transition_snr_lo", -8.0)),
        moe_transition_snr_hi=float(getattr(args, "moe_transition_snr_hi", -2.0)),
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
    if bool(getattr(args, "dn_diff_enable", False)) and args.arch != "cldnn":
        raise ValueError("dn_diff_enable is currently supported only for --arch cldnn.")
    if (
        bool(getattr(args, "dn_diff_enable", False))
        and str(getattr(args, "dn_diff_train_forward_mode", "onestep")).strip().lower() == "onestep"
        and str(getattr(args, "dn_diff_eval_mode", "onestep")).strip().lower() == "ddim"
        and not bool(getattr(args, "dn_diff_allow_eval_ddim_mismatch", False))
    ):
        raise ValueError(
            "dn_diff eval/training objective mismatch is blocked by default in eval-only mode too. "
            "Use --dn-diff-eval-mode onestep or pass --dn-diff-allow-eval-ddim-mismatch."
        )
    if bool(getattr(args, "cldnn_snr_cond", False)) and bool(getattr(args, "cldnn_noise_cond", False)):
        raise ValueError("Use only one conditioning path: --cldnn-snr-cond OR --cldnn-noise-cond.")
    if args.arch != "cldnn" and (
        bool(getattr(args, "cldnn_noise_cond", False)) or bool(getattr(args, "cldnn_denoiser", False))
    ):
        raise ValueError("Noise/denoiser options are currently supported only for --arch cldnn.")
    os.makedirs(args.out_dir, exist_ok=True)
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
            cldnn_backbone=str(getattr(args, "cldnn_backbone", "lstm")),
            cldnn_tcn_levels=int(getattr(args, "cldnn_tcn_levels", 6)),
            cldnn_tcn_channels=int(getattr(args, "cldnn_tcn_channels", 128)),
            cldnn_tcn_kernel=int(getattr(args, "cldnn_tcn_kernel", 3)),
            cldnn_tcn_dilation_base=int(getattr(args, "cldnn_tcn_dilation_base", 2)),
            cldnn_tcn_dropout=float(getattr(args, "cldnn_tcn_dropout", 0.15)),
            cldnn_resnet_blocks=int(getattr(args, "cldnn_resnet_blocks", 8)),
            cldnn_resnet_channels=int(getattr(args, "cldnn_resnet_channels", 128)),
            cldnn_resnet_kernel=int(getattr(args, "cldnn_resnet_kernel", 5)),
            cldnn_resnet_dilation_cycle=int(getattr(args, "cldnn_resnet_dilation_cycle", 4)),
            cldnn_resnet_dropout=float(getattr(args, "cldnn_resnet_dropout", 0.15)),
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
            moe_n_experts=int(getattr(args, "moe_n_experts", 1)),
            moe_gate_type=str(getattr(args, "moe_gate_type", "eta-sigmoid")),
            moe_gate_center=float(getattr(args, "moe_gate_center", 0.5)),
            moe_gate_tau=float(getattr(args, "moe_gate_tau", 0.3)),
            moe_gate_use_feat=bool(getattr(args, "moe_gate_use_feat", False)),
            supcon_proj_dim=int(getattr(args, 'supcon_proj_dim', 0)) if getattr(args, 'supcon', False) else 0,
            dn_diff_enable=bool(getattr(args, "dn_diff_enable", False)),
            dn_diff_target=str(getattr(args, "dn_diff_target", "v")),
            dn_diff_train_timesteps=int(getattr(args, "dn_diff_train_timesteps", 100)),
            dn_diff_beta_start=float(getattr(args, "dn_diff_beta_start", 1e-4)),
            dn_diff_beta_end=float(getattr(args, "dn_diff_beta_end", 2e-2)),
            dn_diff_train_t_start_source=str(getattr(args, "dn_diff_train_t_start_source", "snr_pred")),
            dn_diff_train_forward_mode=str(getattr(args, "dn_diff_train_forward_mode", "onestep")),
            dn_diff_eval_mode=str(getattr(args, "dn_diff_eval_mode", "onestep")),
            dn_diff_eval_steps=int(getattr(args, "dn_diff_eval_steps", 8)),
            dn_diff_ddim_eta=float(getattr(args, "dn_diff_ddim_eta", 0.0)),
            dn_diff_multisample=int(getattr(args, "dn_diff_multisample", 1)),
            dn_diff_eval_t_start_source=str(getattr(args, "dn_diff_eval_t_start_source", "snr_pred")),
            dn_diff_fixed_t_start=int(getattr(args, "dn_diff_fixed_t_start", 30)),
            dn_diff_snr2t_scale=float(getattr(args, "dn_diff_snr2t_scale", 1.0)),
            dn_diff_snr2t_bias=float(getattr(args, "dn_diff_snr2t_bias", 0.0)),
            dn_diff_detach_eta_cond=bool(getattr(args, "dn_diff_detach_eta_cond", True)),
            dn_diff_low_snr_thresh=float(getattr(args, "dn_diff_low_snr_thresh", -6.0)),
            dn_diff_high_snr_margin=float(getattr(args, "dn_diff_high_snr_margin", 2.0)),
            dn_diff_hard_bypass_high_snr=bool(getattr(args, "dn_diff_hard_bypass_high_snr", True)),
            dn_diff_cond_diagnostic=str(getattr(args, "dn_diff_cond_diagnostic", "none")),
            dn_diff_force_deterministic_multisample=bool(
                getattr(args, "dn_diff_force_deterministic_multisample", False)
            ),
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
        moe_oracle_gate_eval=bool(getattr(args, "moe_oracle_gate_eval", False)),
        moe_transition_snr_lo=float(getattr(args, "moe_transition_snr_lo", -8.0)),
        moe_transition_snr_hi=float(getattr(args, "moe_transition_snr_hi", -2.0)),
    )
    with open(os.path.join(args.out_dir, "test_acc_by_snr.json"), "w", encoding="utf-8") as f:
        json.dump(test_acc_by_snr, f, indent=2)
    with open(os.path.join(args.out_dir, "test_macro_summary.json"), "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2)
    print(f"Test acc: {test_acc:.4f}")
    print("Test acc by SNR:", test_acc_by_snr)
    print("Test macro summary:", test_summary)


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
