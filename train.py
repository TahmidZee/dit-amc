import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    load_rml2016a,
    parse_snrs,
    RML2016aDataset,
    RML2016aGroupedDataset,
    RML2016aVariableGroupedDataset,
)
from diffusion import DiffusionSchedule
from model import DiffusionAMC


PRESETS = {
    "S": {"patch_size": 8, "dim": 192, "depth": 10, "heads": 6, "lr": 2e-4},
    "B": {"patch_size": 4, "dim": 256, "depth": 12, "heads": 8, "lr": 1e-4},
}


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
    parser = argparse.ArgumentParser(description="Diffusion-regularized AMC (RML2016.10a)")
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/tahit/Modulation/RML2016.10a_dict.pkl",
    )
    parser.add_argument("--out-dir", type=str, default="./runs/dit_amc")
    parser.add_argument("--preset", type=str, choices=["S", "B"], default="S")

    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--stem-channels", type=int, default=64)
    parser.add_argument("--stem-layers", type=int, default=2)
    # Improved classifier architecture options
    parser.add_argument("--use-cls-token", action="store_true", help="Use learnable [CLS] token instead of mean pooling.")
    parser.add_argument("--cls-head-hidden", type=int, default=0, help="Hidden dim for MLP classifier (0 = single linear).")
    parser.add_argument("--cls-head-layers", type=int, default=1, help="Number of layers in classifier head.")
    parser.add_argument("--use-lstm", action="store_true", help="Add LSTM after Transformer for temporal modeling.")
    parser.add_argument("--lstm-hidden", type=int, default=128, help="LSTM hidden size (bidirectional, so 2x this).")
    parser.add_argument("--group-k", type=int, default=1, help="Number of random windows per (mod,SNR) bucket per sample.")
    parser.add_argument("--group-pool", type=str, choices=["mean", "attn"], default="mean", help="How to pool over group-k windows.")
    parser.add_argument("--k-max", type=int, default=None, help="Max windows for variable-K mode (defaults to --group-k).")
    parser.add_argument("--k-choices", type=str, default=None, help="Comma-separated K choices for variable-K training (e.g. 4,8,16).")
    parser.add_argument("--window-dropout", type=float, default=0.0, help="Extra random window dropout fraction applied on top of sampled K.")

    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2016)
    parser.add_argument("--train-per", type=int, default=600)
    parser.add_argument("--val-per", type=int, default=200)
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
        default=18.0,
        help="Upper SNR (dB) used for per-sample timestep cap scaling.",
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
    parser.add_argument("--snr-mode", type=str, choices=["predict", "known", "none"], default="predict")
    parser.add_argument("--snr-scale", type=float, default=20.0)
    parser.add_argument("--t-eval", type=int, default=0)

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
    X, y, snr, mods, snrs, train_idx, val_idx, test_idx = load_rml2016a(
        args.data_path,
        seed=args.seed,
        train_per=args.train_per,
        val_per=args.val_per,
    )
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
    return train_loader, val_loader, test_loader, mods, snrs


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr: float):
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
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


def write_jsonl(path: str, record: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    t_eval: int,
    snr_mode: str,
    amp: bool = False,
) -> Tuple[float, float, Dict[int, float]]:
    model.eval()
    total_correct = 0
    total = 0
    snr_correct: Dict[int, int] = {}
    snr_total: Dict[int, int] = {}

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

            snr_cpu = snr.detach().cpu().numpy().astype(np.int32)
            preds_cpu = preds.detach().cpu().numpy()
            y_cpu = y.detach().cpu().numpy()
            for snr_val, pred_val, y_val in zip(snr_cpu, preds_cpu, y_cpu):
                snr_key = int(snr_val)
                snr_correct[snr_key] = snr_correct.get(snr_key, 0) + int(pred_val == y_val)
                snr_total[snr_key] = snr_total.get(snr_key, 0) + 1

    acc = float(total_correct) / max(1, total)
    acc_by_snr = {snr: snr_correct[snr] / snr_total[snr] for snr in snr_total.keys()}
    return acc, total, acc_by_snr


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


def train(args: argparse.Namespace) -> None:
    apply_preset(args)
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

    train_loader, val_loader, test_loader, mods, snrs = build_loaders(args, device)
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
    model = DiffusionAMC(
        num_classes=len(mods),
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
        use_cls_token=args.use_cls_token,
        cls_head_hidden=args.cls_head_hidden,
        cls_head_layers=args.cls_head_layers,
        use_lstm=args.use_lstm,
        lstm_hidden=args.lstm_hidden,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_steps, args.min_lr)
    schedule = DiffusionSchedule(timesteps=args.timesteps).to(device)

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

    metrics_path = os.path.join(args.out_dir, "metrics.jsonl")
    best_val = -1.0
    best_epoch = None
    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs):
        phase1 = epoch < args.phase1_epochs
        lambda_diff = args.phase1_lambda_diff if phase1 else args.lambda_diff
        p_clean = args.phase1_p_clean if phase1 else args.p_clean
        model.train()
        epoch_loss = 0.0
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

            # --- Token-space diffusion: z_t = sqrt(a)*z0 + sqrt(1-a)*eps ---
            # z0 are patch embeddings BEFORE positional embedding; model adds pos internally.
            group_size = None
            if x.ndim == 4:
                group_size = x.shape[1]
                x_flat = x.reshape(-1, x.shape[2], x.shape[3])
            else:
                x_flat = x
            z0 = model.encode(x_flat)
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

            optimizer.zero_grad(set_to_none=True)
            snr_in = snr if args.snr_mode == "known" else None
            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                    logits, x0_pred, snr_pred = model.forward_tokens(
                        zt,
                        t,
                        snr=snr_in,
                        snr_mode=args.snr_mode,
                        group_size=group_size,
                        group_mask=mask,
                    )
                    if args.low_snr_boost and args.low_snr_boost > 0 and args.snr_floor_db is not None:
                        snr_floor = float(args.snr_floor_db)
                        snr_cap_max = float(args.snr_cap_max_db)
                        denom = max(1e-6, snr_cap_max - snr_floor)
                        snr_clamped = torch.clamp(snr.float(), min=snr_floor, max=snr_cap_max)
                        frac = (snr_clamped - snr_floor) / denom
                        weights = 1.0 + float(args.low_snr_boost) * (1.0 - frac)
                        ce = F.cross_entropy(logits, y, reduction="none", label_smoothing=float(args.label_smoothing))
                        loss_cls = torch.mean(ce * weights)
                    else:
                        loss_cls = F.cross_entropy(logits, y, label_smoothing=float(args.label_smoothing))
                    loss_diff = F.mse_loss(x0_pred, z0)
                    loss_snr = F.smooth_l1_loss(snr_pred, snr) if args.lambda_snr > 0 else 0.0
                    loss = loss_cls + lambda_diff * loss_diff + args.lambda_snr * loss_snr
            else:
                logits, x0_pred, snr_pred = model.forward_tokens(
                    zt,
                    t,
                    snr=snr_in,
                    snr_mode=args.snr_mode,
                    group_size=group_size,
                    group_mask=mask,
                )
                if args.low_snr_boost and args.low_snr_boost > 0 and args.snr_floor_db is not None:
                    snr_floor = float(args.snr_floor_db)
                    snr_cap_max = float(args.snr_cap_max_db)
                    denom = max(1e-6, snr_cap_max - snr_floor)
                    snr_clamped = torch.clamp(snr.float(), min=snr_floor, max=snr_cap_max)
                    frac = (snr_clamped - snr_floor) / denom
                    weights = 1.0 + float(args.low_snr_boost) * (1.0 - frac)
                    ce = F.cross_entropy(logits, y, reduction="none", label_smoothing=float(args.label_smoothing))
                    loss_cls = torch.mean(ce * weights)
                else:
                    loss_cls = F.cross_entropy(logits, y, label_smoothing=float(args.label_smoothing))
                loss_diff = F.mse_loss(x0_pred, z0)
                loss_snr = F.smooth_l1_loss(snr_pred, snr) if args.lambda_snr > 0 else 0.0
                loss = loss_cls + lambda_diff * loss_diff + args.lambda_snr * loss_snr

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

            epoch_loss += loss.item() * x.shape[0]
            epoch_correct += (logits.argmax(dim=1) == y).sum().item()
            epoch_total += x.shape[0]
            if epoch_total > 0:
                progress.set_postfix(
                    train_loss=f"{epoch_loss / epoch_total:.4f}",
                    train_acc=f"{epoch_correct / epoch_total:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        train_loss = epoch_loss / max(1, epoch_total)
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

        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
            val_acc, _, val_acc_by_snr = evaluate(
                model, val_loader, device, args.t_eval, args.snr_mode, amp=args.amp
            )
            ema.restore(model)
        else:
            val_acc, _, val_acc_by_snr = evaluate(
                model, val_loader, device, args.t_eval, args.snr_mode, amp=args.amp
            )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_acc_clean": train_acc_clean,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": time.time() - start_time,
            "lambda_diff": lambda_diff,
            "p_clean": p_clean,
            "t_schedule": args.t_schedule,
            "snr_floor_db": args.snr_floor_db,
            "snr_cap_max_db": args.snr_cap_max_db,
            "low_snr_boost": args.low_snr_boost,
        }
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
    test_acc, _, test_acc_by_snr = evaluate(
        model, test_loader, device, args.t_eval, args.snr_mode, amp=args.amp
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
    write_jsonl(
        metrics_path,
        {
            "epoch": best_epoch if best_epoch is not None else args.epochs,
            "test_acc": test_acc,
            "best_val_acc": best_val,
            **(dyn if dyn is not None else {}),
        },
    )


def run_eval(args: argparse.Namespace) -> None:
    apply_preset(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, mods, _ = build_loaders(args, device)

    model = DiffusionAMC(
        num_classes=len(mods),
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
        use_cls_token=args.use_cls_token,
        cls_head_hidden=args.cls_head_hidden,
        cls_head_layers=args.cls_head_layers,
        use_lstm=args.use_lstm,
        lstm_hidden=args.lstm_hidden,
    ).to(device)

    if args.ckpt is None:
        raise ValueError("Provide --ckpt for eval-only mode.")
    ema = EMA.create(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    ckpt = load_checkpoint(args.ckpt, model, ema=ema)
    if ema is not None and ckpt.get("ema") is not None:
        ema.copy_to(model)

    test_acc, _, test_acc_by_snr = evaluate(
        model, test_loader, device, args.t_eval, args.snr_mode, amp=args.amp
    )
    print(f"Test acc: {test_acc:.4f}")
    print("Test acc by SNR:", test_acc_by_snr)


def main() -> None:
    args = parse_args()
    if args.lr is None:
        args.lr = PRESETS[args.preset]["lr"]
    if args.eval_only:
        run_eval(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
