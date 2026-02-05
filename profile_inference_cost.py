"""
Profile inference *cost* for DiT_AMC checkpoints.

What it reports:
  - parameter count (trainable params)
  - approximate FLOPs per decision for different K (using torch.profiler with_flops)
  - optional latency / throughput and peak CUDA memory (if CUDA is available)

Notes:
  - FLOPs are estimated from traced ops and are best-effort (good enough for reporting).
  - Timing excludes data loading and uses synthetic inputs of shape (B, K, 2, 128).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from model import CLDNNAMC, DiffusionAMC


def _infer_arch_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, int]:
    if "pos_embed" not in sd:
        raise ValueError("Checkpoint missing pos_embed; cannot infer architecture.")
    _, _num_patches, dim = sd["pos_embed"].shape

    if "patch_embed.weight" not in sd:
        raise ValueError("Checkpoint missing patch_embed.weight; cannot infer patch_size.")
    patch_size = int(sd["patch_embed.weight"].shape[-1])

    stem_channels = 0
    stem_layers = 0
    if any(k.startswith("stem.") for k in sd.keys()):
        w = sd.get("stem.0.weight", None)
        if w is not None and w.ndim == 3:
            stem_channels = int(w.shape[0])
        idxs = set()
        pat = re.compile(r"^stem\.(\d+)\.conv1\.weight$")
        for k in sd.keys():
            m = pat.match(k)
            if m:
                idxs.add(int(m.group(1)))
        stem_layers = len(idxs)

    block_idxs = set()
    pat_b = re.compile(r"^blocks\.(\d+)\.")
    for k in sd.keys():
        m = pat_b.match(k)
        if m:
            block_idxs.add(int(m.group(1)))
    depth = (max(block_idxs) + 1) if block_idxs else 0

    # Heads are not recoverable from the state dict (MHA weights are head-count agnostic).
    # Choose a sensible divisor for D for reporting.
    heads = 6 if dim % 6 == 0 else 4 if dim % 4 == 0 else 3 if dim % 3 == 0 else 2 if dim % 2 == 0 else 1

    return {
        "patch_size": int(patch_size),
        "dim": int(dim),
        "depth": int(depth),
        "heads": int(heads),
        "stem_channels": int(stem_channels),
        "stem_layers": int(stem_layers),
    }


def _infer_arch_cldnn(sd: Dict[str, torch.Tensor]) -> Dict[str, int]:
    if "conv_iq.weight" not in sd or "conv_merge.weight" not in sd or "lstm.weight_ih_l0" not in sd:
        raise ValueError("Checkpoint does not look like CLDNNAMC state dict.")

    conv_channels = int(sd["conv_iq.weight"].shape[0])
    merge_channels = int(sd["conv_merge.weight"].shape[0])
    # LSTM hidden: weight_ih has shape (4*hidden, input_size)
    lstm_hidden = int(sd["lstm.weight_ih_l0"].shape[0] // 4)

    lstm_layers = 0
    for k in sd.keys():
        if k.startswith("lstm.weight_ih_l") and not k.endswith("_reverse"):
            # e.g. lstm.weight_ih_l0, lstm.weight_ih_l1, ...
            idx = int(k.split("lstm.weight_ih_l", 1)[1])
            lstm_layers = max(lstm_layers, idx + 1)

    bidirectional = any(k.startswith("lstm.weight_ih_l0_reverse") for k in sd.keys())

    return {
        "seq_len": 128,
        "conv_channels": conv_channels,
        "merge_channels": merge_channels,
        "lstm_hidden": lstm_hidden,
        "lstm_layers": int(lstm_layers),
        "bidirectional": int(bidirectional),
    }


def _build_model_from_ckpt(ckpt_path: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if "pos_embed" in sd and "patch_embed.weight" in sd:
        arch = _infer_arch_from_state_dict(sd)
        model: torch.nn.Module = DiffusionAMC(
            num_classes=11,
            in_channels=2,
            seq_len=128,
            patch_size=arch["patch_size"],
            dim=arch["dim"],
            depth=arch["depth"],
            heads=arch["heads"],
            mlp_ratio=4.0,
            dropout=0.0,
            stem_channels=arch["stem_channels"],
            stem_layers=max(0, arch["stem_layers"]),
            group_pool="mean",
        )
        model.load_state_dict(sd, strict=False)
        model.eval()
        model._arch = arch  # type: ignore[attr-defined]
        return model

    # CLDNN checkpoint
    arch = _infer_arch_cldnn(sd)
    model = CLDNNAMC(
        num_classes=11,
        seq_len=128,
        conv_channels=int(arch["conv_channels"]),
        merge_channels=int(arch["merge_channels"]),
        lstm_hidden=int(arch["lstm_hidden"]),
        lstm_layers=int(arch["lstm_layers"]),
        bidirectional=bool(arch["bidirectional"]),
        dropout=0.0,
        pool="attn",
    )
    model.load_state_dict(sd, strict=False)
    model.eval()
    model._arch = arch  # type: ignore[attr-defined]
    return model


def _count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


@torch.no_grad()
def estimate_flops_per_decision(
    model: torch.nn.Module,
    k_list: List[int],
    snr_mode: str,
    t_eval: int,
) -> Dict[int, float]:
    """
    Returns FLOPs per *decision* for each K (batch=1).
    """
    out: Dict[int, float] = {}
    device = torch.device("cpu")
    model_cpu = model.to(device)

    for k in k_list:
        bsz = 1
        x = torch.randn(bsz, k, 2, 128, dtype=torch.float32, device=device)
        t = torch.full((bsz,), int(t_eval), dtype=torch.long, device=device)
        snr = torch.zeros((bsz,), dtype=torch.float32, device=device) if snr_mode == "known" else None

        with profile(
            activities=[ProfilerActivity.CPU],
            with_flops=True,
            record_shapes=False,
        ) as prof:
            _logits, _x0, _snr_pred = model_cpu(x, t, snr=snr, snr_mode=snr_mode)

        total_flops = 0
        for e in prof.key_averages():
            f = getattr(e, "flops", 0)
            if f:
                total_flops += int(f)

        # total_flops already corresponds to a single decision (bsz=1).
        out[int(k)] = float(total_flops)

    return out


@torch.no_grad()
def bench_latency(
    model: torch.nn.Module,
    k_list: List[int],
    snr_mode: str,
    t_eval: int,
    batch_size: int,
    iters: int,
    warmup: int,
    amp: bool,
) -> Dict[int, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    use_cuda = device.type == "cuda"

    results: Dict[int, Dict[str, float]] = {}

    for k in k_list:
        x = torch.randn(batch_size, k, 2, 128, dtype=torch.float32, device=device)
        t = torch.full((batch_size,), int(t_eval), dtype=torch.long, device=device)
        snr = torch.zeros((batch_size,), dtype=torch.float32, device=device) if snr_mode == "known" else None

        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Warmup
        for _ in range(int(warmup)):
            if amp and use_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    _ = model(x, t, snr=snr, snr_mode=snr_mode)
            else:
                _ = model(x, t, snr=snr, snr_mode=snr_mode)

        if use_cuda:
            torch.cuda.synchronize()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            for _ in range(int(iters)):
                if amp and use_cuda:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        _ = model(x, t, snr=snr, snr_mode=snr_mode)
                else:
                    _ = model(x, t, snr=snr, snr_mode=snr_mode)
            end_evt.record()
            torch.cuda.synchronize()
            total_ms = float(start_evt.elapsed_time(end_evt))
            avg_ms = total_ms / max(1, int(iters))
            peak_mem = float(torch.cuda.max_memory_allocated() / (1024**2))
        else:
            t0 = time.perf_counter()
            for _ in range(int(iters)):
                _ = model(x, t, snr=snr, snr_mode=snr_mode)
            t1 = time.perf_counter()
            total_ms = (t1 - t0) * 1000.0
            avg_ms = total_ms / max(1, int(iters))
            peak_mem = float("nan")

        decisions_per_s = float(batch_size) / max(1e-9, avg_ms / 1000.0)
        windows_per_s = decisions_per_s * float(k)
        avg_ms_per_decision = float(avg_ms) / max(1.0, float(batch_size))
        avg_ms_per_window = float(avg_ms) / max(1.0, float(batch_size) * float(k))

        results[int(k)] = {
            "avg_ms_per_batch": float(avg_ms),
            "avg_ms_per_decision": float(avg_ms_per_decision),
            "avg_ms_per_window": float(avg_ms_per_window),
            "decisions_per_s": float(decisions_per_s),
            "windows_per_s": float(windows_per_s),
            "peak_cuda_mem_mb": float(peak_mem),
        }

    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile inference cost for DiT_AMC checkpoints.")
    p.add_argument("--ckpt", type=str, required=True, help="Path to best.pt checkpoint.")
    p.add_argument("--out", type=str, default=None, help="Optional JSON output path to write results.")
    p.add_argument("--k-list", type=str, default="1,2,4,8,16", help="Comma-separated K values to profile.")
    p.add_argument("--snr-mode", type=str, choices=["predict", "known", "none"], default="predict")
    p.add_argument("--t-eval", type=int, default=0)

    # Timing
    p.add_argument("--bench", action="store_true", help="Run timing benchmark (uses CUDA if available).")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--amp", action="store_true", help="Use bf16 autocast on CUDA for benchmarking.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    k_list = [int(v.strip()) for v in args.k_list.split(",") if v.strip() != ""]
    if not k_list:
        raise ValueError("--k-list produced empty list")

    model = _build_model_from_ckpt(args.ckpt)
    arch = getattr(model, "_arch", {})  # type: ignore[attr-defined]
    params = _count_params(model)

    flops = estimate_flops_per_decision(model, k_list, snr_mode=args.snr_mode, t_eval=args.t_eval)

    timing: Optional[Dict[int, Dict[str, float]]] = None
    if args.bench:
        timing = bench_latency(
            model,
            k_list,
            snr_mode=args.snr_mode,
            t_eval=args.t_eval,
            batch_size=int(args.batch_size),
            iters=int(args.iters),
            warmup=int(args.warmup),
            amp=bool(args.amp),
        )

    report = {
        "ckpt": os.path.abspath(args.ckpt),
        "arch_inferred": arch,
        "params": int(params),
        "params_m": float(params) / 1e6,
        "snr_mode": args.snr_mode,
        "t_eval": int(args.t_eval),
        "k_list": k_list,
        "flops_per_decision": {str(k): float(v) for k, v in flops.items()},
        "timing": timing,
    }

    print(json.dumps(report, indent=2))

    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir != "":
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()

