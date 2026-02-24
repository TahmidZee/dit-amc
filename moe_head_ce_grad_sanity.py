#!/usr/bin/env python3
"""
One-batch gradient sanity checks for MoE head-specific CE gradient isolation.

Case A:
  aux CE only + --moe-head-ce-detach-trunk
  => trunk grad ~ 0, head grad > 0

Case B:
  aux CE only + detach disabled
  => trunk grad > 0
"""

import argparse
import json
from typing import Dict

import torch

from model import CLDNNAMC
from train import compute_moe_head_specialization_losses


def _grad_norm(param: torch.nn.Parameter) -> float:
    if param.grad is None:
        return 0.0
    return float(param.grad.detach().norm().item())


def run_case(
    *,
    detach_trunk: bool,
    device: torch.device,
    seed: int,
    batch_size: int,
    seq_len: int,
    num_classes: int,
    low_lambda: float,
    high_lambda: float,
) -> Dict[str, float]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = CLDNNAMC(
        num_classes=num_classes,
        seq_len=seq_len,
        dropout=0.0,
        moe_n_experts=2,
        moe_gate_type="eta-sigmoid",
        moe_gate_center=1.55,
        moe_gate_tau=0.3,
        moe_gate_use_feat=False,
    ).to(device)
    model.train()

    x = torch.randn(batch_size, 2, seq_len, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    # Keep all samples active in both low/high masks.
    snr = torch.zeros(batch_size, device=device)
    curriculum_mask = torch.ones(batch_size, device=device)
    t = torch.zeros(batch_size, dtype=torch.long, device=device)

    model.zero_grad(set_to_none=True)
    _logits, _x0, _snr = model(
        x,
        t,
        snr=None,
        snr_mode="predict",
        moe_head_ce_detach_trunk=detach_trunk,
    )
    if detach_trunk:
        logits_experts = getattr(model, "_moe_logits_experts_aux", None)
    else:
        logits_experts = getattr(model, "_moe_logits_experts", None)
    if logits_experts is None:
        raise RuntimeError("Expected MoE expert logits were not produced.")

    loss_low, loss_high, low_frac, high_frac = compute_moe_head_specialization_losses(
        logits_experts=logits_experts,
        y=y,
        snr_db=snr,
        curriculum_mask=curriculum_mask,
        use_mixup=False,
        y_a=None,
        y_b=None,
        lam=1.0,
        low_head_idx=0,
        high_head_idx=1,
        low_lambda=low_lambda,
        high_lambda=high_lambda,
        low_snr_lo=-14.0,
        low_snr_hi=2.0,
        high_snr_lo=-6.0,
        high_snr_hi=18.0,
        label_smoothing=0.0,
    )
    loss = loss_low + loss_high
    loss.backward()

    if model.moe_head is None:
        raise RuntimeError("MoE head was not initialized.")

    out = {
        "detach_trunk": float(1 if detach_trunk else 0),
        "loss": float(loss.detach().item()),
        "loss_low": float(loss_low.detach().item()),
        "loss_high": float(loss_high.detach().item()),
        "low_active_frac": float(low_frac.detach().item()),
        "high_active_frac": float(high_frac.detach().item()),
        "trunk_grad_norm_conv_iq": _grad_norm(model.conv_iq.weight),
        "trunk_grad_norm_conv_merge": _grad_norm(model.conv_merge.weight),
        "head_grad_norm_fc1_e0": _grad_norm(model.moe_head.fc1[0].weight),
        "head_grad_norm_fc_out_e0": _grad_norm(model.moe_head.fc_out[0].weight),
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="MoE head-specific CE gradient sanity check.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--low-lambda", type=float, default=0.2)
    parser.add_argument("--high-lambda", type=float, default=0.2)
    parser.add_argument("--trunk-zero-tol", type=float, default=1e-12)
    parser.add_argument("--grad-min", type=float, default=1e-8)
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    case_a = run_case(
        detach_trunk=True,
        device=device,
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        low_lambda=args.low_lambda,
        high_lambda=args.high_lambda,
    )
    case_b = run_case(
        detach_trunk=False,
        device=device,
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_classes=args.num_classes,
        low_lambda=args.low_lambda,
        high_lambda=args.high_lambda,
    )

    trunk_a = max(case_a["trunk_grad_norm_conv_iq"], case_a["trunk_grad_norm_conv_merge"])
    trunk_b = max(case_b["trunk_grad_norm_conv_iq"], case_b["trunk_grad_norm_conv_merge"])
    head_a = max(case_a["head_grad_norm_fc1_e0"], case_a["head_grad_norm_fc_out_e0"])
    head_b = max(case_b["head_grad_norm_fc1_e0"], case_b["head_grad_norm_fc_out_e0"])

    ok_a = trunk_a <= float(args.trunk_zero_tol) and head_a > float(args.grad_min)
    ok_b = trunk_b > float(args.grad_min) and head_b > float(args.grad_min)
    ok = bool(ok_a and ok_b)

    report = {
        "device": str(device),
        "case_a_detach_on": case_a,
        "case_b_detach_off": case_b,
        "checks": {
            "case_a_trunk_near_zero_and_head_nonzero": bool(ok_a),
            "case_b_trunk_nonzero_and_head_nonzero": bool(ok_b),
            "overall_pass": bool(ok),
        },
        "thresholds": {
            "trunk_zero_tol": float(args.trunk_zero_tol),
            "grad_min": float(args.grad_min),
        },
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"device={report['device']}")
        print(f"case_a(detach=on): trunk_max={trunk_a:.3e} head_max={head_a:.3e}")
        print(f"case_b(detach=off): trunk_max={trunk_b:.3e} head_max={head_b:.3e}")
        print(
            "checks:",
            {
                "case_a_trunk_near_zero_and_head_nonzero": ok_a,
                "case_b_trunk_nonzero_and_head_nonzero": ok_b,
                "overall_pass": ok,
            },
        )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
