"""
Evaluate the paper-style one-step denoiser as an inference-time preprocessor for AMC.

This script reports accuracy vs SNR for:
  1) Baseline classifier on raw signals
  2) Same classifier on denoised signals (PaperOneStepDenoiser)

NOTE:
  - This uses *known* SNR labels from the dataset to compute t and α (Eq.(21)/(22)).
  - For a deployable system, you would replace SNR labels with an SNR estimator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from paper_dm import PaperOneStepDenoiser
from data_ddpm import load_rml2016a, build_split_indices, RML2016aFullDataset
from model import UNetAMC


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--classifier-ckpt", type=str, required=True, help="Path to a UNetAMC checkpoint (best.pt from train.py).")
    p.add_argument("--denoiser-ckpt", type=str, required=True, help="Path to paper denoiser checkpoint (best.pt from train_paper_dm.py).")
    p.add_argument("--out-json", type=str, default="./runs/paper_dm_eval.json")
    p.add_argument("--normalize", type=str, default="rms", choices=["rms", "max", "none"])
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2016)

    # denoiser params must match training assumptions
    p.add_argument("--signal-power", type=float, default=1.0)
    return p.parse_args()


@torch.no_grad()
def acc_by_snr(logits: torch.Tensor, y: torch.Tensor, snr: torch.Tensor, out: dict):
    pred = logits.argmax(dim=1)
    for i in range(y.size(0)):
        s = int(snr[i].item())
        out.setdefault(s, [0, 0])
        out[s][0] += int((pred[i] == y[i]).item())
        out[s][1] += 1


def finalize_acc(out: dict) -> dict:
    return {str(k): out[k][0] / max(1, out[k][1]) for k in sorted(out.keys())}


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # data split
    data = load_rml2016a(args.data_path)
    _, _, test_idx = build_split_indices(data, seed=args.seed)
    test_ds = RML2016aFullDataset(data, test_idx, normalize=args.normalize)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    num_classes = test_ds.num_classes

    # classifier
    clf_ckpt = torch.load(args.classifier_ckpt, weights_only=False, map_location=device)
    clf_cfg = clf_ckpt.get("config", {}) or {}
    clf = UNetAMC(
        in_channels=2,
        base_channels=int(clf_cfg.get("base_channels", 64)),
        depth=int(clf_cfg.get("depth", 4)),
        num_classes=num_classes,
        lstm_hidden=int(clf_cfg.get("lstm_hidden", 128)),
        dropout=float(clf_cfg.get("dropout", 0.5)),
    ).to(device)
    clf.load_state_dict(clf_ckpt["model_state_dict"])
    clf.eval()

    # denoiser
    den_ckpt = torch.load(args.denoiser_ckpt, weights_only=False, map_location=device)
    den_args = den_ckpt.get("args", {}) or {}
    den = PaperOneStepDenoiser(
        in_channels=2,
        base_channels=int(den_args.get("base_channels", 64)),
        depth=int(den_args.get("depth", 4)),
        time_dim=int(den_args.get("time_dim", 256)),
        signal_power=float(den_args.get("signal_power", args.signal_power)),
    ).to(device)
    den.load_state_dict(den_ckpt["model"])
    den.eval()

    # evaluate
    base_out = {}
    den_out = {}
    base_correct = 0
    den_correct = 0
    total = 0

    for x, y, snr in test_loader:
        x = x.to(device)
        y = y.to(device)

        logits = clf(x, return_denoised=False)
        base_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
        acc_by_snr(logits, y, snr, base_out)

        x_hat = den.denoise_observation(x, snr_db=snr.to(device))
        logits_hat = clf(x_hat, return_denoised=False)
        den_correct += (logits_hat.argmax(dim=1) == y).sum().item()
        acc_by_snr(logits_hat, y, snr, den_out)

    base_overall = base_correct / max(1, total)
    den_overall = den_correct / max(1, total)

    results = {
        "baseline_overall": base_overall,
        "denoised_overall": den_overall,
        "baseline_acc_by_snr": finalize_acc(base_out),
        "denoised_acc_by_snr": finalize_acc(den_out),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("baseline_overall:", base_overall)
    print("denoised_overall:", den_overall)
    print("wrote:", out_path)


if __name__ == "__main__":
    main()

