"""
Evaluate a trained DiT_AMC checkpoint in a strict K=1 setting (single 128-sample window),
exporting accuracy-vs-SNR JSON for apples-to-apples comparisons with single-window baselines
like MCLDNN.

This script intentionally evaluates on the exact test indices (no grouped bucket resampling),
and uses the same split protocol as training (seed=2016, train_per=600, val_per=200 by default).
"""

import argparse
import json
import os
import re
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import RML2016aDataset, build_tensors, load_rml2016a
from model import DiffusionAMC


def _infer_arch_from_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, int]:
    # dim + patches from pos_embed
    if "pos_embed" not in sd:
        raise ValueError("Checkpoint missing pos_embed; cannot infer architecture.")
    _, num_patches, dim = sd["pos_embed"].shape

    # patch_size from patch_embed conv kernel
    if "patch_embed.weight" not in sd:
        raise ValueError("Checkpoint missing patch_embed.weight; cannot infer patch_size.")
    patch_size = int(sd["patch_embed.weight"].shape[-1])

    # stem_channels/stem_layers: optional
    stem_channels = 0
    stem_layers = 0
    if any(k.startswith("stem.") for k in sd.keys()):
        w = sd.get("stem.0.weight", None)
        if w is not None and w.ndim == 3:
            stem_channels = int(w.shape[0])
        # Count ResBlock modules inside nn.Sequential by detecting "<idx>.conv1.weight"
        # Example: stem.3.conv1.weight, stem.4.conv1.weight, ...
        idxs = set()
        pat = re.compile(r"^stem\.(\d+)\.conv1\.weight$")
        for k in sd.keys():
            m = pat.match(k)
            if m:
                idxs.add(int(m.group(1)))
        stem_layers = len(idxs)

    # depth: count blocks.<i>.*
    block_idxs = set()
    pat_b = re.compile(r"^blocks\.(\d+)\.")
    for k in sd.keys():
        m = pat_b.match(k)
        if m:
            block_idxs.add(int(m.group(1)))
    depth = (max(block_idxs) + 1) if block_idxs else 0

    # Heads aren't identifiable from the state_dict (MHA weights are head-count agnostic).
    # We'll default to the repo default (6) if possible.
    heads = 6 if dim % 6 == 0 else 4 if dim % 4 == 0 else 3 if dim % 3 == 0 else 2 if dim % 2 == 0 else 1

    return {
        "num_patches": int(num_patches),
        "patch_size": int(patch_size),
        "dim": int(dim),
        "depth": int(depth),
        "heads": int(heads),
        "stem_channels": int(stem_channels),
        "stem_layers": int(stem_layers),
    }


@torch.no_grad()
def evaluate_k1(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    t_eval: int,
    snr_mode: str,
) -> Tuple[float, Dict[int, float]]:
    model.eval()
    total = 0
    correct = 0
    snr_correct: Dict[int, int] = {}
    snr_total: Dict[int, int] = {}

    for x, y, snr in loader:
        x = x.to(device)
        y = y.to(device)
        snr = snr.to(device)
        t = torch.full((x.shape[0],), int(t_eval), device=device, dtype=torch.long)

        snr_in = snr if snr_mode == "known" else None
        logits, _, _ = model(x, t, snr=snr_in, snr_mode=snr_mode)
        pred = logits.argmax(dim=1)

        total += int(y.shape[0])
        correct += int((pred == y).sum().item())

        snr_cpu = snr.detach().cpu().numpy().astype(np.int32)
        pred_cpu = pred.detach().cpu().numpy().astype(np.int64)
        y_cpu = y.detach().cpu().numpy().astype(np.int64)
        for s, p, yt in zip(snr_cpu.tolist(), pred_cpu.tolist(), y_cpu.tolist()):
            s = int(s)
            snr_total[s] = snr_total.get(s, 0) + 1
            snr_correct[s] = snr_correct.get(s, 0) + int(p == yt)

    acc = float(correct) / max(1, total)
    by_snr = {int(s): float(snr_correct[s]) / max(1, int(snr_total[s])) for s in snr_total.keys()}
    return acc, by_snr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DiT_AMC checkpoint with K=1 and export acc-vs-SNR JSON.")
    p.add_argument("--data-path", type=str, default="/home/tahit/Modulation/RML2016.10a_dict.pkl")
    p.add_argument("--ckpt", type=str, required=True, help="Path to best.pt checkpoint")
    p.add_argument("--out", type=str, required=True, help="Output JSON path (e.g., test_acc_by_snr_k1.json)")

    p.add_argument("--seed", type=int, default=2016)
    p.add_argument("--train-per", type=int, default=600)
    p.add_argument("--val-per", type=int, default=200)
    p.add_argument("--normalize", type=str, choices=["rms", "none"], default="rms")

    p.add_argument("--snr-mode", type=str, choices=["predict", "known", "none"], default="predict")
    p.add_argument("--t-eval", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    # NOTE: Some newer PyTorch versions default to weights_only=True, which can fail
    # for checkpoints containing non-tensor metadata. This project writes plain
    # dict checkpoints; load them fully.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    arch = _infer_arch_from_state_dict(sd)

    # Instantiate model in a way that matches the checkpoint weights.
    model = DiffusionAMC(
        num_classes=11,
        in_channels=2,
        seq_len=128,
        patch_size=arch["patch_size"],
        dim=arch["dim"],
        depth=arch["depth"],
        heads=arch["heads"],
        mlp_ratio=4.0,
        dropout=0.0,  # eval only
        stem_channels=arch["stem_channels"],
        stem_layers=max(0, arch["stem_layers"]),
        group_pool="mean",
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected[:10]}")
    if missing:
        # Be conservative: some missing keys could indicate inference mismatch.
        # However, buffers like pos_embed are required; most others should match.
        print(f"[warn] missing keys (first 10): {missing[:10]}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[arch] dim={arch['dim']} depth={arch['depth']} heads={arch['heads']} patch={arch['patch_size']} stem_channels={arch['stem_channels']} stem_layers={arch['stem_layers']}")
    print(f"[arch] params={n_params/1e6:.3f}M")

    # Build strict K=1 dataset (no grouped sampling).
    X, y, snr, _mods, _snrs, _train_idx, _val_idx, test_idx = load_rml2016a(
        args.data_path, seed=args.seed, train_per=args.train_per, val_per=args.val_per
    )
    X_t, y_t, snr_t = build_tensors(X, y, snr)
    test_ds = RML2016aDataset(
        X_t,
        y_t,
        snr_t,
        test_idx,
        normalize=args.normalize,
        aug_phase=False,
        aug_shift=False,
        aug_gain=0.0,
        aug_cfo=0.0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    acc, by_snr = evaluate_k1(model, test_loader, device, t_eval=args.t_eval, snr_mode=args.snr_mode)
    print(f"[k=1] test_acc={acc:.6f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({str(int(k)): float(v) for k, v in sorted(by_snr.items())}, f, indent=2)
    print(f"[k=1] wrote {args.out}")


if __name__ == "__main__":
    main()

