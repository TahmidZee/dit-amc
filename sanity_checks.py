import argparse
import json
from types import SimpleNamespace

import numpy as np
import torch

import train as dit_train


def _intersection_size(a, b) -> int:
    sa = set(int(x) for x in a)
    sb = set(int(x) for x in b)
    return len(sa.intersection(sb))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity checks for DiT_AMC evaluation (splits, per-SNR mean, etc.).")
    p.add_argument("--data-path", type=str, default="/home/tahit/Modulation/RML2016.10a_dict.pkl")
    p.add_argument("--seed", type=int, default=2016)
    p.add_argument("--train-per", type=int, default=600)
    p.add_argument("--val-per", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--normalize", type=str, default="rms", choices=["rms", "none"])

    # Model/data-shape args needed to reproduce grouped/variable-K evaluation.
    p.add_argument("--preset", type=str, default="B", choices=["S", "B"])
    p.add_argument("--patch-size", type=int, default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--stem-channels", type=int, default=64)
    p.add_argument("--stem-layers", type=int, default=2)
    p.add_argument("--group-k", type=int, default=1)
    p.add_argument("--group-pool", type=str, default="mean", choices=["mean", "attn"])
    p.add_argument("--k-max", type=int, default=None)
    p.add_argument("--k-choices", type=str, default=None)
    p.add_argument("--window-dropout", type=float, default=0.0)

    p.add_argument("--snr-mode", type=str, default="predict", choices=["predict", "known", "none"])
    p.add_argument("--snr-scale", type=float, default=20.0)
    p.add_argument("--t-eval", type=int, default=0)
    p.add_argument("--amp", action="store_true")

    p.add_argument("--ckpt", type=str, default=None, help="If provided, run test evaluation for this checkpoint.")
    p.add_argument("--write-json", type=str, default=None, help="Optional path to write a small JSON report.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dit_train.apply_preset(args)

    # --- Split sanity check (no overlap) ---
    X, y, snr, mods, snrs, train_idx, val_idx, test_idx = dit_train.load_rml2016a(
        args.data_path,
        seed=args.seed,
        train_per=args.train_per,
        val_per=args.val_per,
    )
    inter_tv = _intersection_size(train_idx, val_idx)
    inter_tt = _intersection_size(train_idx, test_idx)
    inter_vt = _intersection_size(val_idx, test_idx)
    print("Split sizes:", {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)})
    print("Split overlaps:", {"train∩val": inter_tv, "train∩test": inter_tt, "val∩test": inter_vt})

    report = {
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "split_overlaps": {"train_val": inter_tv, "train_test": inter_tt, "val_test": inter_vt},
    }

    # --- Eval sanity check (optional) ---
    if args.ckpt:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Reuse train.py's loader building to ensure exact same shapes/masks.
        train_loader, val_loader, test_loader, _mods, _snrs = dit_train.build_loaders(args, device)
        model = dit_train.DiffusionAMC(
            num_classes=len(_mods),
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
        ema = dit_train.EMA.create(model, decay=0.9996)
        ckpt = dit_train.load_checkpoint(args.ckpt, model, ema=ema)
        if ckpt.get("ema") is not None:
            ema.copy_to(model)

        test_acc, _n, test_by_snr = dit_train.evaluate(model, test_loader, device, args.t_eval, args.snr_mode, amp=args.amp)
        # Dataset is balanced per SNR in the standard split, so mean(per_snr) == overall.
        mean_by_snr = float(np.mean(list(test_by_snr.values()))) if len(test_by_snr) else 0.0
        print(f"Test acc: {test_acc:.6f}")
        print(f"Mean acc_by_snr: {mean_by_snr:.6f}")
        report["test_acc"] = float(test_acc)
        report["mean_acc_by_snr"] = float(mean_by_snr)
        report["test_acc_by_snr"] = {str(int(k)): float(v) for k, v in test_by_snr.items()}

    if args.write_json:
        with open(args.write_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

