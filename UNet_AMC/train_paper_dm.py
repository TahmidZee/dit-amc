"""
Train the paper-style one-step diffusion denoiser (SDE formulation).

We train on *clean-ish* high-SNR samples x0. For each batch:
  - sample continuous t ~ U(t_min, t_max) in (0,1]
  - sample eps ~ N(0, I)
  - construct x_t = (1 - t) * x0 + sqrt(t) * eps
  - targets:
      h_target   = -x0
      eps_target = sqrt(t) * eps
  - model predicts (h_hat, eps_hat) from (x_t, t)
  - loss = MSE(h_hat, h_target) + MSE(eps_hat, eps_target)

This matches the paper's one-step denoise form:
  x0_hat = x_t - t * h_hat - eps_hat

Then you can apply the denoiser to real received signals r using:
  - Eq.(21): t from SNR (via σ^2)
  - Eq.(22): scale α and feed x_t = α r into the denoiser
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from paper_dm import PaperOneStepDenoiser, count_parameters
from data_ddpm import load_rml2016a, build_split_indices, RML2016aCleanDataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="./runs/paper_dm_denoiser")
    p.add_argument("--normalize", type=str, default="rms", choices=["rms", "max", "none"])
    p.add_argument("--high-snr-threshold", type=int, default=14)

    # model
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--time-dim", type=int, default=256)
    p.add_argument("--signal-power", type=float, default=1.0, help="E||x0||^2 assumed in Eq.(22). If using RMS norm, 1.0 is a good default.")

    # diffusion training
    p.add_argument("--t-min", type=float, default=1e-3)
    p.add_argument("--t-max", type=float, default=0.999)

    # optimization
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)

    # hardware
    p.add_argument("--amp", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2016)
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("device:", device)
    print("out_dir:", out_dir)

    # data
    data = load_rml2016a(args.data_path)
    train_idx, val_idx, _ = build_split_indices(data, seed=args.seed)

    train_ds = RML2016aCleanDataset(
        data,
        train_idx,
        normalize=args.normalize,
        high_snr_threshold=args.high_snr_threshold,
        augment=True,
    )
    val_ds = RML2016aCleanDataset(
        data,
        val_idx,
        normalize=args.normalize,
        high_snr_threshold=args.high_snr_threshold,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # model
    model = PaperOneStepDenoiser(
        in_channels=2,
        base_channels=args.base_channels,
        depth=args.depth,
        time_dim=args.time_dim,
        signal_power=args.signal_power,
    ).to(device)
    print("params:", count_parameters(model))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler("cuda", enabled=use_amp)

    best_val_recon = float("inf")
    patience = 0
    log_path = out_dir / "metrics.jsonl"

    def sample_t(batch: int, device_: torch.device) -> torch.Tensor:
        t = torch.rand(batch, device=device_)
        t = args.t_min + (args.t_max - args.t_min) * t
        return torch.clamp(t, min=1e-6, max=1.0)

    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()

        train_loss = 0.0
        train_recon = 0.0
        for x0, _, _ in train_loader:
            x0 = x0.to(device)  # (B,2,128)
            b = x0.size(0)
            t = sample_t(b, device)

            eps = torch.randn_like(x0)
            eps_scaled = torch.sqrt(t)[:, None, None] * eps
            x_t = (1.0 - t)[:, None, None] * x0 + eps_scaled

            h_target = -x0
            eps_target = eps_scaled

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                h_hat, eps_hat = model(x_t, t)
                loss_h = F.mse_loss(h_hat, h_target)
                loss_eps = F.mse_loss(eps_hat, eps_target)
                loss = loss_h + loss_eps

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                x0_hat = x_t - t[:, None, None] * h_hat - eps_hat
                recon = F.mse_loss(x0_hat, x0).item()

            train_loss += loss.item()
            train_recon += recon

        train_loss /= len(train_loader)
        train_recon /= len(train_loader)

        # val
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        with torch.no_grad():
            for x0, _, _ in val_loader:
                x0 = x0.to(device)
                b = x0.size(0)
                t = sample_t(b, device)
                eps = torch.randn_like(x0)
                eps_scaled = torch.sqrt(t)[:, None, None] * eps
                x_t = (1.0 - t)[:, None, None] * x0 + eps_scaled
                h_target = -x0
                eps_target = eps_scaled

                h_hat, eps_hat = model(x_t, t)
                loss = F.mse_loss(h_hat, h_target) + F.mse_loss(eps_hat, eps_target)

                x0_hat = x_t - t[:, None, None] * h_hat - eps_hat
                recon = F.mse_loss(x0_hat, x0)

                val_loss += loss.item()
                val_recon += recon.item()

        val_loss /= len(val_loader)
        val_recon /= len(val_loader)

        sched.step()
        dt = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_recon_mse": train_recon,
            "val_loss": val_loss,
            "val_recon_mse": val_recon,
            "lr": opt.param_groups[0]["lr"],
            "time_sec": dt,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_loss:.4f} recon {train_recon:.6f} | "
            f"val loss {val_loss:.4f} recon {val_recon:.6f} | "
            f"{dt:.1f}s"
        )

        # checkpoint
        if val_recon < best_val_recon:
            best_val_recon = val_recon
            patience = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "args": vars(args),
                    "best_val_recon": best_val_recon,
                },
                out_dir / "best.pt",
            )
            print(f"  -> new best val recon mse: {best_val_recon:.6f}")
        else:
            patience += 1

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
            },
            out_dir / "last.pt",
        )

        if patience >= args.patience:
            print("Early stopping.")
            break

    # save config
    cfg = vars(args)
    cfg["num_params"] = count_parameters(model)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print("done:", out_dir)


if __name__ == "__main__":
    main()

