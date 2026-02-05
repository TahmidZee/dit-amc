"""
Training script for UNet-AMC: Joint Denoising + Classification

Usage:
    python train.py --data-path /path/to/RML2016.10a_dict.pkl --out-dir ./runs/unet_amc
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from model import UNetAMC, UNetAMCWithMultiWindow, count_parameters
from data import (
    load_rml2016a,
    build_split_indices,
    RML2016aDataset,
    RML2016aDenoisingDatasetV2,
    RML2016aGroupedDenoisingDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet-AMC")
    
    # Data
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to RML2016.10a_dict.pkl")
    parser.add_argument("--out-dir", type=str, default="./runs/unet_amc",
                        help="Output directory")
    parser.add_argument("--normalize", type=str, default="rms",
                        choices=["rms", "max", "none"])
    
    # Model
    parser.add_argument("--base-channels", type=int, default=64,
                        help="Base channels for U-Net")
    parser.add_argument("--depth", type=int, default=4,
                        help="U-Net depth (number of downsampling levels)")
    parser.add_argument("--lstm-hidden", type=int, default=128,
                        help="LSTM hidden size")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout rate")
    
    # Multi-window
    parser.add_argument("--group-k", type=int, default=1,
                        help="Number of windows to group (1 = single window)")
    parser.add_argument("--pool-method", type=str, default="mean",
                        choices=["mean", "attn"])
    
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-denoise", type=float, default=0.1,
                        help="Weight for denoising loss")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience")
    
    # Denoising dataset
    parser.add_argument("--clean-snr-threshold", type=int, default=14,
                        help="SNR threshold for 'clean' samples")
    parser.add_argument("--target-snr-min", type=float, default=-20,
                        help="Min target SNR for synthetic noise")
    parser.add_argument("--target-snr-max", type=float, default=10,
                        help="Max target SNR for synthetic noise")
    
    # Training mode
    parser.add_argument("--train-mode", type=str, default="joint",
                        choices=["joint", "cls_only", "denoise_only"],
                        help="Training mode")
    
    # Hardware
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2016)
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(args, data, train_idx, val_idx, test_idx):
    """Build data loaders"""
    
    target_snr_range = (args.target_snr_min, args.target_snr_max)
    
    if args.group_k > 1:
        # Grouped (multi-window) dataset
        train_ds = RML2016aGroupedDenoisingDataset(
            data, train_idx,
            group_k=args.group_k,
            normalize=args.normalize,
            clean_snr_threshold=args.clean_snr_threshold,
            target_snr_range=target_snr_range,
            augment=True,
            seed=args.seed
        )
        val_ds = RML2016aGroupedDenoisingDataset(
            data, val_idx,
            group_k=args.group_k,
            normalize=args.normalize,
            clean_snr_threshold=args.clean_snr_threshold,
            target_snr_range=target_snr_range,
            augment=False,
            seed=args.seed + 1
        )
    else:
        # Single window dataset
        train_ds = RML2016aDenoisingDatasetV2(
            data, train_idx,
            normalize=args.normalize,
            clean_snr_threshold=args.clean_snr_threshold,
            target_snr_range=target_snr_range,
            augment=True
        )
        val_ds = RML2016aDenoisingDatasetV2(
            data, val_idx,
            normalize=args.normalize,
            clean_snr_threshold=args.clean_snr_threshold,
            target_snr_range=target_snr_range,
            augment=False
        )
    
    # Test set uses ALL samples (not just high-SNR)
    test_ds = RML2016aDataset(data, test_idx, normalize=args.normalize)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_ds.num_classes


def train_epoch(model, loader, optimizer, scaler, args, device, epoch):
    """Train one epoch"""
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_denoise_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(loader):
        if args.group_k > 1:
            noisy, clean, labels, orig_snr, target_snr = batch
            noisy = noisy.to(device)  # (B, K, 2, 128)
            clean = clean.to(device)
        else:
            noisy, clean, labels, orig_snr, target_snr = batch
            noisy = noisy.to(device)  # (B, 2, 128)
            clean = clean.to(device)
        
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=args.amp):
            if args.group_k > 1:
                # Multi-window model
                logits, denoised = model(noisy, return_denoised=True)
            else:
                logits, denoised = model(noisy, return_denoised=True)
            
            # Classification loss
            cls_loss = F.cross_entropy(logits, labels)
            
            # Denoising loss (MSE)
            denoise_loss = F.mse_loss(denoised, clean)
            
            # Combined loss
            if args.train_mode == "joint":
                loss = cls_loss + args.lambda_denoise * denoise_loss
            elif args.train_mode == "cls_only":
                loss = cls_loss
            elif args.train_mode == "denoise_only":
                loss = denoise_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_denoise_loss += denoise_loss.item()
        
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    n_batches = len(loader)
    return {
        "loss": total_loss / n_batches,
        "cls_loss": total_cls_loss / n_batches,
        "denoise_loss": total_denoise_loss / n_batches,
        "acc": correct / total
    }


@torch.no_grad()
def evaluate(model, loader, args, device, is_grouped=False):
    """Evaluate on validation/test set"""
    model.eval()
    
    correct = 0
    total = 0
    total_loss = 0
    
    for batch in loader:
        if is_grouped:
            noisy, clean, labels, orig_snr, target_snr = batch
            noisy = noisy.to(device)
            clean = clean.to(device)
            labels = labels.to(device)
            
            logits, denoised = model(noisy, return_denoised=True)
            loss = F.cross_entropy(logits, labels)
        else:
            x, labels, snr = batch
            x = x.to(device)
            labels = labels.to(device)
            
            # For test set, we classify directly (no denoising target available)
            logits = model(x, return_denoised=False)
            loss = F.cross_entropy(logits, labels)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return {
        "loss": total_loss / len(loader),
        "acc": correct / total
    }


@torch.no_grad()
def evaluate_by_snr(model, loader, device):
    """Evaluate accuracy per SNR level"""
    model.eval()
    
    snr_correct = {}
    snr_total = {}
    
    for batch in loader:
        x, labels, snrs = batch
        x = x.to(device)
        labels = labels.to(device)
        
        logits = model(x, return_denoised=False)
        preds = logits.argmax(dim=1)
        
        for i, snr in enumerate(snrs.numpy()):
            snr = int(snr)
            if snr not in snr_correct:
                snr_correct[snr] = 0
                snr_total[snr] = 0
            snr_correct[snr] += (preds[i] == labels[i]).item()
            snr_total[snr] += 1
    
    acc_by_snr = {snr: snr_correct[snr] / snr_total[snr] for snr in sorted(snr_correct.keys())}
    return acc_by_snr


@torch.no_grad()
def evaluate_with_denoising(model, loader, device, num_iterations=1):
    """
    Evaluate using iterative denoising at inference.
    
    Phase 1: Denoise the signal N times
    Phase 2: Classify the denoised signal
    """
    model.eval()
    
    snr_correct = {}
    snr_total = {}
    
    for batch in loader:
        x, labels, snrs = batch
        x = x.to(device)
        labels = labels.to(device)
        
        # Iterative denoising
        logits, _ = model.denoise_then_classify(x, num_iterations=num_iterations)
        preds = logits.argmax(dim=1)
        
        for i, snr in enumerate(snrs.numpy()):
            snr = int(snr)
            if snr not in snr_correct:
                snr_correct[snr] = 0
                snr_total[snr] = 0
            snr_correct[snr] += (preds[i] == labels[i]).item()
            snr_total[snr] += 1
    
    acc_by_snr = {snr: snr_correct[snr] / snr_total[snr] for snr in sorted(snr_correct.keys())}
    overall_acc = sum(snr_correct.values()) / sum(snr_total.values())
    
    return overall_acc, acc_by_snr


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = load_rml2016a(args.data_path)
    train_idx, val_idx, test_idx = build_split_indices(data, seed=args.seed)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Build loaders
    train_loader, val_loader, test_loader, num_classes = build_loaders(
        args, data, train_idx, val_idx, test_idx
    )
    
    # Build model
    if args.group_k > 1:
        model = UNetAMCWithMultiWindow(
            in_channels=2,
            base_channels=args.base_channels,
            depth=args.depth,
            num_classes=num_classes,
            lstm_hidden=args.lstm_hidden,
            dropout=args.dropout,
            pool_method=args.pool_method
        )
    else:
        model = UNetAMC(
            in_channels=2,
            base_channels=args.base_channels,
            depth=args.depth,
            num_classes=num_classes,
            lstm_hidden=args.lstm_hidden,
            dropout=args.dropout
        )
    
    model = model.to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    scaler = GradScaler(enabled=args.amp)
    
    # Save config
    config = vars(args)
    config["num_params"] = count_parameters(model)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    metrics_log = []
    
    for epoch in range(args.epochs):
        t0 = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, args, device, epoch)
        
        # Validate
        val_metrics = evaluate(model, val_loader, args, device, is_grouped=(args.group_k > 1))
        
        scheduler.step()
        
        elapsed = time.time() - t0
        
        # Log
        log_entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_cls_loss": train_metrics["cls_loss"],
            "train_denoise_loss": train_metrics["denoise_loss"],
            "train_acc": train_metrics["acc"],
            "val_acc": val_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": elapsed
        }
        metrics_log.append(log_entry)
        
        print(f"Epoch {epoch:3d} | "
              f"Train Acc: {train_metrics['acc']:.4f} | "
              f"Val Acc: {val_metrics['acc']:.4f} | "
              f"Loss: {train_metrics['loss']:.4f} | "
              f"Denoise: {train_metrics['denoise_loss']:.4f} | "
              f"Time: {elapsed:.1f}s")
        
        # Checkpointing
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "config": config
            }, out_dir / "best.pt")
            print(f"  → New best: {best_val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Save last
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_metrics["acc"],
            "config": config
        }, out_dir / "last.pt")
        
        # Save metrics
        with open(out_dir / "metrics.jsonl", "w") as f:
            for entry in metrics_log:
                f.write(json.dumps(entry) + "\n")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Final Evaluation on Test Set")
    print("=" * 60)
    
    # Load best model
    ckpt = torch.load(out_dir / "best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Evaluate without denoising at inference
    print("\n1. Direct classification (no denoising at inference):")
    acc_by_snr = evaluate_by_snr(model, test_loader, device)
    overall_acc = np.mean(list(acc_by_snr.values()))
    print(f"   Overall accuracy: {overall_acc:.4f}")
    print(f"   Mean(acc by SNR): {np.mean(list(acc_by_snr.values())):.4f}")
    
    # Save per-SNR results
    with open(out_dir / "test_acc_by_snr_direct.json", "w") as f:
        json.dump(acc_by_snr, f, indent=2)
    
    # Evaluate with denoising at inference (if single-window model)
    if args.group_k == 1:
        print("\n2. With iterative denoising at inference:")
        for n_iter in [1, 2, 3]:
            acc, acc_by_snr_denoise = evaluate_with_denoising(
                model, test_loader, device, num_iterations=n_iter
            )
            print(f"   {n_iter} iteration(s): {acc:.4f}")
            
            with open(out_dir / f"test_acc_by_snr_denoise_{n_iter}.json", "w") as f:
                json.dump(acc_by_snr_denoise, f, indent=2)
    
    print("\nDone! Results saved to:", out_dir)


if __name__ == "__main__":
    main()
