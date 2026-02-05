"""
Training script for proper DDPM denoising + classification.

Two-phase approach:
1. Train DDPM on clean (high-SNR) samples to learn the clean signal distribution
2. Train classifier on all samples (optionally with DDPM denoising)

At inference:
- Low-SNR signals are denoised using DDPM reverse process
- Classifier predicts on denoised signals
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
from torch.amp import autocast, GradScaler

from ddpm import DDPM, count_parameters
from model import MCLDNNClassifier, UNetEncoder
from data_ddpm import (
    load_rml2016a,
    build_split_indices,
    RML2016aCleanDataset,
    RML2016aDDPMDataset,
    RML2016aFullDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DDPM + Classifier")
    
    # Data
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="./runs/ddpm_amc")
    parser.add_argument("--normalize", type=str, default="rms")
    
    # DDPM
    parser.add_argument("--ddpm-base-channels", type=int, default=64)
    parser.add_argument("--ddpm-depth", type=int, default=4)
    parser.add_argument("--ddpm-timesteps", type=int, default=1000)
    parser.add_argument("--ddpm-schedule", type=str, default="cosine")
    parser.add_argument("--denoise-steps", type=int, default=50,
                        help="Number of denoising steps at inference")
    
    # Classifier
    parser.add_argument("--cls-base-channels", type=int, default=64)
    parser.add_argument("--cls-depth", type=int, default=4)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    
    # Training
    parser.add_argument("--ddpm-epochs", type=int, default=50,
                        help="Epochs to train DDPM")
    parser.add_argument("--cls-epochs", type=int, default=50,
                        help="Epochs to train classifier")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    
    # SNR thresholds for pairing
    parser.add_argument("--high-snr-threshold", type=int, default=14)
    parser.add_argument("--low-snr-threshold", type=int, default=6)
    
    # Hardware
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2016)
    
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SimpleClassifier(nn.Module):
    """Simple CNN classifier for baseline comparison"""
    
    def __init__(self, in_channels=2, base_channels=64, num_classes=11, dropout=0.5):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, 7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base_channels, base_channels * 2, 5, padding=2),
            nn.BatchNorm1d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.BatchNorm1d(base_channels * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 2, num_classes),
        )
    
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)


def train_ddpm_epoch(ddpm, loader, optimizer, scaler, args, device):
    """Train DDPM for one epoch"""
    ddpm.train()
    total_loss = 0
    
    for batch in loader:
        x, label, snr = batch
        x = x.to(device)
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=args.amp):
            loss = ddpm.training_loss(x)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def train_classifier_epoch(classifier, loader, optimizer, scaler, args, device, ddpm=None):
    """Train classifier for one epoch (optionally with DDPM denoising)"""
    classifier.train()
    if ddpm is not None:
        ddpm.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        x, labels, snr = batch
        x = x.to(device)
        labels = labels.to(device)
        
        # Optionally denoise before classification
        if ddpm is not None:
            with torch.no_grad():
                # Treat observation as x_t at an SNR-derived timestep, then reverse to x_0
                x = ddpm.denoise_observation(
                    x,
                    snr_db=snr,
                    num_steps=args.denoise_steps,
                    deterministic=True,
                    strength=1.0,
                )
        
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=args.amp):
            logits = classifier(x)
            loss = F.cross_entropy(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate_classifier(classifier, loader, device, ddpm=None, denoise_steps=50):
    """Evaluate classifier (optionally with DDPM denoising)"""
    classifier.eval()
    if ddpm is not None:
        ddpm.eval()
    
    correct = 0
    total = 0
    
    for batch in loader:
        x, labels, snr = batch
        x = x.to(device)
        labels = labels.to(device)
        
        if ddpm is not None:
            x = ddpm.denoise_observation(
                x,
                snr_db=snr,
                num_steps=denoise_steps,
                deterministic=True,
                strength=1.0,
            )
        
        logits = classifier(x)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return correct / total


@torch.no_grad()
def evaluate_by_snr(classifier, loader, device, ddpm=None, denoise_steps=50):
    """Evaluate accuracy per SNR level"""
    classifier.eval()
    if ddpm is not None:
        ddpm.eval()
    
    snr_correct = {}
    snr_total = {}
    
    for batch in loader:
        x, labels, snrs = batch
        x = x.to(device)
        labels = labels.to(device)
        
        if ddpm is not None:
            x = ddpm.denoise_observation(
                x,
                snr_db=snrs,
                num_steps=denoise_steps,
                deterministic=True,
                strength=1.0,
            )
        
        logits = classifier(x)
        preds = logits.argmax(dim=1)
        
        for i, snr in enumerate(snrs.numpy()):
            snr = int(snr)
            if snr not in snr_correct:
                snr_correct[snr] = 0
                snr_total[snr] = 0
            snr_correct[snr] += (preds[i] == labels[i]).item()
            snr_total[snr] += 1
    
    acc_by_snr = {snr: snr_correct[snr] / snr_total[snr] for snr in sorted(snr_correct.keys())}
    overall = sum(snr_correct.values()) / sum(snr_total.values())
    
    return overall, acc_by_snr


def main():
    args = parse_args()
    set_seed(args.seed)
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = load_rml2016a(args.data_path)
    train_idx, val_idx, test_idx = build_split_indices(data, seed=args.seed)
    
    # Create datasets
    # For DDPM training: only clean (high-SNR) samples
    clean_train_ds = RML2016aCleanDataset(
        data, train_idx, 
        normalize=args.normalize,
        high_snr_threshold=args.high_snr_threshold
    )
    clean_val_ds = RML2016aCleanDataset(
        data, val_idx,
        normalize=args.normalize,
        high_snr_threshold=args.high_snr_threshold
    )
    
    # For classifier: all samples
    full_train_ds = RML2016aFullDataset(data, train_idx, normalize=args.normalize)
    full_val_ds = RML2016aFullDataset(data, val_idx, normalize=args.normalize)
    full_test_ds = RML2016aFullDataset(data, test_idx, normalize=args.normalize)
    
    num_classes = full_train_ds.num_classes
    
    # Create loaders
    clean_train_loader = DataLoader(
        clean_train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    clean_val_loader = DataLoader(
        clean_val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    full_train_loader = DataLoader(
        full_train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    full_val_loader = DataLoader(
        full_val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    full_test_loader = DataLoader(
        full_test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # =========================================================================
    # Phase 1: Train DDPM on clean samples
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 1: Training DDPM on clean (high-SNR) samples")
    print("=" * 60)
    
    ddpm = DDPM(
        in_channels=2,
        base_channels=args.ddpm_base_channels,
        time_dim=256,
        depth=args.ddpm_depth,
        timesteps=args.ddpm_timesteps,
        schedule=args.ddpm_schedule,
        device=device
    ).to(device)
    
    print(f"DDPM parameters: {count_parameters(ddpm):,}")
    
    ddpm_optimizer = torch.optim.AdamW(ddpm.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ddpm_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ddpm_optimizer, T_max=args.ddpm_epochs)
    ddpm_scaler = GradScaler('cuda', enabled=args.amp)
    
    best_ddpm_loss = float('inf')
    ddpm_patience = 0
    
    for epoch in range(args.ddpm_epochs):
        t0 = time.time()
        
        train_loss = train_ddpm_epoch(ddpm, clean_train_loader, ddpm_optimizer, ddpm_scaler, args, device)
        
        # Validation loss
        ddpm.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in clean_val_loader:
                x, _, _ = batch
                x = x.to(device)
                val_loss += ddpm.training_loss(x).item()
        val_loss /= len(clean_val_loader)
        
        ddpm_scheduler.step()
        elapsed = time.time() - t0
        
        print(f"DDPM Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")
        
        if val_loss < best_ddpm_loss:
            best_ddpm_loss = val_loss
            ddpm_patience = 0
            torch.save(ddpm.state_dict(), out_dir / "ddpm_best.pt")
            print(f"  → New best DDPM: {val_loss:.4f}")
        else:
            ddpm_patience += 1
        
        if ddpm_patience >= args.patience:
            print(f"DDPM early stopping at epoch {epoch}")
            break
    
    # Load best DDPM
    ddpm.load_state_dict(torch.load(out_dir / "ddpm_best.pt", weights_only=True))
    
    # =========================================================================
    # Phase 2: Train classifier (with and without DDPM denoising)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Training classifier")
    print("=" * 60)
    
    # Train two classifiers: with and without DDPM
    for use_ddpm in [False, True]:
        name = "with_ddpm" if use_ddpm else "without_ddpm"
        print(f"\n--- Training classifier {name} ---")
        
        classifier = SimpleClassifier(
            in_channels=2,
            base_channels=args.cls_base_channels,
            num_classes=num_classes,
            dropout=args.dropout
        ).to(device)
        
        print(f"Classifier parameters: {count_parameters(classifier):,}")
        
        cls_optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        cls_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(cls_optimizer, T_max=args.cls_epochs)
        cls_scaler = GradScaler('cuda', enabled=args.amp)
        
        best_val_acc = 0
        cls_patience = 0
        
        for epoch in range(args.cls_epochs):
            t0 = time.time()
            
            train_loss, train_acc = train_classifier_epoch(
                classifier, full_train_loader, cls_optimizer, cls_scaler, args, device,
                ddpm=ddpm if use_ddpm else None
            )
            
            val_acc = evaluate_classifier(
                classifier, full_val_loader, device,
                ddpm=ddpm if use_ddpm else None,
                denoise_steps=args.denoise_steps
            )
            
            cls_scheduler.step()
            elapsed = time.time() - t0
            
            print(f"  Epoch {epoch:3d} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                cls_patience = 0
                torch.save(classifier.state_dict(), out_dir / f"classifier_{name}_best.pt")
                print(f"    → New best: {val_acc:.4f}")
            else:
                cls_patience += 1
            
            if cls_patience >= args.patience:
                print(f"  Classifier early stopping at epoch {epoch}")
                break
        
        # Load best classifier and evaluate
        classifier.load_state_dict(torch.load(out_dir / f"classifier_{name}_best.pt", weights_only=True))
        
        print(f"\n  Final test evaluation ({name}):")
        overall, acc_by_snr = evaluate_by_snr(
            classifier, full_test_loader, device,
            ddpm=ddpm if use_ddpm else None,
            denoise_steps=args.denoise_steps
        )
        print(f"  Overall accuracy: {overall:.4f}")
        print(f"  Mean(acc by SNR): {np.mean(list(acc_by_snr.values())):.4f}")
        
        # Save results
        with open(out_dir / f"test_acc_by_snr_{name}.json", "w") as f:
            json.dump(acc_by_snr, f, indent=2)
    
    # =========================================================================
    # Final comparison: Test with different denoising steps
    # =========================================================================
    print("\n" + "=" * 60)
    print("Testing DDPM denoising with different step counts")
    print("=" * 60)
    
    classifier.load_state_dict(torch.load(out_dir / "classifier_with_ddpm_best.pt", weights_only=True))
    
    for steps in [10, 25, 50, 100, 200]:
        overall, _ = evaluate_by_snr(classifier, full_test_loader, device, ddpm=ddpm, denoise_steps=steps)
        print(f"  {steps} steps: {overall:.4f}")
    
    # Save config
    config = vars(args)
    config["ddpm_params"] = count_parameters(ddpm)
    config["classifier_params"] = count_parameters(classifier)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nDone! Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
