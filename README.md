# DiT-AMC: Diffusion-Regularized Automatic Modulation Classification

End-to-end modulation classifier using a DiT-style transformer backbone with token-space diffusion regularization. **Single forward pass at inference** (no multi-step denoising).

## Features

- **End-to-end training**: One model, one forward pass, no separate denoiser
- **SNR-blind inference**: Self-estimated SNR via internal head (no external SNR required)
- **Token-space diffusion**: Predicts clean patch embeddings (x0-prediction) from noisy tokens
- **Multi-window pooling**: Aggregate evidence from K windows per decision (recommended for low SNR)
- **CNN stem**: Optional learnable filterbank front-end for better low-level feature extraction

## Installation

```bash
pip install -r requirements.txt
```

Requires:
- Python 3.8+
- PyTorch 2.0+ (with CUDA for GPU training)
- NumPy, tqdm

## Dataset

Tested on **RML2016.10a** (RadioML 2016.10a):
- 11 modulation classes (8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM)
- 20 SNR levels: -20 to +18 dB
- 220,000 examples total (1000 per (mod, SNR) combo)
- Input shape: `(2, 128)` IQ samples per window

Download from [RadioML](http://radioml.com) and update `--data-path` accordingly.

## Quick Start

### Basic training (SNR-blind, single window)

```bash
python train.py \
  --data-path /path/to/RML2016.10a_dict.pkl \
  --out-dir ./runs/dit_amc_baseline \
  --preset B \
  --snr-mode predict \
  --batch-size 512 \
  --epochs 100 \
  --lr 1e-4 \
  --lambda-diff 0.2 \
  --lambda-snr 0.1 \
  --normalize rms \
  --amp
```

### Recommended: Multi-window pooling + CNN stem (best for low SNR)

```bash
python train.py \
  --data-path /path/to/RML2016.10a_dict.pkl \
  --out-dir ./runs/dit_amc_group8_stem \
  --preset B \
  --snr-mode predict \
  --group-k 8 \
  --group-pool attn \
  --stem-channels 64 \
  --stem-layers 2 \
  --batch-size 512 \
  --epochs 100 \
  --lr 1e-4 \
  --lambda-diff 0.2 \
  --lambda-snr 0.1 \
  --snr-balanced \
  --label-smoothing 0.1 \
  --aug-phase \
  --aug-shift \
  --t-schedule uniform \
  --t-max 200 \
  --p-clean 0.3 \
  --phase1-epochs 10 \
  --phase1-lambda-diff 0.0 \
  --phase1-p-clean 1.0 \
  --normalize rms \
  --amp
```

### Evaluation only

```bash
python train.py \
  --data-path /path/to/RML2016.10a_dict.pkl \
  --preset B \
  --snr-mode predict \
  --eval-only \
  --ckpt ./runs/dit_amc_group8_stem/best.pt
```

## Key Arguments

### Model architecture
- `--preset {S,B}`: Model size (S=small, B=base). S: p=8, d=192, depth=10; B: p=4, d=256, depth=12
- `--group-k N`: Multi-window pooling (N random windows from same (mod,SNR) bucket). Default: 1
- `--group-pool {mean,attn}`: Pool across the `group-k` windows (mean or learned attention). Default: mean
- `--stem-channels C`: CNN stem channels (0 to disable). Default: 0
- `--stem-layers L`: CNN stem depth. Default: 0

### Training
- `--batch-size N`: Batch size (reduce if OOM with `group-k > 1`)
- `--epochs N`: Training epochs
- `--lr F`: Learning rate (default: preset-dependent)
- `--amp`: Enable mixed precision (BF16 on Ampere/Hopper, FP16 otherwise)
- `--label-smoothing F`: Cross-entropy label smoothing (e.g. 0.05–0.1). Default: 0.0
- `--snr-balanced`: Use SNR-balanced sampling for training batches (reduces domination by high-SNR “easy” samples)
- `--snr-balance-power F`: Weight exponent for SNR-balanced sampling. Default: 1.0

### Train-time signal augmentations (label-preserving)
Applied to the **training** split only (after normalization):

- `--aug-phase`: Random global phase rotation per window
- `--aug-shift`: Random circular time shift per window
- `--aug-gain F`: Gain jitter magnitude (e.g. 0.2 ⇒ ×[0.8,1.2]); mostly redundant if `--normalize rms`
- `--aug-cfo F`: Max CFO in cycles/sample (keep small, e.g. 0.005–0.02)

### Diffusion regularization
- `--lambda-diff F`: Diffusion loss weight (default: 0.2)
- `--t-schedule {uniform,snr}`: Timestep sampling (default: uniform)
- `--t-max N`: Maximum diffusion timestep (default: 1000)
- `--p-clean F`: Probability of clean (t=0) samples during training (default: 0.0)

### SNR handling
- `--snr-mode {predict,known,none}`: SNR conditioning mode
  - `predict`: Self-estimated SNR (recommended, SNR-blind externally)
  - `known`: Oracle SNR (upper bound, ablation only)
  - `none`: No SNR conditioning
- `--lambda-snr F`: SNR head loss weight (default: 0.1)

### Two-phase curriculum (optional)
- `--phase1-epochs N`: Warmup epochs with `lambda_diff=0`, `p_clean=1` (default: 0)
- `--phase1-lambda-diff F`: Override `lambda_diff` in phase 1 (default: 0.0)
- `--phase1-p-clean F`: Override `p_clean` in phase 1 (default: 1.0)

### Data filtering (for cross-SNR experiments)
- `--train-snrs LIST`: Comma-separated SNR list (e.g., `-2,0,2,4,6,8,10,12,14,16,18`)
- `--val-snrs LIST`: Validation SNRs (default: same as train)
- `--test-snrs LIST`: Test SNRs (default: all)

## Hyperparameter Sweep

Use `sweep.py` for grid search:

```bash
python sweep.py \
  --out-dir-root ./runs/sweep \
  --grid "t_max=100,200;p_clean=0.2,0.3;lambda_diff=0.1,0.2" \
  --base-args "--data-path /path/to/RML2016.10a_dict.pkl --preset B --snr-mode predict --epochs 60 --batch-size 512 --lr 1e-4 --normalize rms --amp"
```

Results are logged to `runs/sweep/sweep_results.jsonl`.

## Outputs

Each training run produces:
- `best.pt`, `last.pt`: Model checkpoints
- `metrics.jsonl`: Per-epoch metrics (train/val loss, accuracy, LR, etc.)
- `val_acc_by_snr.json`, `test_acc_by_snr.json`: Accuracy breakdown by SNR level

## Early stopping (recommended)

Use early stopping to prevent overfitting once validation accuracy plateaus:

```
--early-stop-patience 15 --early-stop-min-delta 0.001
```

## Notes

- **Inference**: Uses original input with `t=0` (no extra diffusion noise injected)
- **SNR-blind**: `--snr-mode predict` does not require external SNR at inference
- **Multi-window**: `--group-k 8` averages predictions across 8 independent windows (one forward pass)
- **CNN stem**: Improves low-SNR robustness by learning better local features before tokenization

## Citation

If you use this code, please cite:

```bibtex
@software{dit_amc,
  title = {DiT-AMC: Diffusion-Regularized Automatic Modulation Classification},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/dit-amc}
}
```

## License

[Specify your license here, e.g., MIT, Apache 2.0]
