# UNet-AMC: Signal-Space Denoising for Automatic Modulation Classification

A unified architecture that jointly learns **signal denoising** and **modulation classification**.

## Architecture

```
                     Noisy IQ (2, 128)
                            ↓
┌──────────────────────────────────────────────────────────┐
│                 U-Net ENCODER                            │
│  Conv blocks with skip connections                       │
│  (2,128) → (64,64) → (128,32) → (256,16) → (512,8)      │
└──────────────────────────────────────────────────────────┘
                            ↓
                   Bottleneck (512, 8)
                            ↓
         ┌──────────────────┴──────────────────┐
         ↓                                      ↓
┌─────────────────────┐              ┌─────────────────────┐
│   U-Net DECODER     │              │  MCLDNN-style Head  │
│   (denoising path)  │              │  (classification)   │
│                     │              │                     │
│  Upsample + skips   │              │  BiLSTM(128)        │
│  → Clean IQ (2,128) │              │  Dense(128)         │
│                     │              │  Dense(11) → class  │
└─────────────────────┘              └─────────────────────┘
```

## Key Features

- **One model**: Shared encoder for both denoising and classification
- **Joint training**: `L = L_cls + λ * L_denoise`
- **Signal-space denoising**: Actually removes noise from IQ samples
- **MCLDNN-style classifier**: Proven LSTM + Dense architecture
- **Multi-window support**: Optional K-window aggregation

## Training

### Single-window training
```bash
python train.py \
    --data-path /path/to/RML2016.10a_dict.pkl \
    --out-dir ./runs/unet_amc_single \
    --epochs 100 \
    --batch-size 256 \
    --lr 1e-3 \
    --lambda-denoise 0.1 \
    --amp
```

### Multi-window training (K=8)
```bash
python train.py \
    --data-path /path/to/RML2016.10a_dict.pkl \
    --out-dir ./runs/unet_amc_k8 \
    --group-k 8 \
    --pool-method attn \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-3 \
    --lambda-denoise 0.1 \
    --amp
```

## Inference Modes

1. **Fast (no denoising)**: 
   - Just run encoder → classifier
   - Skip the decoder path
   
2. **Accurate (with denoising)**:
   - Run full U-Net to get denoised signal
   - Re-encode denoised signal for classification
   - Can iterate multiple times

## Denoising Strategy

The model is trained with **synthetic noisy/clean pairs**:

1. Take high-SNR samples (≥ 14 dB) as "clean" targets
2. Add AWGN to create low-SNR versions (−20 to +10 dB)
3. Train denoiser: `predict(noisy) → clean`

At inference, the model can denoise real low-SNR signals.

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | required | Path to RML2016.10a pickle |
| `--out-dir` | ./runs/unet_amc | Output directory |
| `--base-channels` | 64 | U-Net base channels |
| `--depth` | 4 | U-Net depth |
| `--lstm-hidden` | 128 | LSTM hidden size |
| `--group-k` | 1 | Number of windows (1=single) |
| `--lambda-denoise` | 0.1 | Denoising loss weight |
| `--clean-snr-threshold` | 14 | Min SNR for "clean" samples |
| `--target-snr-min` | -20 | Min synthetic noise SNR |
| `--target-snr-max` | 10 | Max synthetic noise SNR |
| `--train-mode` | joint | joint, cls_only, denoise_only |
| `--amp` | flag | Use mixed precision |

## Model Size

With default settings (base_channels=64, depth=4):
- **Parameters**: ~3.5M (lighter than DiT-AMC's 13M)
- **Inference**: Fast (single forward pass)

## Comparison with DiT-AMC

| Aspect | DiT-AMC | UNet-AMC |
|--------|---------|----------|
| Denoising location | Token-space (embeddings) | Signal-space (IQ samples) |
| Denoising at inference | No (training only) | Yes (optional) |
| Architecture | Transformer | U-Net + LSTM |
| Parameters | ~13M | ~3.5M |
| Iterative refinement | Not used | Supported |
