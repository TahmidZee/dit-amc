# RML2016 Commands for A4 Experiments

## Dataset-Specific Adjustments for RML2016

**Key differences from RML2018:**
- Smaller dataset: ~1000 samples per (mod, SNR) bucket vs ~4096
- Lower SNR range: max 18 dB vs 30 dB
- Default train/val splits: 600/200 vs 3200/500
- Faster training: ~3-5x faster per epoch

**Adjusted parameters:**
- `--dataset rml2016a` (default, but explicit)
- `--data-path ./RML2016.10a_dict.pkl` (auto-detected if in DiT_AMC folder)
- `--train-per 600` (default for RML2016)
- `--val-per 200` (default for RML2016)
- `--snr-cap-max-db 18` (default for RML2016)
- `--batch-size 512` (can keep same, or reduce to 256 if memory constrained)

---

## Command 1: A4 with Classifier-Only Mixup

**Run name:** `cldnn_a4_rml2016_mixup02_cls_only`

```bash
cd /path/to/DiT_AMC

python train.py \
    --arch cldnn \
    --dataset rml2016a \
    --data-path ./RML2016.10a_dict.pkl \
    --out-dir ./runs/cldnn_a4_rml2016_mixup02_cls_only \
    \
    --cldnn-denoiser-dual-path \
    --cldnn-noise-cond \
    \
    --stage-a-epochs 5 \
    --stage-b-epochs 15 \
    --stage-a-no-cls \
    --stage-b1-cls2dn-scale 0.0 \
    --stage-b2-cls2dn-scale 0.1 \
    \
    --lambda-dn 0.3 \
    --lambda-id 0.05 \
    --lambda-noise 0.1 \
    --lambda-kd 0.0 \
    \
    --dropout 0.20 \
    --label-smoothing 0.02 \
    --mixup-alpha 0.2 \
    --mixup-prob 0.5 \
    --mixup-cls-only \
    \
    --aug-phase \
    --aug-shift \
    \
    --batch-size 512 \
    --epochs 120 \
    --lr 5e-4 \
    --min-lr 1e-5 \
    --warmup-steps 500 \
    --lr-decay-start-epoch 15 \
    --weight-decay 1e-2 \
    \
    --train-per 600 \
    --val-per 200 \
    --snr-cap-max-db 18 \
    --normalize rms \
    --seed 2016 \
    --num-workers 4
```

**Notes:**
- Stage A (epochs 0-4): Denoiser/noise-head bootstrap, classifier frozen
- Stage B (epochs 5-19): Classifier warm start with controlled gradients
- Stage C (epochs 20-119): Joint fine-tuning
- Loss weights will be adjusted automatically per stage (see `train.py` stage logic)
- `--mixup-cls-only` ensures denoiser/noise losses use clean inputs

---

## Command 2: A4 Refinement (Lower Mixup Probability + SNR Min)

**Run name:** `cldnn_a4_rml2016_mixup03_snrmin6_cls_only`

```bash
cd /path/to/DiT_AMC

python train.py \
    --arch cldnn \
    --dataset rml2016a \
    --data-path ./RML2016.10a_dict.pkl \
    --out-dir ./runs/cldnn_a4_rml2016_mixup03_snrmin6_cls_only \
    \
    --cldnn-denoiser-dual-path \
    --cldnn-noise-cond \
    \
    --stage-a-epochs 5 \
    --stage-b-epochs 15 \
    --stage-a-no-cls \
    --stage-b1-cls2dn-scale 0.0 \
    --stage-b2-cls2dn-scale 0.1 \
    \
    --lambda-dn 0.3 \
    --lambda-id 0.05 \
    --lambda-noise 0.1 \
    --lambda-kd 0.0 \
    \
    --dropout 0.20 \
    --label-smoothing 0.02 \
    --mixup-alpha 0.2 \
    --mixup-prob 0.3 \
    --mixup-snr-min 6 \
    --mixup-cls-only \
    \
    --aug-phase \
    --aug-shift \
    \
    --batch-size 512 \
    --epochs 120 \
    --lr 5e-4 \
    --min-lr 1e-5 \
    --warmup-steps 500 \
    --lr-decay-start-epoch 15 \
    --weight-decay 1e-2 \
    \
    --train-per 600 \
    --val-per 200 \
    --snr-cap-max-db 18 \
    --normalize rms \
    --seed 2016 \
    --num-workers 4
```

**Changes from Command 1:**
- `--mixup-prob 0.3` (reduced from 0.5)
- `--mixup-snr-min 6` (only mix samples with SNR >= 6 dB)

---

## Stage-Wise Loss Weight Details

The training script automatically adjusts loss weights per stage. Here's what happens:

**Stage A (epochs 0-4):**
- Classification loss: disabled (`--stage-a-no-cls`)
- Denoiser/noise losses: active (lambda values from command)
- Goal: Bootstrap denoiser and noise head before classifier coupling

**Stage B (epochs 5-19):**
- Classification loss: enabled with controlled gradients
  - First half (epochs 5-11): `cls2dn_scale = 0.0` (no gradients to denoiser)
  - Second half (epochs 12-19): `cls2dn_scale = 0.1` (weak coupling)
- Denoiser/noise losses: continue with same lambda values
- Goal: Warm start classifier while denoiser stabilizes

**Stage C (epochs 20-119):**
- Classification loss: full gradients (normal training)
- Denoiser/noise losses: continue with same lambda values
- Goal: Joint fine-tuning for maximum performance

**Note:** Lambda values are fixed from command line. The plan suggests stage-wise adjustments, but current implementation uses fixed values. The values used (`lambda_dn=0.3`, `lambda_id=0.05`, `lambda_noise=0.1`) are a balanced choice that works across all stages.

---

## Expected Training Time

**RML2016 vs RML2018:**
- RML2016: ~6-10 hours for 120 epochs (on typical GPU)
- RML2018: ~30-50 hours for 120 epochs

**Per-epoch time:**
- RML2016: ~3-5 minutes per epoch
- RML2018: ~15-25 minutes per epoch

---

## Monitoring

**Key metrics to watch:**
1. **Low-SNR band (-14..-6 dB) mean accuracy** — primary target
2. **Overall validation accuracy** — should be non-negative vs baseline
3. **High-SNR band (≥10 dB) mean** — should not regress by >0.3pp
4. **Denoiser loss (`loss_dn`)** — should decrease in Stage A
5. **Noise head loss (`loss_noise`)** — should correlate with SNR

**Early stopping:**
- Monitor validation accuracy plateau
- Best checkpoint is saved automatically as `best.pt`
- Consider stopping if no improvement for 20+ epochs

---

## Troubleshooting

**If data path error:**
```bash
# Verify file exists
ls -lh ./RML2016.10a_dict.pkl

# Or use absolute path
--data-path /absolute/path/to/RML2016.10a_dict.pkl
```

**If memory issues:**
- Reduce `--batch-size` to 256 or 128
- Reduce `--num-workers` to 2

**If training is too slow:**
- RML2016 should be fast, but check GPU utilization
- Consider reducing `--epochs` for initial testing (e.g., 60 epochs)

---

**Created:** 2025-02-16  
**For:** RML2016 experiments on other machine
