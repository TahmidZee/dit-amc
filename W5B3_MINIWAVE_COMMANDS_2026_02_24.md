# W5B3 Mini-Wave Commands (D0-D5)

This file defines the 6-run deconfounded mini-wave for W5B.2 recovery.

## Base command (anchor skeleton)

Use this base and append each run's delta flags:

```bash
python train.py \
  --arch cldnn \
  --preset B \
  --out-dir runs/rml2016_goose/<RUN_NAME> \
  --epochs 110 \
  --batch-size 1024 \
  --num-workers 8 \
  --group-k 1 \
  --train-per 600 --val-per 200 \
  --snr-mode predict \
  --mixup-alpha 0.2 --mixup-prob 0.5 --mixup-cls-only \
  --cldnn-denoiser --cldnn-denoiser-dual-path \
  --lambda-noise 0.1 --lambda-dn 0.3 --lambda-id 0.03 \
  --moe-n-experts 2 \
  --moe-gate-type eta-sigmoid \
  --moe-gate-center 1.55 \
  --moe-gate-tau 0.30 \
  --moe-balance-lambda 0.01 \
  --moe-transition-snr-lo -8 --moe-transition-snr-hi -2
```

## Run deltas

| ID | Run name | Delta flags |
|---|---|---|
| D0 | `w5b3_eta_noaux_anchor` | `--moe-head-low-lambda 0.0 --moe-head-high-lambda 0.0` |
| D1 | `w5b3_oratrain_noaux` | D0 + `--moe-oracle-gate-train` |
| D2 | `w5b3_oratraineval_noaux` | D1 + `--moe-oracle-gate-eval` |
| D3 | `w5b3_oratraineval_headonly_a005` | D2 + `--moe-head-low-lambda 0.05 --moe-head-high-lambda 0.05 --moe-head-ce-detach-trunk --moe-head-ce-source clean --moe-head-ce-warmup 10 --moe-head-ce-ramp 10` |
| D4 | `w5b3_eta_headonly_a005` | D0 + `--moe-head-low-lambda 0.05 --moe-head-high-lambda 0.05 --moe-head-ce-detach-trunk --moe-head-ce-source clean --moe-head-ce-warmup 10 --moe-head-ce-ramp 10` |
| D5 | `w5b3_oratraineval_headonly_a030` | D2 + `--moe-head-low-lambda 0.03 --moe-head-high-lambda 0.03 --moe-head-ce-detach-trunk --moe-head-ce-source clean --moe-head-ce-warmup 10 --moe-head-ce-ramp 10` |

Optional 7th run (if one slot frees): `w5b3_eta_headonly_a010` as the stronger deployable dose-response check.

## Diagnostic script

Before launching full runs, execute:

```bash
python moe_head_ce_grad_sanity.py --json
```

Expected:
- Case A (`detach` on): trunk grad near zero, head grad non-zero.
- Case B (`detach` off): trunk grad non-zero.

## Early-kill checkpoints

- Check epochs: `25`, `35`, `50`.
- Kill if both hold:
  - best val is below wave median by `> 0.004`,
  - post-peak decline exceeds `0.015` for `>= 10` epochs.
- Exception: do not kill before epoch 35 if low-band is rising and high-band remains stable.

## Decision gates

1. Oracle routing diagnostic (D2 vs D1): pass if `low-band >= +0.010` **or** `overall >= +0.003`.
2. Head-only oracle specialization (D3 vs D2): pass if no collapse and clear low-band gain (`>= +0.010`) even if overall gain is smaller.
3. Deployable check (D4 vs D0): pass if `overall >= +0.003` with `low-band >= +0.010` and high-band drop `<= 0.003`.
4. Pivot trigger: if all D3/D4/D5 fail both `low-band >= +0.010` and `overall >= +0.003` (without violating high-band drop bound), pivot to SSL/trunk-level work.
