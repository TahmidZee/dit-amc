# RML2016 C-Series V2 — Full Wave 1 Commands

## Scope

These are the revised **Wave 1 (8 runs)** commands aligned with:
- `RML2016_C_SERIES_PLAN.md` (current version)
- ST-ACF expert branch search
- no-denoiser expert deconfound run

All commands below are for **Goose**.

- repo: `/home/tahit/Modulation/AMR-Benchmark/RML201610a/DiT_AMC`
- python: `/home/tahit/Modulation/AMR-Benchmark/venv2/bin/python`
- data: `/home/tahit/Modulation/RML2016.10a_dict.pkl`

For Athena, adjust only: `cd`, python path, `--data-path`, and `--out-dir` root.

---

## W1-0: dn48 control

```bash
cd /home/tahit/Modulation/AMR-Benchmark/RML201610a/DiT_AMC && \
/home/tahit/Modulation/AMR-Benchmark/venv2/bin/python train.py \
  --arch cldnn \
  --dataset rml2016a \
  --data-path /home/tahit/Modulation/RML2016.10a_dict.pkl \
  --out-dir ./runs/rml2016/w1_dn48_control \
  --cldnn-denoiser \
  --cldnn-denoiser-dual-path \
  --cldnn-denoiser-base-ch 48 \
  --cldnn-denoiser-dropout 0.05 \
  --fit-noise-proxy-calibration \
  --stage-a-epochs 12 \
  --stage-b-epochs 16 \
  --stage-a-no-cls \
  --stage-b1-cls2dn-scale 0.0 \
  --stage-b2-cls2dn-scale 0.1 \
  --lambda-dn 0.3 \
  --lambda-id 0.03 \
  --lambda-noise 0.1 \
  --lambda-snr 0.0 \
  --dropout 0.15 \
  --label-smoothing 0.02 \
  --aug-phase \
  --aug-shift \
  --amp \
  --batch-size 512 \
  --epochs 120 \
  --lr 5e-4 \
  --min-lr 1e-5 \
  --warmup-steps 500 \
  --lr-decay-start-epoch 15 \
  --weight-decay 1e-4 \
  --early-stop-patience 25 \
  --train-per 600 \
  --val-per 200 \
  --snr-cap-max-db 18 \
  --normalize rms \
  --seed 2016 \
  --num-workers 4
```

## W1-1: expert raw baseline

```bash
cd /home/tahit/Modulation/AMR-Benchmark/RML201610a/DiT_AMC && \
/home/tahit/Modulation/AMR-Benchmark/venv2/bin/python train.py \
  --arch cldnn \
  --dataset rml2016a \
  --data-path /home/tahit/Modulation/RML2016.10a_dict.pkl \
  --out-dir ./runs/rml2016/w1_expert_raw \
  --cldnn-denoiser \
  --cldnn-denoiser-dual-path \
  --cldnn-denoiser-base-ch 48 \
  --cldnn-denoiser-dropout 0.05 \
  --cldnn-expert-features \
  --fit-noise-proxy-calibration \
  --stage-a-epochs 12 \
  --stage-b-epochs 16 \
  --stage-a-no-cls \
  --stage-b1-cls2dn-scale 0.0 \
  --stage-b2-cls2dn-scale 0.1 \
  --lambda-dn 0.3 \
  --lambda-id 0.03 \
  --lambda-noise 0.1 \
  --lambda-snr 0.0 \
  --dropout 0.15 \
  --label-smoothing 0.02 \
  --aug-phase \
  --aug-shift \
  --amp \
  --batch-size 512 \
  --epochs 120 \
  --lr 5e-4 \
  --min-lr 1e-5 \
  --warmup-steps 500 \
  --lr-decay-start-epoch 15 \
  --weight-decay 1e-4 \
  --early-stop-patience 25 \
  --train-per 600 \
  --val-per 200 \
  --snr-cap-max-db 18 \
  --normalize rms \
  --seed 2016 \
  --num-workers 4
```

## W1-2: expert + ST-ACF win=5

```bash
cd /home/tahit/Modulation/AMR-Benchmark/RML201610a/DiT_AMC && \
/home/tahit/Modulation/AMR-Benchmark/venv2/bin/python train.py \
  --arch cldnn \
  --dataset rml2016a \
  --data-path /home/tahit/Modulation/RML2016.10a_dict.pkl \
  --out-dir ./runs/rml2016/w1_expert_stacf5 \
  --cldnn-denoiser \
  --cldnn-denoiser-dual-path \
  --cldnn-denoiser-base-ch 48 \
  --cldnn-denoiser-dropout 0.05 \
  --cldnn-expert-features \
  --cldnn-expert-stacf-win 5 \
  --fit-noise-proxy-calibration \
  --stage-a-epochs 12 \
  --stage-b-epochs 16 \
  --stage-a-no-cls \
  --stage-b1-cls2dn-scale 0.0 \
  --stage-b2-cls2dn-scale 0.1 \
  --lambda-dn 0.3 \
  --lambda-id 0.03 \
  --lambda-noise 0.1 \
  --lambda-snr 0.0 \
  --dropout 0.15 \
  --label-smoothing 0.02 \
  --aug-phase \
  --aug-shift \
  --amp \
  --batch-size 512 \
  --epochs 120 \
  --lr 5e-4 \
  --min-lr 1e-5 \
  --warmup-steps 500 \
  --lr-decay-start-epoch 15 \
  --weight-decay 1e-4 \
  --early-stop-patience 25 \
  --train-per 600 \
  --val-per 200 \
  --snr-cap-max-db 18 \
  --normalize rms \
  --seed 2016 \
  --num-workers 4
```

## W1-3: expert + ST-ACF win=9

```bash
cd /home/tahit/Modulation/AMR-Benchmark/RML201610a/DiT_AMC && \
/home/tahit/Modulation/AMR-Benchmark/venv2/bin/python train.py \
  --arch cldnn \
  --dataset rml2016a \
  --data-path /home/tahit/Modulation/RML2016.10a_dict.pkl \
  --out-dir ./runs/rml2016/w1_expert_stacf9 \
  --cldnn-denoiser \
  --cldnn-denoiser-dual-path \
  --cldnn-denoiser-base-ch 48 \
  --cldnn-denoiser-dropout 0.05 \
  --cldnn-expert-features \
  --cldnn-expert-stacf-win 9 \
  --fit-noise-proxy-calibration \
  --stage-a-epochs 12 \
  --stage-b-epochs 16 \
  --stage-a-no-cls \
  --stage-b1-cls2dn-scale 0.0 \
  --stage-b2-cls2dn-scale 0.1 \
  --lambda-dn 0.3 \
  --lambda-id 0.03 \
  --lambda-noise 0.1 \
  --lambda-snr 0.0 \
  --dropout 0.15 \
  --label-smoothing 0.02 \
  --aug-phase \
  --aug-shift \
  --amp \
  --batch-size 512 \
  --epochs 120 \
  --lr 5e-4 \
  --min-lr 1e-5 \
  --warmup-steps 500 \
  --lr-decay-start-epoch 15 \
  --weight-decay 1e-4 \
  --early-stop-patience 25 \
  --train-per 600 \
  --val-per 200 \
  --snr-cap-max-db 18 \
  --normalize rms \
  --seed 2016 \
  --num-workers 4
```

## W1-4: expert + ST-ACF win=13

```bash
cd /home/tahit/Modulation/AMR-Benchmark/RML201610a/DiT_AMC && \
/home/tahit/Modulation/AMR-Benchmark/venv2/bin/python train.py \
  --arch cldnn \
  --dataset rml2016a \
  --data-path /home/tahit/Modulation/RML2016.10a_dict.pkl \
  --out-dir ./runs/rml2016/w1_expert_stacf13 \
  --cldnn-denoiser \
  --cldnn-denoiser-dual-path \
  --cldnn-denoiser-base-ch 48 \
  --cldnn-denoiser-dropout 0.05 \
  --cldnn-expert-features \
  --cldnn-expert-stacf-win 13 \
  --fit-noise-proxy-calibration \
  --stage-a-epochs 12 \
  --stage-b-epochs 16 \
  --stage-a-no-cls \
  --stage-b1-cls2dn-scale 0.0 \
  --stage-b2-cls2dn-scale 0.1 \
  --lambda-dn 0.3 \
  --lambda-id 0.03 \
  --lambda-noise 0.1 \
  --lambda-snr 0.0 \
  --dropout 0.15 \
  --label-smoothing 0.02 \
  --aug-phase \
  --aug-shift \
  --amp \
  --batch-size 512 \
  --epochs 120 \
  --lr 5e-4 \
  --min-lr 1e-5 \
  --warmup-steps 500 \
  --lr-decay-start-epoch 15 \
  --weight-decay 1e-4 \
  --early-stop-patience 25 \
  --train-per 600 \
  --val-per 200 \
  --snr-cap-max-db 18 \
  --normalize rms \
  --seed 2016 \
  --num-workers 4
```

## W1-5: expert + ST-ACF win=9 + no cyclo scalars

```bash
cd /home/tahit/Modulation/AMR-Benchmark/RML201610a/DiT_AMC && \
/home/tahit/Modulation/AMR-Benchmark/venv2/bin/python train.py \
  --arch cldnn \
  --dataset rml2016a \
  --data-path /home/tahit/Modulation/RML2016.10a_dict.pkl \
  --out-dir ./runs/rml2016/w1_expert_stacf9_no_cyclo \
  --cldnn-denoiser \
  --cldnn-denoiser-dual-path \
  --cldnn-denoiser-base-ch 48 \
  --cldnn-denoiser-dropout 0.05 \
  --cldnn-expert-features \
  --cldnn-expert-stacf-win 9 \
  --cldnn-no-cyclo-stats \
  --fit-noise-proxy-calibration \
  --stage-a-epochs 12 \
  --stage-b-epochs 16 \
  --stage-a-no-cls \
  --stage-b1-cls2dn-scale 0.0 \
  --stage-b2-cls2dn-scale 0.1 \
  --lambda-dn 0.3 \
  --lambda-id 0.03 \
  --lambda-noise 0.1 \
  --lambda-snr 0.0 \
  --dropout 0.15 \
  --label-smoothing 0.02 \
  --aug-phase \
  --aug-shift \
  --amp \
  --batch-size 512 \
  --epochs 120 \
  --lr 5e-4 \
  --min-lr 1e-5 \
  --warmup-steps 500 \
  --lr-decay-start-epoch 15 \
  --weight-decay 1e-4 \
  --early-stop-patience 25 \
  --train-per 600 \
  --val-per 200 \
  --snr-cap-max-db 18 \
  --normalize rms \
  --seed 2016 \
  --num-workers 4
```

## W1-6: expert + ST-ACF win=13 + no cyclo scalars

```bash
cd /home/tahit/Modulation/AMR-Benchmark/RML201610a/DiT_AMC && \
/home/tahit/Modulation/AMR-Benchmark/venv2/bin/python train.py \
  --arch cldnn \
  --dataset rml2016a \
  --data-path /home/tahit/Modulation/RML2016.10a_dict.pkl \
  --out-dir ./runs/rml2016/w1_expert_stacf13_no_cyclo \
  --cldnn-denoiser \
  --cldnn-denoiser-dual-path \
  --cldnn-denoiser-base-ch 48 \
  --cldnn-denoiser-dropout 0.05 \
  --cldnn-expert-features \
  --cldnn-expert-stacf-win 13 \
  --cldnn-no-cyclo-stats \
  --fit-noise-proxy-calibration \
  --stage-a-epochs 12 \
  --stage-b-epochs 16 \
  --stage-a-no-cls \
  --stage-b1-cls2dn-scale 0.0 \
  --stage-b2-cls2dn-scale 0.1 \
  --lambda-dn 0.3 \
  --lambda-id 0.03 \
  --lambda-noise 0.1 \
  --lambda-snr 0.0 \
  --dropout 0.15 \
  --label-smoothing 0.02 \
  --aug-phase \
  --aug-shift \
  --amp \
  --batch-size 512 \
  --epochs 120 \
  --lr 5e-4 \
  --min-lr 1e-5 \
  --warmup-steps 500 \
  --lr-decay-start-epoch 15 \
  --weight-decay 1e-4 \
  --early-stop-patience 25 \
  --train-per 600 \
  --val-per 200 \
  --snr-cap-max-db 18 \
  --normalize rms \
  --seed 2016 \
  --num-workers 4
```

## W1-7: expert + ST-ACF win=9, no denoiser (deconfound)

```bash
cd /home/tahit/Modulation/AMR-Benchmark/RML201610a/DiT_AMC && \
/home/tahit/Modulation/AMR-Benchmark/venv2/bin/python train.py \
  --arch cldnn \
  --dataset rml2016a \
  --data-path /home/tahit/Modulation/RML2016.10a_dict.pkl \
  --out-dir ./runs/rml2016/w1_expert_stacf9_no_dn \
  --cldnn-expert-features \
  --cldnn-expert-stacf-win 9 \
  --stage-a-epochs 0 \
  --stage-b-epochs 0 \
  --lambda-dn 0.0 \
  --lambda-id 0.0 \
  --lambda-noise 0.0 \
  --lambda-snr 0.0 \
  --dropout 0.15 \
  --label-smoothing 0.02 \
  --aug-phase \
  --aug-shift \
  --amp \
  --batch-size 512 \
  --epochs 120 \
  --lr 5e-4 \
  --min-lr 1e-5 \
  --warmup-steps 500 \
  --lr-decay-start-epoch 15 \
  --weight-decay 1e-4 \
  --early-stop-patience 25 \
  --train-per 600 \
  --val-per 200 \
  --snr-cap-max-db 18 \
  --normalize rms \
  --seed 2016 \
  --num-workers 4
```

---

## Optional reserve run (if a slot frees early)

```bash
cd /home/tahit/Modulation/AMR-Benchmark/RML201610a/DiT_AMC && \
/home/tahit/Modulation/AMR-Benchmark/venv2/bin/python train.py \
  --arch cldnn \
  --dataset rml2016a \
  --data-path /home/tahit/Modulation/RML2016.10a_dict.pkl \
  --out-dir ./runs/rml2016/w1_expert_stacf9_impair_aug \
  --cldnn-denoiser \
  --cldnn-denoiser-dual-path \
  --cldnn-denoiser-base-ch 48 \
  --cldnn-denoiser-dropout 0.05 \
  --cldnn-expert-features \
  --cldnn-expert-stacf-win 9 \
  --fit-noise-proxy-calibration \
  --stage-a-epochs 12 \
  --stage-b-epochs 16 \
  --stage-a-no-cls \
  --stage-b1-cls2dn-scale 0.0 \
  --stage-b2-cls2dn-scale 0.1 \
  --lambda-dn 0.3 \
  --lambda-id 0.03 \
  --lambda-noise 0.1 \
  --lambda-snr 0.0 \
  --dropout 0.15 \
  --label-smoothing 0.02 \
  --aug-phase \
  --aug-shift \
  --aug-cfo 0.005 \
  --aug-gain 0.15 \
  --amp \
  --batch-size 512 \
  --epochs 120 \
  --lr 5e-4 \
  --min-lr 1e-5 \
  --warmup-steps 500 \
  --lr-decay-start-epoch 15 \
  --weight-decay 1e-4 \
  --early-stop-patience 25 \
  --train-per 600 \
  --val-per 200 \
  --snr-cap-max-db 18 \
  --normalize rms \
  --seed 2016 \
  --num-workers 4
```
