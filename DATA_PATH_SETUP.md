# Data Path Setup Guide

## Overview
This guide ensures correct data paths for running experiments on different machines:
- **Goose (current machine)**: Run RML2018 experiments
- **Other machine**: Run RML2016 experiments

## Auto-Detection Logic

The `train.py` script now auto-detects data paths in this order:

### For RML2016 (`--dataset rml2016a`):
1. Checks for `RML2016.10a_dict.pkl` in the `DiT_AMC` folder (script directory)
2. Falls back to `/home/tahit/Modulation/RML2016.10a_dict.pkl`

### For RML2018 (`--dataset rml2018a`):
- Uses `/home/tahit/Modulation/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5`

## Setup Instructions

### On Goose (RML2018 runs):
**Data file location:**
```
/home/tahit/Modulation/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5
```

**Example command:**
```bash
python train.py \
    --dataset rml2018a \
    --data-path /home/tahit/Modulation/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5 \
    # ... other args
```

**Note:** The auto-detection will work, but it's safer to explicitly specify `--data-path` for RML2018 runs.

### On Other Machine (RML2016 runs):
**Data file location:**
```
/path/to/DiT_AMC/RML2016.10a_dict.pkl
```

**Example command:**
```bash
cd /path/to/DiT_AMC
python train.py \
    --dataset rml2016a \
    --data-path ./RML2016.10a_dict.pkl \
    # ... other args
```

**Note:** Since `RML2016.10a_dict.pkl` is in the `DiT_AMC` folder, the auto-detection will find it automatically. You can omit `--data-path` if running from the `DiT_AMC` directory.

## Verification

### On Goose:
```bash
cd /home/tahit/Modulation/AMR-Benchmark/RML201610a/DiT_AMC

# Verify RML2018 file exists
ls -lh /home/tahit/Modulation/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5

# Test auto-detection (should show RML2016 path, but you'll use RML2018 explicitly)
python train.py --help | grep -A 2 "data-path"
```

### On Other Machine:
```bash
cd /path/to/DiT_AMC

# Verify RML2016 file exists
ls -lh RML2016.10a_dict.pkl

# Test auto-detection (should find RML2016 in current directory)
python train.py --help | grep -A 2 "data-path"
```

## Troubleshooting

### Error: "FileNotFoundError: [Errno 2] No such file or directory"
- **Solution:** Explicitly specify `--data-path` with the full path to your data file.

### Error: "KeyError" or "Invalid dataset format"
- **Solution:** Make sure `--dataset` matches your data file:
  - `--dataset rml2016a` for `.pkl` files
  - `--dataset rml2018a` for `.hdf5` files

### Auto-detection not working
- **Solution:** Always explicitly specify `--data-path` in your training commands.

## Quick Reference

| Machine | Dataset | Default Path | Explicit Path |
|---------|---------|--------------|---------------|
| Goose | RML2018 | `/home/tahit/Modulation/radioml2018/GOLD_XYZ_OSC.0001_1024.hdf5` | Use `--data-path` explicitly |
| Other | RML2016 | `./RML2016.10a_dict.pkl` (in DiT_AMC folder) | Can omit if in DiT_AMC folder |
