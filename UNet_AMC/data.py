"""
Dataset for UNet-AMC with synthetic noisy/clean signal pairs.

Strategy:
- Use high-SNR samples (+18dB, +16dB, etc.) as "clean" targets
- Synthetically add AWGN to create low-SNR versions
- Train denoiser to recover clean from noisy
"""

import pickle
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Suppress deprecation warnings from pickle loading
warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_rml2016a(path):
    """Load RML2016.10a dataset"""
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def get_snr_from_key(key):
    """Extract SNR value from data key"""
    return key[1]


def normalize_iq(x, method='rms'):
    """Normalize IQ signal"""
    if method == 'rms':
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + 1e-8)
        return x / rms
    elif method == 'max':
        max_val = np.max(np.abs(x), axis=-1, keepdims=True) + 1e-8
        return x / max_val
    elif method == 'none':
        return x
    else:
        raise ValueError(f"Unknown normalization: {method}")


def add_awgn(signal, target_snr_db):
    """
    Add AWGN to achieve target SNR.
    
    Args:
        signal: (2, L) IQ signal (assumed to be at high SNR / "clean")
        target_snr_db: desired SNR in dB
    
    Returns:
        noisy_signal: (2, L) signal with added noise
    """
    # Signal power
    signal_power = np.mean(signal ** 2)
    
    # Noise power for target SNR
    snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate noise
    noise = np.random.randn(*signal.shape) * np.sqrt(noise_power)
    
    return signal + noise


class RML2016aDataset(Dataset):
    """Basic dataset for single-window classification"""
    
    def __init__(self, data, indices, normalize='rms'):
        """
        Args:
            data: loaded RML2016.10a dict
            indices: list of (key, sample_idx) tuples
            normalize: normalization method
        """
        self.data = data
        self.indices = indices
        self.normalize = normalize
        
        # Build label mapping
        mods = sorted(set(k[0] for k in data.keys()))
        self.mod_to_idx = {m: i for i, m in enumerate(mods)}
        self.idx_to_mod = {i: m for m, i in self.mod_to_idx.items()}
        self.num_classes = len(mods)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        key, sample_idx = self.indices[idx]
        mod, snr = key
        
        x = self.data[key][sample_idx].copy()  # (2, 128)
        x = normalize_iq(x, self.normalize)
        
        label = self.mod_to_idx[mod]
        
        return torch.tensor(x, dtype=torch.float32), label, snr


class RML2016aDenoisingDataset(Dataset):
    """
    Dataset with synthetic noisy/clean pairs for denoising training.
    
    Uses ALL samples for classification, but only applies denoising loss
    to high-SNR samples where we have valid clean targets.
    
    Returns a flag indicating whether denoising loss should be computed.
    """
    
    def __init__(
        self,
        data,
        indices,
        normalize='rms',
        clean_snr_threshold=14,  # Consider samples >= this as "clean"
        target_snr_range=(-20, 10),  # Range of SNRs to train denoiser on
        augment=True
    ):
        """
        Args:
            data: loaded RML2016.10a dict
            indices: list of (key, sample_idx) tuples
            normalize: normalization method
            clean_snr_threshold: SNR above which samples are considered "clean"
            target_snr_range: (min_snr, max_snr) for synthetic noise addition
            augment: whether to apply augmentations
        """
        self.data = data
        self.normalize = normalize
        self.clean_snr_threshold = clean_snr_threshold
        self.target_snr_range = target_snr_range
        self.augment = augment
        
        # Build label mapping
        mods = sorted(set(k[0] for k in data.keys()))
        self.mod_to_idx = {m: i for i, m in enumerate(mods)}
        self.idx_to_mod = {i: m for m, i in self.mod_to_idx.items()}
        self.num_classes = len(mods)
        
        # Use ALL indices for training
        self.indices = indices
        
        # Count how many are high-SNR (for logging)
        n_clean = sum(1 for k, i in indices if get_snr_from_key(k) >= clean_snr_threshold)
        print(f"Hybrid dataset: {len(indices)} total samples, "
              f"{n_clean} clean (SNR >= {clean_snr_threshold}) for denoising")
    
    def __len__(self):
        return len(self.indices)
    
    def _augment(self, x):
        """Apply augmentations"""
        # Random phase rotation
        if np.random.random() < 0.5:
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            i, q = x[0], x[1]
            x = np.stack([i * cos_a - q * sin_a, i * sin_a + q * cos_a], axis=0)
        
        # Random circular time shift
        if np.random.random() < 0.5:
            shift = np.random.randint(0, x.shape[-1])
            x = np.roll(x, shift, axis=-1)
        
        return x
    
    def __getitem__(self, idx):
        """
        Returns:
            x: IQ signal (possibly with synthetic noise added for high-SNR samples)
            clean_x: clean IQ signal (target for denoising, zeros if not applicable)
            label: modulation class
            original_snr: original SNR of the sample
            has_denoise_target: 1.0 if denoising loss should be computed, 0.0 otherwise
        """
        key, sample_idx = self.indices[idx]
        mod, original_snr = key
        
        # Get the sample
        x = self.data[key][sample_idx].copy()  # (2, 128)
        
        if self.augment:
            x = self._augment(x)
        
        # Normalize
        x = normalize_iq(x, self.normalize)
        
        label = self.mod_to_idx[mod]
        
        # For high-SNR samples: create synthetic noisy version for denoising training
        if original_snr >= self.clean_snr_threshold:
            clean_x = x.copy()
            target_snr = np.random.uniform(*self.target_snr_range)
            noisy_x = add_awgn(clean_x, target_snr)
            noisy_x = normalize_iq(noisy_x, self.normalize)
            has_denoise_target = 1.0
        else:
            # Low-SNR sample: use as-is, no denoising target
            noisy_x = x.copy()
            clean_x = np.zeros_like(x)  # Placeholder, won't be used
            has_denoise_target = 0.0
        
        return (
            torch.tensor(noisy_x, dtype=torch.float32),
            torch.tensor(clean_x, dtype=torch.float32),
            label,
            original_snr,
            has_denoise_target  # Changed from target_snr to flag
        )


class RML2016aDenoisingDatasetV2(Dataset):
    """
    Improved denoising dataset that ONLY uses high-SNR samples.
    
    More principled: we only train on samples where we have a clear
    "clean" reference (high SNR) and can add controlled noise.
    """
    
    def __init__(
        self,
        data,
        indices,
        normalize='rms',
        clean_snr_threshold=14,
        target_snr_range=(-20, 10),
        augment=True
    ):
        self.data = data
        self.normalize = normalize
        self.target_snr_range = target_snr_range
        self.augment = augment
        
        # Build label mapping
        mods = sorted(set(k[0] for k in data.keys()))
        self.mod_to_idx = {m: i for i, m in enumerate(mods)}
        self.idx_to_mod = {i: m for m, i in self.mod_to_idx.items()}
        self.num_classes = len(mods)
        
        # Only keep high-SNR samples
        self.indices = [
            (k, i) for k, i in indices if get_snr_from_key(k) >= clean_snr_threshold
        ]
        
        print(f"DenoisingV2 dataset: {len(self.indices)} clean samples "
              f"(SNR >= {clean_snr_threshold})")
    
    def __len__(self):
        return len(self.indices)
    
    def _augment(self, x):
        if np.random.random() < 0.5:
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            i, q = x[0], x[1]
            x = np.stack([i * cos_a - q * sin_a, i * sin_a + q * cos_a], axis=0)
        if np.random.random() < 0.5:
            shift = np.random.randint(0, x.shape[-1])
            x = np.roll(x, shift, axis=-1)
        return x
    
    def __getitem__(self, idx):
        key, sample_idx = self.indices[idx]
        mod, original_snr = key
        
        x = self.data[key][sample_idx].copy()
        
        if self.augment:
            x = self._augment(x)
        
        x = normalize_iq(x, self.normalize)
        
        # Clean version
        clean_x = x.copy()
        
        # Add noise to create noisy version
        target_snr = np.random.uniform(*self.target_snr_range)
        noisy_x = add_awgn(clean_x, target_snr)
        noisy_x = normalize_iq(noisy_x, self.normalize)
        
        label = self.mod_to_idx[mod]
        
        return (
            torch.tensor(noisy_x, dtype=torch.float32),
            torch.tensor(clean_x, dtype=torch.float32),
            label,
            original_snr,
            target_snr
        )


class RML2016aGroupedDenoisingDataset(Dataset):
    """
    Denoising dataset with K-window grouping.
    
    Each sample returns K windows with the same (noisy, clean) pair structure.
    """
    
    def __init__(
        self,
        data,
        indices,
        group_k=8,
        normalize='rms',
        clean_snr_threshold=14,
        target_snr_range=(-20, 10),
        augment=True,
        seed=None
    ):
        self.data = data
        self.group_k = group_k
        self.normalize = normalize
        self.target_snr_range = target_snr_range
        self.augment = augment
        
        # Build label mapping
        mods = sorted(set(k[0] for k in data.keys()))
        self.mod_to_idx = {m: i for i, m in enumerate(mods)}
        self.idx_to_mod = {i: m for m, i in self.mod_to_idx.items()}
        self.num_classes = len(mods)
        
        # Only keep high-SNR samples
        self.indices = [
            (k, i) for k, i in indices if get_snr_from_key(k) >= clean_snr_threshold
        ]
        
        # Group by (mod, snr) for sampling K windows
        self.groups = {}
        for k, i in self.indices:
            if k not in self.groups:
                self.groups[k] = []
            self.groups[k].append(i)
        
        self.group_keys = list(self.groups.keys())
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        
        print(f"GroupedDenoising dataset: {len(self.indices)} samples, "
              f"{len(self.group_keys)} groups, K={group_k}")
    
    def __len__(self):
        return len(self.group_keys)
    
    def _augment(self, x):
        if np.random.random() < 0.5:
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            i, q = x[0], x[1]
            x = np.stack([i * cos_a - q * sin_a, i * sin_a + q * cos_a], axis=0)
        if np.random.random() < 0.5:
            shift = np.random.randint(0, x.shape[-1])
            x = np.roll(x, shift, axis=-1)
        return x
    
    def __getitem__(self, idx):
        key = self.group_keys[idx]
        mod, original_snr = key
        
        # Sample K windows from this group
        available = self.groups[key]
        if len(available) >= self.group_k:
            chosen = self.rng.choice(available, self.group_k, replace=False)
        else:
            chosen = self.rng.choice(available, self.group_k, replace=True)
        
        noisy_windows = []
        clean_windows = []
        target_snrs = []
        
        for sample_idx in chosen:
            x = self.data[key][sample_idx].copy()
            if self.augment:
                x = self._augment(x)
            x = normalize_iq(x, self.normalize)
            
            clean_x = x.copy()
            target_snr = np.random.uniform(*self.target_snr_range)
            noisy_x = add_awgn(clean_x, target_snr)
            noisy_x = normalize_iq(noisy_x, self.normalize)
            
            noisy_windows.append(noisy_x)
            clean_windows.append(clean_x)
            target_snrs.append(target_snr)
        
        label = self.mod_to_idx[mod]
        
        return (
            torch.tensor(np.stack(noisy_windows), dtype=torch.float32),  # (K, 2, 128)
            torch.tensor(np.stack(clean_windows), dtype=torch.float32),  # (K, 2, 128)
            label,
            original_snr,
            np.mean(target_snrs)
        )


def build_split_indices(data, train_ratio=0.6, val_ratio=0.2, seed=2016):
    """
    Split data indices into train/val/test.
    
    Returns:
        train_indices, val_indices, test_indices: lists of (key, sample_idx)
    """
    rng = np.random.default_rng(seed)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for key in data.keys():
        n_samples = len(data[key])
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        for i in indices[:n_train]:
            train_indices.append((key, i))
        for i in indices[n_train:n_train + n_val]:
            val_indices.append((key, i))
        for i in indices[n_train + n_val:]:
            test_indices.append((key, i))
    
    return train_indices, val_indices, test_indices


if __name__ == "__main__":
    # Test
    import sys
    
    data_path = "/home/tahit/Modulation/RML2016.10a_dict.pkl"
    data = load_rml2016a(data_path)
    
    train_idx, val_idx, test_idx = build_split_indices(data)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Test denoising dataset
    ds = RML2016aDenoisingDatasetV2(data, train_idx)
    noisy, clean, label, orig_snr, target_snr = ds[0]
    print(f"\nDenoising sample:")
    print(f"  Noisy shape: {noisy.shape}")
    print(f"  Clean shape: {clean.shape}")
    print(f"  Label: {label}, Orig SNR: {orig_snr}, Target SNR: {target_snr:.1f}")
    
    # Test grouped dataset
    ds_k = RML2016aGroupedDenoisingDataset(data, train_idx, group_k=8)
    noisy_k, clean_k, label, orig_snr, target_snr = ds_k[0]
    print(f"\nGrouped denoising sample:")
    print(f"  Noisy shape: {noisy_k.shape}")
    print(f"  Clean shape: {clean_k.shape}")
