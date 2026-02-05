"""
Dataset for proper DDPM training with REAL RF noise pairs.

Key idea: Pair low-SNR samples with high-SNR samples of the SAME modulation class.
This teaches the diffusion model what real RF noise looks like and how to remove it.
"""

import pickle
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_rml2016a(path):
    """Load RML2016.10a dataset"""
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


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


class RML2016aDDPMDataset(Dataset):
    """
    Dataset for DDPM training with REAL RF noise pairs.
    
    For each low-SNR sample, we pair it with a randomly chosen high-SNR sample
    from the SAME modulation class. This teaches the model real RF denoising.
    """
    
    def __init__(
        self,
        data,
        indices,
        normalize='rms',
        high_snr_threshold=14,  # SNR >= this is "clean"
        low_snr_threshold=6,    # SNR <= this is "noisy"
        augment=True
    ):
        self.data = data
        self.normalize = normalize
        self.high_snr_threshold = high_snr_threshold
        self.low_snr_threshold = low_snr_threshold
        self.augment = augment
        
        # Build label mapping
        mods = sorted(set(k[0] for k in data.keys()))
        self.mod_to_idx = {m: i for i, m in enumerate(mods)}
        self.idx_to_mod = {i: m for m, i in self.mod_to_idx.items()}
        self.num_classes = len(mods)
        
        # Organize indices by (modulation, snr_category)
        self.high_snr_by_mod = {m: [] for m in mods}  # Clean samples per mod
        self.low_snr_by_mod = {m: [] for m in mods}   # Noisy samples per mod
        self.all_indices = []
        
        for key, sample_idx in indices:
            mod, snr = key
            if snr >= high_snr_threshold:
                self.high_snr_by_mod[mod].append((key, sample_idx))
            if snr <= low_snr_threshold:
                self.low_snr_by_mod[mod].append((key, sample_idx))
            self.all_indices.append((key, sample_idx))
        
        # Create training pairs: (noisy, clean) from same modulation
        self.pairs = []
        for mod in mods:
            low_snr_samples = self.low_snr_by_mod[mod]
            high_snr_samples = self.high_snr_by_mod[mod]
            if len(low_snr_samples) > 0 and len(high_snr_samples) > 0:
                for noisy_idx in low_snr_samples:
                    self.pairs.append((noisy_idx, mod))
        
        # For DDPM training, we also need pure high-SNR samples
        self.clean_samples = []
        for mod in mods:
            self.clean_samples.extend(self.high_snr_by_mod[mod])
        
        print(f"DDPM Dataset: {len(self.pairs)} noisy-clean pairs, "
              f"{len(self.clean_samples)} clean samples for diffusion training")
    
    def __len__(self):
        return len(self.pairs)
    
    def _get_sample(self, key, sample_idx):
        """Get and normalize a sample"""
        x = self.data[key][sample_idx].copy()
        return normalize_iq(x, self.normalize)
    
    def _augment(self, x):
        """Apply augmentations"""
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
        """
        Returns:
            noisy: (2, 128) low-SNR sample
            clean: (2, 128) high-SNR sample from SAME modulation
            label: modulation class
            noisy_snr: SNR of noisy sample
            clean_snr: SNR of clean sample
        """
        (noisy_key, noisy_sample_idx), mod = self.pairs[idx]
        
        # Get noisy sample
        noisy = self._get_sample(noisy_key, noisy_sample_idx)
        noisy_snr = noisy_key[1]
        
        # Randomly select a clean sample from the same modulation
        clean_options = self.high_snr_by_mod[mod]
        clean_key, clean_sample_idx = clean_options[np.random.randint(len(clean_options))]
        clean = self._get_sample(clean_key, clean_sample_idx)
        clean_snr = clean_key[1]
        
        if self.augment:
            # Apply same augmentation to both (for consistency)
            if np.random.random() < 0.5:
                angle = np.random.uniform(0, 2 * np.pi)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                noisy = np.stack([
                    noisy[0] * cos_a - noisy[1] * sin_a,
                    noisy[0] * sin_a + noisy[1] * cos_a
                ], axis=0)
                clean = np.stack([
                    clean[0] * cos_a - clean[1] * sin_a,
                    clean[0] * sin_a + clean[1] * cos_a
                ], axis=0)
        
        label = self.mod_to_idx[mod]
        
        return (
            torch.tensor(noisy, dtype=torch.float32),
            torch.tensor(clean, dtype=torch.float32),
            label,
            noisy_snr,
            clean_snr
        )


class RML2016aCleanDataset(Dataset):
    """
    Dataset of only high-SNR (clean) samples for DDPM training.
    
    The diffusion model learns the distribution of clean signals.
    """
    
    def __init__(self, data, indices, normalize='rms', high_snr_threshold=14, augment=True):
        self.data = data
        self.normalize = normalize
        self.augment = augment
        
        # Build label mapping
        mods = sorted(set(k[0] for k in data.keys()))
        self.mod_to_idx = {m: i for i, m in enumerate(mods)}
        self.num_classes = len(mods)
        
        # Filter to high-SNR samples only
        self.indices = [
            (key, i) for key, i in indices if key[1] >= high_snr_threshold
        ]
        
        print(f"Clean dataset: {len(self.indices)} samples (SNR >= {high_snr_threshold})")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        key, sample_idx = self.indices[idx]
        mod, snr = key
        
        x = self.data[key][sample_idx].copy()
        x = normalize_iq(x, self.normalize)
        
        if self.augment:
            if np.random.random() < 0.5:
                angle = np.random.uniform(0, 2 * np.pi)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                i, q = x[0], x[1]
                x = np.stack([i * cos_a - q * sin_a, i * sin_a + q * cos_a], axis=0)
        
        label = self.mod_to_idx[mod]
        return torch.tensor(x, dtype=torch.float32), label, snr


class RML2016aFullDataset(Dataset):
    """Full dataset for classifier evaluation (all SNR levels)"""
    
    def __init__(self, data, indices, normalize='rms'):
        self.data = data
        self.normalize = normalize
        self.indices = indices
        
        mods = sorted(set(k[0] for k in data.keys()))
        self.mod_to_idx = {m: i for i, m in enumerate(mods)}
        self.num_classes = len(mods)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        key, sample_idx = self.indices[idx]
        mod, snr = key
        
        x = self.data[key][sample_idx].copy()
        x = normalize_iq(x, self.normalize)
        
        label = self.mod_to_idx[mod]
        return torch.tensor(x, dtype=torch.float32), label, snr


def build_split_indices(data, train_ratio=0.6, val_ratio=0.2, seed=2016):
    """Split data indices into train/val/test"""
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
    data_path = "/home/tahit/Modulation/RML2016.10a_dict.pkl"
    data = load_rml2016a(data_path)
    
    train_idx, val_idx, test_idx = build_split_indices(data)
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Test DDPM dataset
    ds = RML2016aDDPMDataset(data, train_idx)
    noisy, clean, label, noisy_snr, clean_snr = ds[0]
    print(f"\nDDPM pair sample:")
    print(f"  Noisy: {noisy.shape}, SNR: {noisy_snr}")
    print(f"  Clean: {clean.shape}, SNR: {clean_snr}")
    print(f"  Label: {label}")
    
    # Test clean dataset
    ds_clean = RML2016aCleanDataset(data, train_idx)
    x, label, snr = ds_clean[0]
    print(f"\nClean sample: {x.shape}, SNR: {snr}, Label: {label}")
