import math
import pickle
import warnings
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _sorted_unique(items: Iterable) -> List:
    return sorted(list(set(items)))


def load_rml2016a(
    path: str,
    seed: int = 2016,
    train_per: int = 600,
    val_per: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[int], List[int], List[int], List[int]]:
    with open(path, "rb") as f:
        with warnings.catch_warnings():
            warning_cls = getattr(np, "VisibleDeprecationWarning", Warning)
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed",
                category=warning_cls,
            )
            Xd = pickle.load(f, encoding="iso-8859-1")
    mods = _sorted_unique([k[0] for k in Xd.keys()])
    snrs = _sorted_unique([k[1] for k in Xd.keys()])
    mod_to_idx = {m: i for i, m in enumerate(mods)}

    rng = np.random.RandomState(seed)
    X_list = []
    y_list = []
    snr_list = []
    train_idx = []
    val_idx = []
    cursor = 0

    for mod in mods:
        mod_idx = mod_to_idx[mod]
        for snr in snrs:
            data = Xd[(mod, snr)]
            num = data.shape[0]
            if train_per + val_per > num:
                raise ValueError(
                    f"train_per+val_per exceeds samples for ({mod}, {snr})."
                )
            X_list.append(data)
            y_list.extend([mod_idx] * num)
            snr_list.extend([snr] * num)

            idx = np.arange(cursor, cursor + num)
            train_sel = rng.choice(idx, size=train_per, replace=False)
            remain = np.setdiff1d(idx, train_sel)
            val_sel = rng.choice(remain, size=val_per, replace=False)
            train_idx.extend(train_sel.tolist())
            val_idx.extend(val_sel.tolist())
            cursor += num

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    snr = np.array(snr_list, dtype=np.float32)

    n_examples = X.shape[0]
    all_idx = np.arange(n_examples)
    used = np.union1d(train_idx, val_idx)
    test_idx = np.setdiff1d(all_idx, used).tolist()

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return X, y, snr, mods, snrs, train_idx, val_idx, test_idx


def parse_snrs(value: Optional[str]) -> Optional[List[float]]:
    if value is None or value.strip() == "":
        return None
    return [float(v.strip()) for v in value.split(",") if v.strip() != ""]


def filter_indices_by_snrs(
    indices: List[int],
    snr_values: np.ndarray,
    allowed_snrs: Optional[List[float]],
) -> List[int]:
    if allowed_snrs is None:
        return indices
    mask = np.isin(snr_values[indices], np.array(allowed_snrs, dtype=snr_values.dtype))
    indices = np.asarray(indices)
    return indices[mask].tolist()


class RML2016aDataset(Dataset):
    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        snr: torch.Tensor,
        indices: List[int],
        normalize: str = "rms",
        eps: float = 1e-8,
        aug_phase: bool = False,
        aug_shift: bool = False,
        aug_gain: float = 0.0,
        aug_cfo: float = 0.0,
    ) -> None:
        self.X = X
        self.y = y
        self.snr = snr
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalize = normalize
        self.eps = float(eps)
        self.aug_phase = bool(aug_phase)
        self.aug_shift = bool(aug_shift)
        self.aug_gain = float(aug_gain)
        self.aug_cfo = float(aug_cfo)

    def __len__(self) -> int:
        return self.indices.shape[0]

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize == "none":
            return x
        if self.normalize == "rms":
            rms = torch.sqrt(torch.mean(x * x) + self.eps)
            return x / rms
        raise ValueError(f"Unknown normalize mode: {self.normalize}")

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (2, L) IQ tensor.
        Augmentations are label-preserving invariances for AMC.
        """
        # Circular time shift (start-point invariance).
        if self.aug_shift:
            shift = int(torch.randint(0, x.shape[-1], (1,)).item())
            x = torch.roll(x, shifts=shift, dims=-1)

        # Random phase rotation (carrier phase invariance).
        if self.aug_phase:
            theta = float(torch.rand(1).item()) * (2.0 * math.pi)
            c = math.cos(theta)
            s = math.sin(theta)
            i = x[0]
            q = x[1]
            x = torch.stack([c * i - s * q, s * i + c * q], dim=0)

        # Small CFO (cycles per sample). Conservative by default.
        if self.aug_cfo and self.aug_cfo > 0:
            f = (torch.rand(1).item() * 2.0 - 1.0) * float(self.aug_cfo)
            n = torch.arange(x.shape[-1], device=x.device, dtype=torch.float32)
            ang = 2.0 * math.pi * f * n
            c = torch.cos(ang)
            s = torch.sin(ang)
            i = x[0]
            q = x[1]
            x = torch.stack([c * i - s * q, s * i + c * q], dim=0)

        # Gain jitter. With RMS normalization this is mostly redundant, but can help if normalize=none.
        if self.aug_gain and self.aug_gain > 0:
            g = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * float(self.aug_gain)
            x = x * float(g)

        return x

    def __getitem__(self, idx: int):
        i = int(self.indices[idx])
        x = self._norm(self.X[i])
        if self.aug_phase or self.aug_shift or (self.aug_gain and self.aug_gain > 0) or (self.aug_cfo and self.aug_cfo > 0):
            x = self._augment(x)
        return x, self.y[i], self.snr[i]


class RML2016aGroupedDataset(Dataset):
    """
    Returns K random windows from the same (class, SNR) bucket within the provided indices.
    Output x has shape (K, 2, 128).
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        snr: torch.Tensor,
        indices: List[int],
        group_k: int,
        normalize: str = "rms",
        eps: float = 1e-8,
        aug_phase: bool = False,
        aug_shift: bool = False,
        aug_gain: float = 0.0,
        aug_cfo: float = 0.0,
    ) -> None:
        if group_k < 1:
            raise ValueError("group_k must be >= 1")
        self.X = X
        self.y = y
        self.snr = snr
        self.indices = np.asarray(indices, dtype=np.int64)
        self.group_k = int(group_k)
        self.normalize = normalize
        self.eps = float(eps)
        self.aug_phase = bool(aug_phase)
        self.aug_shift = bool(aug_shift)
        self.aug_gain = float(aug_gain)
        self.aug_cfo = float(aug_cfo)

        # Build bucket -> list of indices (within this split).
        y_np = self.y[self.indices].cpu().numpy()
        snr_np = self.snr[self.indices].cpu().numpy().astype(np.int32)
        self.bucket_map = {}
        for local_i, (yy, ss) in enumerate(zip(y_np, snr_np)):
            key = (int(yy), int(ss))
            self.bucket_map.setdefault(key, []).append(int(self.indices[local_i]))

    def __len__(self) -> int:
        return self.indices.shape[0]

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize == "none":
            return x
        if self.normalize == "rms":
            rms = torch.sqrt(torch.mean(x * x) + self.eps)
            return x / rms
        raise ValueError(f"Unknown normalize mode: {self.normalize}")

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # Same as RML2016aDataset._augment, duplicated to keep Dataset objects self-contained/picklable.
        if self.aug_shift:
            shift = int(torch.randint(0, x.shape[-1], (1,)).item())
            x = torch.roll(x, shifts=shift, dims=-1)
        if self.aug_phase:
            theta = float(torch.rand(1).item()) * (2.0 * math.pi)
            c = math.cos(theta)
            s = math.sin(theta)
            i = x[0]
            q = x[1]
            x = torch.stack([c * i - s * q, s * i + c * q], dim=0)
        if self.aug_cfo and self.aug_cfo > 0:
            f = (torch.rand(1).item() * 2.0 - 1.0) * float(self.aug_cfo)
            n = torch.arange(x.shape[-1], device=x.device, dtype=torch.float32)
            ang = 2.0 * math.pi * f * n
            c = torch.cos(ang)
            s = torch.sin(ang)
            i = x[0]
            q = x[1]
            x = torch.stack([c * i - s * q, s * i + c * q], dim=0)
        if self.aug_gain and self.aug_gain > 0:
            g = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * float(self.aug_gain)
            x = x * float(g)
        return x

    def __getitem__(self, idx: int):
        anchor = int(self.indices[int(idx)])
        yy = int(self.y[anchor].item())
        ss = int(self.snr[anchor].item())
        bucket = self.bucket_map[(yy, ss)]

        if self.group_k == 1:
            x = self._norm(self.X[anchor])
            if self.aug_phase or self.aug_shift or (self.aug_gain and self.aug_gain > 0) or (self.aug_cfo and self.aug_cfo > 0):
                x = self._augment(x)
            return x.unsqueeze(0), self.y[anchor], self.snr[anchor]

        # Sample K indices from bucket (with replacement if needed).
        if len(bucket) >= self.group_k:
            chosen = np.random.choice(bucket, size=self.group_k, replace=False)
        else:
            chosen = np.random.choice(bucket, size=self.group_k, replace=True)

        xs = []
        for i in chosen:
            x = self._norm(self.X[int(i)])
            if self.aug_phase or self.aug_shift or (self.aug_gain and self.aug_gain > 0) or (self.aug_cfo and self.aug_cfo > 0):
                x = self._augment(x)
            xs.append(x)
        xg = torch.stack(xs, dim=0)  # (K,2,128)
        return xg, self.y[anchor], self.snr[anchor]


def build_tensors(X: np.ndarray, y: np.ndarray, snr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X_t = torch.from_numpy(np.ascontiguousarray(X))
    y_t = torch.from_numpy(y).long()
    snr_t = torch.from_numpy(snr).float()
    return X_t, y_t, snr_t
