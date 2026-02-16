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


# Canonical class names for RadioML 2018.01A (24 mods).
# Many public mirrors provide a mis-ordered `classes.txt`; use this list for stable labeling.
RML2018A_CLASSES_FIXED: List[str] = [
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-WC",
    "AM-DSB-SC",
    "FM",
    "GMSK",
    "OQPSK",
]


def load_rml2018a_hdf5(
    path: str,
    seed: int = 2016,
    train_per: int = 600,
    val_per: int = 200,
    class_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[int], List[int], List[int], List[int]]:
    """
    Load RadioML 2018.01A (DeepSig) from an HDF5 file with datasets:
      - X: (N, 1024, 2) float32
      - Y: (N, 24) one-hot {0,1}
      - Z: (N, 1) int (SNR in dB)

    Returns X in the same layout as RML2016.10a loader: (N, 2, 1024).
    """
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("h5py is required to load RML2018.01A HDF5 files") from e

    with h5py.File(path, "r") as f:
        if "X" not in f or "Y" not in f or "Z" not in f:
            raise ValueError(f"Expected datasets X/Y/Z in HDF5 file: {path}")
        X = f["X"][:]
        Y = f["Y"][:]
        Z = f["Z"][:]

    if X.ndim != 3 or X.shape[-1] != 2:
        raise ValueError(f"Expected X with shape (N, 1024, 2); got {X.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Expected Y with shape (N, C); got {Y.shape}")
    if Z.ndim == 2 and Z.shape[1] == 1:
        Z = Z[:, 0]
    if Z.ndim != 1:
        raise ValueError(f"Expected Z with shape (N,) or (N,1); got {Z.shape}")

    n = int(X.shape[0])
    if Y.shape[0] != n or Z.shape[0] != n:
        raise ValueError(f"Mismatched N across X/Y/Z: X={X.shape}, Y={Y.shape}, Z={Z.shape}")

    # Convert to our common layout (N,2,1024) and labels (N,)
    X = np.transpose(X, (0, 2, 1)).astype(np.float32, copy=False)
    y = np.argmax(Y, axis=1).astype(np.int64, copy=False)
    snr = Z.astype(np.float32, copy=False)

    mods = list(class_names) if class_names is not None else list(RML2018A_CLASSES_FIXED)
    snr_int = np.asarray(Z, dtype=np.int32)
    snrs = sorted(np.unique(snr_int).tolist())
    n_mods = int(Y.shape[1])
    if len(mods) != n_mods:
        # If the caller provided a different naming list, keep training working but warn via exception.
        raise ValueError(f"class_names length {len(mods)} != number of classes in Y ({n_mods})")

    # Balanced per-(mod,SNR) split, matching the RML2016.10a loader behavior.
    rng = np.random.RandomState(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []

    unique_snrs = np.asarray(snrs, dtype=np.int32)
    n_snrs = int(unique_snrs.shape[0])
    if n_snrs <= 0:
        raise ValueError("No SNR levels found in Z")

    # Fast-path: RML2018.01A public releases are typically stored as contiguous blocks:
    #   for class in 0..C-1:
    #     for snr in sorted(SNRs):
    #       4096 examples
    bucket_n = n // max(1, (n_mods * n_snrs))
    ordered = (bucket_n * n_mods * n_snrs) == n and bucket_n > 0
    if ordered:
        # Validate a few sentinels to avoid silently relying on wrong assumptions.
        probe = [
            (0, 0),
            (0, n_snrs - 1),
            (n_mods - 1, 0),
            (n_mods - 1, n_snrs - 1),
        ]
        for c, s_i in probe:
            start = (c * n_snrs + s_i) * bucket_n
            if int(y[start]) != int(c) or int(snr_int[start]) != int(unique_snrs[s_i]):
                ordered = False
                break

    if ordered:
        for c in range(n_mods):
            for s_i in range(n_snrs):
                start = (c * n_snrs + s_i) * bucket_n
                seg = np.arange(start, start + bucket_n, dtype=np.int64)
                if int(train_per) + int(val_per) > int(seg.shape[0]):
                    raise ValueError(
                        f"train_per+val_per exceeds samples for (class={c}, snr={int(unique_snrs[s_i])}). "
                        f"Have {int(seg.shape[0])}, need {int(train_per)+int(val_per)}."
                    )
                train_sel = rng.choice(seg, size=int(train_per), replace=False)
                remain = np.setdiff1d(seg, train_sel)
                val_sel = rng.choice(remain, size=int(val_per), replace=False)
                train_idx.extend(train_sel.tolist())
                val_idx.extend(val_sel.tolist())
    else:
        # General-path: group indices by (class, snr) without assuming storage order.
        snr_ord = np.searchsorted(unique_snrs, snr_int)
        bucket_id = y.astype(np.int64) * int(n_snrs) + snr_ord.astype(np.int64)
        order = np.argsort(bucket_id, kind="stable")
        buckets_sorted = bucket_id[order]
        boundaries = np.flatnonzero(np.diff(buckets_sorted)) + 1
        boundaries = np.concatenate(([0], boundaries, [order.shape[0]]))
        for bi in range(boundaries.shape[0] - 1):
            seg = order[boundaries[bi] : boundaries[bi + 1]]
            if int(train_per) + int(val_per) > int(seg.shape[0]):
                raise ValueError(
                    f"train_per+val_per exceeds samples for a (class,snr) bucket. "
                    f"Have {int(seg.shape[0])}, need {int(train_per)+int(val_per)}."
                )
            train_sel = rng.choice(seg, size=int(train_per), replace=False)
            remain = np.setdiff1d(seg, train_sel)
            val_sel = rng.choice(remain, size=int(val_per), replace=False)
            train_idx.extend(train_sel.tolist())
            val_idx.extend(val_sel.tolist())

    all_idx = np.arange(n, dtype=np.int64)
    used = np.union1d(np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64))
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
    Output x has shape (K, 2, L).
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
        xg = torch.stack(xs, dim=0)  # (K,2,L)
        return xg, self.y[anchor], self.snr[anchor]


class RML2016aVariableGroupedDataset(Dataset):
    """
    Returns up to K_max windows per sample from the same (class, SNR) bucket, plus a mask.

    This enables variable-K training (sample K from k_choices each __getitem__) and
    dynamic-K evaluation (progressively unmask more windows).

    Output:
      xg: (K_max, 2, L)
      y: ()
      snr: ()
      mask: (K_max,) float32 in {0,1}
    """

    def __init__(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        snr: torch.Tensor,
        indices: List[int],
        k_max: int,
        k_choices: Optional[List[int]] = None,
        normalize: str = "rms",
        eps: float = 1e-8,
        window_dropout: float = 0.0,
        aug_phase: bool = False,
        aug_shift: bool = False,
        aug_gain: float = 0.0,
        aug_cfo: float = 0.0,
    ) -> None:
        if k_max < 1:
            raise ValueError("k_max must be >= 1")
        self.X = X
        self.y = y
        self.snr = snr
        self.indices = np.asarray(indices, dtype=np.int64)
        self.k_max = int(k_max)
        self.k_choices = [int(k) for k in k_choices] if k_choices else None
        if self.k_choices is not None:
            if any(k < 1 or k > self.k_max for k in self.k_choices):
                raise ValueError("k_choices must be within [1, k_max]")
        self.normalize = normalize
        self.eps = float(eps)
        self.window_dropout = float(window_dropout)

        self.aug_phase = bool(aug_phase)
        self.aug_shift = bool(aug_shift)
        self.aug_gain = float(aug_gain)
        self.aug_cfo = float(aug_cfo)

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

        if self.k_choices is None:
            k = self.k_max
        else:
            k = int(np.random.choice(self.k_choices))

        # Optional window dropout: randomly reduce effective k further.
        if self.window_dropout and self.window_dropout > 0:
            keep = int(max(1, round(k * (1.0 - float(self.window_dropout)))))
            k = min(k, keep)

        # Sample k windows, then pad to k_max by repeating sampled windows.
        if len(bucket) >= k:
            chosen = np.random.choice(bucket, size=k, replace=False)
        else:
            chosen = np.random.choice(bucket, size=k, replace=True)

        xs = []
        for i in chosen:
            x = self._norm(self.X[int(i)])
            if self.aug_phase or self.aug_shift or (self.aug_gain and self.aug_gain > 0) or (self.aug_cfo and self.aug_cfo > 0):
                x = self._augment(x)
            xs.append(x)

        if k < self.k_max:
            pad_idx = np.random.choice(np.arange(k), size=(self.k_max - k), replace=True)
            for pi in pad_idx:
                xs.append(xs[int(pi)])
            mask = torch.zeros(self.k_max, dtype=torch.float32)
            mask[:k] = 1.0
        else:
            mask = torch.ones(self.k_max, dtype=torch.float32)

        xg = torch.stack(xs, dim=0)  # (K_max,2,L)
        return xg, self.y[anchor], self.snr[anchor], mask


def build_tensors(X: np.ndarray, y: np.ndarray, snr: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X_t = torch.from_numpy(np.ascontiguousarray(X))
    y_t = torch.from_numpy(y).long()
    snr_t = torch.from_numpy(snr).float()
    return X_t, y_t, snr_t
