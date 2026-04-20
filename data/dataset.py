"""
EEG Dataset loading and synthetic data generation.

Supports:
  - Synthetic EEG data (for quick testing/development)
  - BCI Competition IV Dataset 2a (4-class motor imagery, 9 subjects)
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, Optional, Dict



# Synthetic EEG Generator (Realistic difficulty)

def generate_synthetic_eeg(
    n_samples: int,
    n_channels: int = 22,
    n_timepoints: int = 256,
    n_classes: int = 4,
    sfreq: int = 256,
    noise_std: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic EEG-like data with REALISTIC difficulty.

    Key differences from trivial synthetic data:
      - High background noise (SNR ~ real EEG)
      - Overlapping channel groups between classes
      - Cross-class signal contamination
      - Variable amplitude and timing per trial
      - Shared background alpha rhythm across all classes

    Target accuracy should be ~70-85% (like real BCI data).
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_timepoints) / sfreq

    # Class signatures with OVERLAPPING channels and similar frequencies
    # (frequency_Hz, primary_channels, amplitude_mean, amplitude_std)
    class_signatures = [
        {"freq": 10.0, "channels": list(range(0, 10)),  "amp": 0.35, "amp_std": 0.15},   # left hand
        {"freq": 11.0, "channels": list(range(4, 14)),  "amp": 0.35, "amp_std": 0.15},   # right hand (overlaps!)
        {"freq": 17.0, "channels": list(range(10, 19)), "amp": 0.30, "amp_std": 0.12},   # feet
        {"freq": 23.0, "channels": list(range(8, 22)),  "amp": 0.25, "amp_std": 0.10},   # tongue
    ]

    # Heavy background noise
    X = rng.randn(n_samples, n_channels, n_timepoints).astype(np.float32) * noise_std
    y = rng.randint(0, n_classes, size=n_samples).astype(np.int64)

    for i in range(n_samples):
        sig = class_signatures[y[i]]

        # --- Add shared background alpha rhythm (8-12 Hz) to ALL channels ---
        alpha_freq = 10.0 + rng.uniform(-1, 1)
        alpha_amp = rng.uniform(0.1, 0.4)
        alpha_phase = rng.uniform(0, 2 * np.pi)
        background = alpha_amp * np.sin(2 * np.pi * alpha_freq * t + alpha_phase)
        X[i, :, :] += background[np.newaxis, :]

        # --- Add class-specific signal (weak, variable) ---
        amp = max(0.05, rng.normal(sig["amp"], sig["amp_std"]))
        phase = rng.uniform(0, 2 * np.pi)
        freq = sig["freq"] + rng.uniform(-1.5, 1.5)  # larger freq jitter

        signal = amp * np.sin(2 * np.pi * freq * t + phase)

        # Variable timing envelope (signal doesn't always peak at same time)
        center = rng.uniform(0.3, 0.7)
        width = rng.uniform(0.12, 0.25)
        envelope = np.exp(-0.5 * ((t - center) / width) ** 2)

        for ch in sig["channels"]:
            # Each channel gets slightly different amplitude
            ch_scale = rng.uniform(0.5, 1.0)
            X[i, ch, :] += signal * envelope * ch_scale

        # --- Cross-class contamination: leak signal into other class channels ---
        other_class = rng.randint(0, n_classes)
        if other_class != y[i]:
            leak_sig = class_signatures[other_class]
            leak_amp = rng.uniform(0.05, 0.15)  # weak leakage
            leak_freq = leak_sig["freq"] + rng.uniform(-1, 1)
            leak_signal = leak_amp * np.sin(2 * np.pi * leak_freq * t + rng.uniform(0, 2*np.pi))
            for ch in leak_sig["channels"][:3]:  # leak into first few channels only
                X[i, ch, :] += leak_signal * envelope * 0.5

        # --- Add random 50/60 Hz line noise ---
        if rng.random() > 0.5:
            line_freq = rng.choice([50.0, 60.0])
            line_amp = rng.uniform(0.05, 0.2)
            X[i, :, :] += line_amp * np.sin(2 * np.pi * line_freq * t)[np.newaxis, :]

        # --- Add random muscle artifact bursts ---
        if rng.random() > 0.7:
            artifact_ch = rng.randint(0, n_channels)
            burst_start = rng.randint(0, n_timepoints - 30)
            burst_len = rng.randint(10, 30)
            X[i, artifact_ch, burst_start:burst_start+burst_len] += rng.randn(burst_len) * 2.0

    return X, y


def save_synthetic_eeg(save_dir: str, **kwargs):
    """Generate and save synthetic EEG data to disk."""
    os.makedirs(save_dir, exist_ok=True)
    for split, n in [("train", kwargs.get("n_train", 2000)),
                     ("test", kwargs.get("n_test", 500)),
                     ("pool", kwargs.get("n_pool", 5000))]:
        seed_offset = {"train": 0, "test": 1000, "pool": 2000}[split]
        X, y = generate_synthetic_eeg(
            n_samples=n,
            n_channels=kwargs.get("n_channels", 22),
            n_timepoints=kwargs.get("n_timepoints", 256),
            n_classes=kwargs.get("n_classes", 4),
            seed=kwargs.get("seed", 42) + seed_offset,
        )
        np.save(os.path.join(save_dir, f"X_{split}.npy"), X)
        np.save(os.path.join(save_dir, f"y_{split}.npy"), y)
    print(f"Saved synthetic EEG data to {save_dir}")


# ---------------------------------------------------------------------------
# BCI Competition IV Dataset 2a Loader
# ---------------------------------------------------------------------------
def load_bci_comp_iv_2a(
    data_dir: str,
    subject_id: int = 1,
    n_timepoints: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load BCI Competition IV Dataset 2a for one subject.

    Requires the MNE library and .gdf files placed in data_dir.
    Returns preprocessed (train_X, train_y, test_X, test_y).

    Download from: https://www.bbci.de/competition/iv/
    """
    try:
        import mne
    except ImportError:
        raise ImportError("Install mne: pip install mne")

    train_file = os.path.join(data_dir, f"A{subject_id:02d}T.gdf")
    test_file = os.path.join(data_dir, f"A{subject_id:02d}E.gdf")

    if not os.path.exists(train_file):
        raise FileNotFoundError(
            f"BCI Competition IV 2a data not found at {train_file}. "
            f"Download from https://www.bbci.de/competition/iv/ and place .gdf files in {data_dir}"
        )

    def _load_gdf(filepath, n_timepoints):
        raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
        raw.pick_types(eeg=True)
        raw.filter(4.0, 38.0, fir_design="firwin", verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        mi_events = {k: v for k, v in event_id.items() if v in [7, 8, 9, 10]}
        if not mi_events:
            mi_events = {k: v for k, v in event_id.items()
                         if k in ['769', '770', '771', '772']}
        epochs = mne.Epochs(
            raw, events, event_id=mi_events,
            tmin=0.5, tmax=0.5 + n_timepoints / raw.info['sfreq'],
            baseline=None, preload=True, verbose=False,
        )
        X = epochs.get_data().astype(np.float32)
        if X.shape[2] > n_timepoints:
            X = X[:, :, :n_timepoints]
        elif X.shape[2] < n_timepoints:
            pad = np.zeros((X.shape[0], X.shape[1], n_timepoints - X.shape[2]))
            X = np.concatenate([X, pad], axis=2)
        y = epochs.events[:, -1]
        unique_labels = sorted(np.unique(y))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y = np.array([label_map[l] for l in y], dtype=np.int64)
        return X, y

    train_X, train_y = _load_gdf(train_file, n_timepoints)
    test_X, test_y = _load_gdf(test_file, n_timepoints)

    mean = train_X.mean(axis=(0, 2), keepdims=True)
    std = train_X.std(axis=(0, 2), keepdims=True) + 1e-8
    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std

    return train_X, train_y, test_X, test_y


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float().unsqueeze(1)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PoolDataset(Dataset):
    """
    Unlabeled pool for active learning.
    Stores raw data; labels are hidden and only revealed when queried.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float().unsqueeze(1)
        self._true_labels = torch.from_numpy(y).long()
        self.queried_mask = torch.zeros(len(y), dtype=torch.bool)

    def __len__(self):
        return len(self._true_labels)

    def __getitem__(self, idx):
        return self.X[idx], idx

    @property
    def unqueried_indices(self) -> np.ndarray:
        return torch.where(~self.queried_mask)[0].numpy()

    def query(self, indices: np.ndarray, target_model: torch.nn.Module,
              device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query the target model for labels at given indices.
        This simulates the black-box query process.
        """
        target_model.eval()
        X_batch = self.X[indices].to(device)
        with torch.no_grad():
            logits = target_model(X_batch)
            y_pred = logits.argmax(dim=1).cpu()
        self.queried_mask[indices] = True
        return self.X[indices], y_pred


# ---------------------------------------------------------------------------
# Convenience loaders
# ---------------------------------------------------------------------------
def load_data(config) -> Dict[str, Dataset]:
    """Load data based on config and return dict of datasets."""
    if config.dataset == "synthetic":
        save_dir = os.path.join(config.data_dir, "synthetic")
        if not os.path.exists(os.path.join(save_dir, "X_train.npy")):
            save_synthetic_eeg(
                save_dir,
                n_train=config.synthetic_n_train,
                n_test=config.synthetic_n_test,
                n_pool=config.synthetic_n_pool,
                n_channels=config.n_channels,
                n_timepoints=config.n_timepoints,
                n_classes=config.n_classes,
            )
        X_train = np.load(os.path.join(save_dir, "X_train.npy"))
        y_train = np.load(os.path.join(save_dir, "y_train.npy"))
        X_test = np.load(os.path.join(save_dir, "X_test.npy"))
        y_test = np.load(os.path.join(save_dir, "y_test.npy"))
        X_pool = np.load(os.path.join(save_dir, "X_pool.npy"))
        y_pool = np.load(os.path.join(save_dir, "y_pool.npy"))

        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std
        X_pool = (X_pool - mean) / std

    elif config.dataset == "bci_comp_iv_2a":
        gdf_dir = os.path.join(config.data_dir, "bci_comp_iv_2a")
        X_train, y_train, X_test, y_test = load_bci_comp_iv_2a(
            gdf_dir, subject_id=config.subject_ids[0],
            n_timepoints=config.n_timepoints,
        )
        n_keep = min(200, len(X_train) // 2)
        X_pool = X_train[n_keep:]
        y_pool = y_train[n_keep:]
        X_train = X_train[:n_keep]
        y_train = y_train[:n_keep]
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    return {
        "train": EEGDataset(X_train, y_train),
        "test": EEGDataset(X_test, y_test),
        "pool": PoolDataset(X_pool, y_pool),
    }
