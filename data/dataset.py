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


# ---------------------------------------------------------------------------
# Synthetic EEG Generator
# ---------------------------------------------------------------------------
def generate_synthetic_eeg(
    n_samples: int,
    n_channels: int = 22,
    n_timepoints: int = 256,
    n_classes: int = 4,
    sfreq: int = 256,
    noise_std: float = 0.3,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic EEG-like data with class-discriminative patterns.

    Each class gets a distinct oscillatory signature injected into specific
    channel groups, mimicking motor imagery (mu/beta rhythms):
      - Class 0 (Left hand):  10 Hz burst in channels 0-5  (right motor cortex)
      - Class 1 (Right hand): 10 Hz burst in channels 6-11 (left motor cortex)
      - Class 2 (Both feet):  18 Hz burst in channels 12-16 (central midline)
      - Class 3 (Tongue):     24 Hz burst in channels 17-21 (distributed)

    Args:
        n_samples:    Number of trials to generate
        n_channels:   Number of EEG channels
        n_timepoints: Samples per trial
        n_classes:    Number of motor imagery classes
        sfreq:        Sampling frequency (Hz)
        noise_std:    Gaussian noise standard deviation
        seed:         Random seed

    Returns:
        X: (n_samples, n_channels, n_timepoints) EEG data
        y: (n_samples,) integer labels
    """
    rng = np.random.RandomState(seed)
    t = np.arange(n_timepoints) / sfreq  # time vector in seconds

    # Class signatures: (frequency_Hz, channel_start, channel_end, amplitude)
    class_signatures = [
        (10.0, 0, 6, 1.0),    # left hand  → right hemisphere
        (10.0, 6, 12, 1.0),   # right hand → left hemisphere
        (18.0, 12, 17, 0.8),  # feet       → central
        (24.0, 17, 22, 0.7),  # tongue     → distributed
    ]

    X = rng.randn(n_samples, n_channels, n_timepoints).astype(np.float32) * noise_std
    y = rng.randint(0, n_classes, size=n_samples).astype(np.int64)

    for i in range(n_samples):
        freq, ch_start, ch_end, amp = class_signatures[y[i]]
        # Add class-specific oscillation with random phase and slight freq jitter
        phase = rng.uniform(0, 2 * np.pi)
        freq_jitter = freq + rng.uniform(-0.5, 0.5)
        signal = amp * np.sin(2 * np.pi * freq_jitter * t + phase)
        # Apply envelope (ramp up/down) to mimic event-related (de)synchronization
        envelope = np.exp(-0.5 * ((t - 0.5) / 0.2) ** 2)
        ch_end_clamped = min(ch_end, n_channels)
        X[i, ch_start:ch_end_clamped, :] += signal * envelope

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
        # Pick only EEG channels (22 channels)
        raw.pick_types(eeg=True)
        # Bandpass filter 4-38 Hz (covers mu and beta bands)
        raw.filter(4.0, 38.0, fir_design="firwin", verbose=False)
        # Extract epochs around motor imagery events
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        # Map event codes to 0-3 (left hand, right hand, feet, tongue)
        mi_events = {k: v for k, v in event_id.items() if v in [7, 8, 9, 10]}
        if not mi_events:
            # Try alternative event codes
            mi_events = {k: v for k, v in event_id.items()
                         if k in ['769', '770', '771', '772']}
        epochs = mne.Epochs(
            raw, events, event_id=mi_events,
            tmin=0.5, tmax=0.5 + n_timepoints / raw.info['sfreq'],
            baseline=None, preload=True, verbose=False,
        )
        X = epochs.get_data().astype(np.float32)  # (n_trials, n_channels, n_times)
        # Trim/pad to exact n_timepoints
        if X.shape[2] > n_timepoints:
            X = X[:, :, :n_timepoints]
        elif X.shape[2] < n_timepoints:
            pad = np.zeros((X.shape[0], X.shape[1], n_timepoints - X.shape[2]))
            X = np.concatenate([X, pad], axis=2)
        y = epochs.events[:, -1]
        # Remap labels to 0-3
        unique_labels = sorted(np.unique(y))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        y = np.array([label_map[l] for l in y], dtype=np.int64)
        return X, y

    train_X, train_y = _load_gdf(train_file, n_timepoints)
    test_X, test_y = _load_gdf(test_file, n_timepoints)

    # Z-score normalize per channel (fit on train)
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
        """
        Args:
            X: (n_samples, n_channels, n_timepoints) EEG data
            y: (n_samples,) integer labels
        """
        self.X = torch.from_numpy(X).float().unsqueeze(1)  # (N, 1, C, T) for conv2d
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
        self._true_labels = torch.from_numpy(y).long()  # hidden from learner
        self.queried_mask = torch.zeros(len(y), dtype=torch.bool)

    def __len__(self):
        return len(self._true_labels)

    def __getitem__(self, idx):
        return self.X[idx], idx  # return index, not label

    @property
    def unqueried_indices(self) -> np.ndarray:
        return torch.where(~self.queried_mask)[0].numpy()

    def query(self, indices: np.ndarray, target_model: torch.nn.Module,
              device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query the target model for labels at given indices.
        This simulates the black-box query process.

        Returns:
            X_queried: (n, 1, C, T) tensor
            y_queried: (n,) predicted labels from target model
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
    """
    Load data based on config and return dict of datasets.

    Returns:
        {
            "train": EEGDataset,        # for training target model
            "test": EEGDataset,         # for evaluation
            "pool": PoolDataset,        # unlabeled pool for active learning
        }
    """
    if config.dataset == "synthetic":
        save_dir = os.path.join(config.data_dir, "synthetic")
        # Generate if not cached
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

        # Normalize
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
        # Use part of training data as pool
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
