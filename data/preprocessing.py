"""
EEG signal preprocessing utilities.
"""
import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    data: np.ndarray,
    low_freq: float = 4.0,
    high_freq: float = 38.0,
    sfreq: int = 256,
    order: int = 5,
) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to EEG data.

    Args:
        data: (n_channels, n_timepoints) or (n_samples, n_channels, n_timepoints)
        low_freq:  Lower cutoff frequency (Hz)
        high_freq: Upper cutoff frequency (Hz)
        sfreq:     Sampling frequency (Hz)
        order:     Filter order

    Returns:
        Filtered data with same shape as input.
    """
    nyq = 0.5 * sfreq
    b, a = butter(order, [low_freq / nyq, high_freq / nyq], btype="band")
    if data.ndim == 2:
        return filtfilt(b, a, data, axis=-1).astype(np.float32)
    elif data.ndim == 3:
        return np.array([filtfilt(b, a, trial, axis=-1) for trial in data],
                        dtype=np.float32)
    else:
        raise ValueError(f"Expected 2D or 3D input, got {data.ndim}D")


def zscore_normalize(
    X: np.ndarray,
    mean: np.ndarray = None,
    std: np.ndarray = None,
) -> tuple:
    """
    Z-score normalize EEG data per channel.

    Args:
        X:    (n_samples, n_channels, n_timepoints)
        mean: Precomputed channel means (for normalizing test data with train stats)
        std:  Precomputed channel stds

    Returns:
        X_norm, mean, std
    """
    if mean is None:
        mean = X.mean(axis=(0, 2), keepdims=True)
    if std is None:
        std = X.std(axis=(0, 2), keepdims=True) + 1e-8
    return (X - mean) / std, mean, std


def exponential_moving_standardize(
    X: np.ndarray,
    factor_new: float = 0.001,
    init_block_size: int = 50,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Exponential moving standardization (Schirrmeister et al. 2017).
    Useful for non-stationary EEG signals.

    Args:
        X: (n_channels, n_timepoints)
        factor_new: Weight for new data point
        init_block_size: Number of initial samples to compute starting stats
        eps: Numerical stability

    Returns:
        Standardized data
    """
    X = X.copy()
    for ch in range(X.shape[0]):
        x = X[ch]
        mean = x[:init_block_size].mean()
        var = x[:init_block_size].var()
        for t in range(len(x)):
            mean = (1 - factor_new) * mean + factor_new * x[t]
            var = (1 - factor_new) * var + factor_new * (x[t] - mean) ** 2
            x[t] = (x[t] - mean) / max(np.sqrt(var), eps)
        X[ch] = x
    return X
