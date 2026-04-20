"""
Query strategies for active learning.

Implements several sample selection strategies that determine which
unlabeled samples to query from the target model next.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


def uncertainty_sampling(
    model: nn.Module,
    X_pool: torch.Tensor,
    pool_indices: np.ndarray,
    n_query: int,
    device: str = "cpu",
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select samples with highest predictive uncertainty (entropy).

    Args:
        model:        Current substitute model
        X_pool:       (N_pool, 1, C, T) all pool data
        pool_indices: Indices of currently unqueried samples
        n_query:      Number of samples to select
        device:       Compute device
        temperature:  Softmax temperature (for DMLE acquisition prob tracking)

    Returns:
        selected_indices: (n_query,) indices into pool_indices
        acq_probs:        (n_query,) acquisition probabilities for each selected sample
    """
    model.eval()
    X_subset = X_pool[pool_indices].to(device)

    # Compute predictive entropy for all unqueried samples
    with torch.no_grad():
        logits = model(X_subset)
        probs = torch.softmax(logits / temperature, dim=1).cpu().numpy()

    # Shannon entropy: H = -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=1)

    # Select top-k by entropy
    top_k_local = np.argsort(entropy)[-n_query:]

    # Compute acquisition probabilities (softmax over entropy scores)
    entropy_exp = np.exp(entropy - entropy.max())  # numerical stability
    acq_probs_all = entropy_exp / entropy_exp.sum()
    acq_probs = acq_probs_all[top_k_local]

    selected_indices = pool_indices[top_k_local]
    return selected_indices, acq_probs


def margin_sampling(
    model: nn.Module,
    X_pool: torch.Tensor,
    pool_indices: np.ndarray,
    n_query: int,
    device: str = "cpu",
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select samples with smallest margin between top-2 class probabilities.
    Smaller margin = more uncertain.
    """
    model.eval()
    X_subset = X_pool[pool_indices].to(device)

    with torch.no_grad():
        logits = model(X_subset)
        probs = torch.softmax(logits / temperature, dim=1).cpu().numpy()

    # Sort probabilities descending per sample
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]

    # Select samples with smallest margins (most uncertain)
    top_k_local = np.argsort(margins)[:n_query]

    # Acquisition probabilities (inverse margin, normalized)
    inv_margins = 1.0 / (margins + 1e-10)
    inv_exp = np.exp(inv_margins - inv_margins.max())
    acq_probs_all = inv_exp / inv_exp.sum()
    acq_probs = acq_probs_all[top_k_local]

    selected_indices = pool_indices[top_k_local]
    return selected_indices, acq_probs


def random_sampling(
    model: nn.Module,
    X_pool: torch.Tensor,
    pool_indices: np.ndarray,
    n_query: int,
    device: str = "cpu",
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random baseline — select samples uniformly at random."""
    selected_local = np.random.choice(len(pool_indices), size=n_query, replace=False)
    acq_probs = np.ones(n_query) / len(pool_indices)  # uniform
    selected_indices = pool_indices[selected_local]
    return selected_indices, acq_probs


# Registry
STRATEGIES = {
    "uncertainty": uncertainty_sampling,
    "entropy": uncertainty_sampling,  # alias
    "margin": margin_sampling,
    "random": random_sampling,
}


def get_strategy(name: str):
    """Get query strategy function by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {list(STRATEGIES.keys())}")
    return STRATEGIES[name]
