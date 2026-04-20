"""
Substitute model for black-box adversarial attacks.

Wraps an EEGNet variant and manages the labeled dataset that grows
through active learning queries to the target model.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional


class SubstituteModel:
    """
    Manages a substitute model and its growing labeled dataset.
    Used in black-box attacks where we query a target model to build
    a surrogate, then craft adversarial examples on the surrogate.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        # Accumulated labeled data from queries
        self.X_labeled = None  # (N, 1, C, T)
        self.y_labeled = None  # (N,)
        # Track acquisition probabilities for DMLE
        self.acquisition_probs = []  # list of arrays, one per query cycle

    def add_labeled_data(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        acq_probs: Optional[np.ndarray] = None,
    ):
        """
        Add newly queried samples to the labeled set.

        Args:
            X: (n, 1, C, T) input tensors
            y: (n,) labels from target model
            acq_probs: (n,) probability of selecting each sample (for DMLE)
        """
        if self.X_labeled is None:
            self.X_labeled = X.cpu()
            self.y_labeled = y.cpu()
        else:
            self.X_labeled = torch.cat([self.X_labeled, X.cpu()], dim=0)
            self.y_labeled = torch.cat([self.y_labeled, y.cpu()], dim=0)

        if acq_probs is not None:
            self.acquisition_probs.append(acq_probs)

    def get_dataloader(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Get DataLoader for current labeled set."""
        dataset = TensorDataset(self.X_labeled, self.y_labeled)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_importance_weights(self) -> Optional[torch.Tensor]:
        """
        Compute DMLE importance weights for all labeled samples.

        For seed samples (no acquisition probability), weight = 1.
        For actively acquired samples, weight = 1 / p(x_i was selected).
        """
        if not self.acquisition_probs:
            return None

        n_total = len(self.y_labeled)
        weights = torch.ones(n_total)

        # Seed samples have uniform weight
        offset = n_total - sum(len(p) for p in self.acquisition_probs)
        # Seeds: indices [0, offset)

        # Acquired samples: apply importance weights
        idx = offset
        for probs in self.acquisition_probs:
            for p in probs:
                weights[idx] = 1.0 / max(p, 1e-8)
                idx += 1

        # Normalize weights so they sum to n_total
        weights = weights * (n_total / weights.sum())
        return weights

    @property
    def n_labeled(self) -> int:
        return 0 if self.y_labeled is None else len(self.y_labeled)
