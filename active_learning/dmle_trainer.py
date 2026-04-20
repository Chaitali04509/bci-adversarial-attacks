"""
Dependency-Aware Maximum Likelihood Estimation (DMLE) trainer.

Reference: Kalkanli et al., "Dependency-aware Maximum Likelihood Estimation
for Active Learning," arXiv:2503.05969, 2025.

Key insight: In active learning, samples are NOT i.i.d. — each sample's selection
depends on the current model state, which itself depends on previously selected
samples. Standard MLE ignores these dependencies, leading to biased parameter
estimates. DMLE corrects for this by reweighting the loss function using
importance weights derived from the acquisition probability of each sample.

The corrected objective is:
    L_DMLE(θ) = (1/N) Σ_i  w_i · ℓ(f_θ(x_i), y_i)

where w_i = 1 / p(x_i selected | model state at acquisition time)
and ℓ is the cross-entropy loss.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional
import numpy as np


class DMLETrainer:
    """
    DMLE training for substitute model in active learning.

    Corrects the MLE objective by applying importance weights that account
    for the non-i.i.d. nature of actively acquired samples.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        weight_clip: float = 10.0,
        correction_start_cycle: int = 2,
        ema_decay: float = 0.99,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.weight_clip = weight_clip
        self.correction_start_cycle = correction_start_cycle
        self.ema_decay = ema_decay
        self.criterion = nn.CrossEntropyLoss(reduction="none")  # per-sample loss
        self.current_cycle = 0

    def compute_importance_weights(
        self,
        acquisition_probs_history: List[np.ndarray],
        n_seed: int,
        n_total: int,
    ) -> torch.Tensor:
        """
        Compute DMLE importance weights for all labeled samples.

        For seed samples (cycle 0, selected randomly): weight = 1.
        For actively acquired samples: weight = 1 / p(x_i | θ_t) where θ_t
        is the model state at the time of acquisition.

        The weights are clipped and normalized to stabilize training.

        Args:
            acquisition_probs_history: List of arrays, one per AL cycle after seed.
                Each array has shape (n_queried_that_cycle,) with the acquisition
                probability for each selected sample.
            n_seed:  Number of seed samples
            n_total: Total labeled samples so far

        Returns:
            weights: (n_total,) importance weights
        """
        weights = torch.ones(n_total, dtype=torch.float32)

        if self.current_cycle < self.correction_start_cycle:
            return weights

        # Seed samples keep weight 1
        idx = n_seed
        for cycle_probs in acquisition_probs_history:
            for p in cycle_probs:
                # Importance weight: inverse of selection probability
                w = 1.0 / max(float(p), 1e-8)
                # Clip to prevent extreme weights
                w = min(w, self.weight_clip)
                weights[idx] = w
                idx += 1

        # Self-normalize: weights sum to n_total
        # This keeps the effective learning rate stable
        weights = weights * (n_total / weights.sum())

        # Apply EMA smoothing to reduce weight variance across cycles
        # (weights from early cycles get smoothed toward 1)
        if hasattr(self, '_prev_weights') and len(self._prev_weights) == n_total:
            weights = self.ema_decay * self._prev_weights + (1 - self.ema_decay) * weights
        self._prev_weights = weights.clone()

        return weights

    def train_cycle(
        self,
        X_labeled: torch.Tensor,
        y_labeled: torch.Tensor,
        acquisition_probs_history: List[np.ndarray],
        n_seed: int,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """
        Train the substitute model using DMLE-corrected loss.

        Args:
            X_labeled:  (N, 1, C, T) all labeled data
            y_labeled:  (N,) labels
            acquisition_probs_history: Acquisition probs from each AL cycle
            n_seed:     Size of initial seed set
            epochs:     Training epochs
            batch_size: Batch size

        Returns:
            History dict with 'loss', 'weighted_loss', and 'accuracy' per epoch.
        """
        self.model.train()
        n_total = len(y_labeled)

        # Compute importance weights
        weights = self.compute_importance_weights(
            acquisition_probs_history, n_seed, n_total
        )

        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Create dataset with sample indices for weight lookup
        indices = torch.arange(n_total)
        dataset = TensorDataset(X_labeled, y_labeled, indices)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        history = {"loss": [], "weighted_loss": [], "accuracy": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_wloss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch, idx_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                w_batch = weights[idx_batch].to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)

                # Per-sample cross-entropy loss
                per_sample_loss = self.criterion(logits, y_batch)

                # DMLE correction: weighted loss
                weighted_loss = (w_batch * per_sample_loss).mean()

                weighted_loss.backward()

                # Gradient clipping for stability (large weights can cause large grads)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += per_sample_loss.sum().item()
                epoch_wloss += weighted_loss.item() * len(y_batch)
                correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total += len(y_batch)

            scheduler.step()
            history["loss"].append(epoch_loss / total)
            history["weighted_loss"].append(epoch_wloss / total)
            history["accuracy"].append(correct / total)

        self.current_cycle += 1
        return history

    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """Evaluate model on a dataset (standard, no weighting)."""
        self.model.eval()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        ce_loss = nn.CrossEntropyLoss()

        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.model(X_batch)
                loss = ce_loss(logits, y_batch)
                total_loss += loss.item() * len(y_batch)
                correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total += len(y_batch)

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }
