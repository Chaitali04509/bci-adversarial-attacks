"""
Standard Maximum Likelihood Estimation (MLE) trainer.

Baseline approach: trains the substitute model with standard cross-entropy
loss, treating all acquired samples as i.i.d. (ignoring dependencies
introduced by active learning selection).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List
import numpy as np


class MLETrainer:
    """Standard MLE training for substitute model."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def train_cycle(
        self,
        X_labeled: torch.Tensor,
        y_labeled: torch.Tensor,
        epochs: int = 50,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """
        Train the substitute model on current labeled set for one AL cycle.

        Reinitializes optimizer each cycle (common in AL literature) to
        prevent momentum from previous cycles biasing training.

        Args:
            X_labeled: (N, 1, C, T) all labeled data so far
            y_labeled: (N,) labels
            epochs:    Training epochs this cycle
            batch_size: Batch size

        Returns:
            History dict with 'loss' and 'accuracy' per epoch.
        """
        self.model.train()
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        dataset = TensorDataset(X_labeled, y_labeled)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(y_batch)
                correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total += len(y_batch)

            scheduler.step()
            history["loss"].append(epoch_loss / total)
            history["accuracy"].append(correct / total)

        return history

    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 256,
    ) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        self.model.eval()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                total_loss += loss.item() * len(y_batch)
                correct += (logits.argmax(dim=1) == y_batch).sum().item()
                total += len(y_batch)

        return {
            "loss": total_loss / total,
            "accuracy": correct / total,
        }
