"""
Fast Gradient Sign Method (FGSM) adversarial attack.

Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples,"
ICLR 2015.

In our black-box setting, FGSM is applied to the SUBSTITUTE model,
and the resulting adversarial examples are evaluated on the TARGET model
(transfer-based attack).
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def fgsm_attack(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    device: str = "cpu",
    targeted: bool = False,
    target_class: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate FGSM adversarial examples.

    x_adv = x + ε · sign(∇_x L(θ, x, y))    [untargeted]
    x_adv = x - ε · sign(∇_x L(θ, x, y_t))  [targeted]

    Args:
        model:        Substitute model to compute gradients on
        X:            (N, 1, C, T) clean input samples
        y:            (N,) true labels
        epsilon:      Perturbation magnitude
        device:       Compute device
        targeted:     If True, push toward target_class
        target_class: Target class for targeted attack

    Returns:
        X_adv:         (N, 1, C, T) adversarial examples
        perturbations: (N, 1, C, T) the added perturbations
    """
    model.eval()  # Keep eval mode for BN, but we need gradients w.r.t. input
    criterion = nn.CrossEntropyLoss()

    X = X.clone().detach().to(device).requires_grad_(True)
    y = y.to(device)

    logits = model(X)

    if targeted:
        assert target_class is not None
        target_labels = torch.full_like(y, target_class)
        loss = criterion(logits, target_labels)
    else:
        loss = criterion(logits, y)

    loss.backward()

    # FGSM perturbation
    grad_sign = X.grad.sign()

    if targeted:
        X_adv = X.detach() - epsilon * grad_sign  # minimize loss toward target
    else:
        X_adv = X.detach() + epsilon * grad_sign  # maximize loss

    perturbations = X_adv - X.detach()
    return X_adv.detach().cpu(), perturbations.detach().cpu()


def batch_fgsm_attack(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    device: str = "cpu",
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    FGSM attack in batches (for memory efficiency).

    Returns:
        X_adv: (N, 1, C, T)
        perturbations: (N, 1, C, T)
    """
    all_adv = []
    all_pert = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        X_batch = X[start:end]
        y_batch = y[start:end]
        X_adv_batch, pert_batch = fgsm_attack(model, X_batch, y_batch, epsilon, device)
        all_adv.append(X_adv_batch)
        all_pert.append(pert_batch)

    return torch.cat(all_adv, dim=0), torch.cat(all_pert, dim=0)
