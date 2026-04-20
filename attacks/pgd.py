"""
Projected Gradient Descent (PGD) adversarial attack.

Reference: Madry et al., "Towards Deep Learning Models Resistant to
Adversarial Attacks," ICLR 2018.

PGD is an iterative version of FGSM that takes multiple smaller steps
and projects back into the ε-ball after each step. Generally stronger
than FGSM.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def pgd_attack(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    alpha: float = 0.01,
    iterations: int = 40,
    random_start: bool = True,
    device: str = "cpu",
    targeted: bool = False,
    target_class: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate PGD adversarial examples.

    Algorithm:
        1. (Optional) Start from random point within ε-ball
        2. For each iteration:
           a. Compute gradient of loss w.r.t. input
           b. Take step: x += α · sign(grad)
           c. Project back into ε-ball: clip(x_adv - x_orig, -ε, ε) + x_orig

    Args:
        model:        Substitute model
        X:            (N, 1, C, T) clean inputs
        y:            (N,) true labels
        epsilon:      Max perturbation (L∞ norm)
        alpha:        Step size per iteration
        iterations:   Number of PGD steps
        random_start: Initialize with random perturbation
        device:       Compute device
        targeted:     If True, push toward target_class
        target_class: Target class for targeted attack

    Returns:
        X_adv:         (N, 1, C, T) adversarial examples
        perturbations: (N, 1, C, T) added perturbations
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    X_orig = X.clone().detach().to(device)

    if random_start:
        X_adv = X_orig + torch.empty_like(X_orig).uniform_(-epsilon, epsilon)
    else:
        X_adv = X_orig.clone()

    for _ in range(iterations):
        X_adv = X_adv.detach().requires_grad_(True)

        logits = model(X_adv)

        if targeted:
            assert target_class is not None
            target_labels = torch.full_like(y.to(device), target_class)
            loss = criterion(logits, target_labels)
        else:
            loss = criterion(logits, y.to(device))

        loss.backward()

        with torch.no_grad():
            grad_sign = X_adv.grad.sign()

            if targeted:
                X_adv = X_adv - alpha * grad_sign
            else:
                X_adv = X_adv + alpha * grad_sign

            # Project back into ε-ball (L∞)
            perturbation = torch.clamp(X_adv - X_orig, -epsilon, epsilon)
            X_adv = X_orig + perturbation

    perturbations = (X_adv - X_orig).detach().cpu()
    return X_adv.detach().cpu(), perturbations


def batch_pgd_attack(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    alpha: float = 0.01,
    iterations: int = 40,
    random_start: bool = True,
    device: str = "cpu",
    batch_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PGD attack in batches for memory efficiency."""
    all_adv = []
    all_pert = []

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        X_batch = X[start:end]
        y_batch = y[start:end]
        X_adv_batch, pert_batch = pgd_attack(
            model, X_batch, y_batch,
            epsilon=epsilon, alpha=alpha, iterations=iterations,
            random_start=random_start, device=device,
        )
        all_adv.append(X_adv_batch)
        all_pert.append(pert_batch)

    return torch.cat(all_adv, dim=0), torch.cat(all_pert, dim=0)
