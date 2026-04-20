"""
Adversarial attack evaluation.

Measures attack success rate, transferability (substitute → target),
and perturbation statistics.
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict

from attacks.fgsm import batch_fgsm_attack
from attacks.pgd import batch_pgd_attack


def evaluate_attack(
    target_model: nn.Module,
    X_clean: torch.Tensor,
    y_true: torch.Tensor,
    X_adv: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 256,
) -> Dict[str, float]:
    """
    Evaluate adversarial examples against the target model.

    Metrics:
        - attack_success_rate: fraction of correctly-classified clean samples
          that are misclassified after perturbation
        - fooling_rate: fraction of ALL samples where prediction changes
        - clean_accuracy: target model accuracy on clean data
        - adversarial_accuracy: target model accuracy on adversarial data
        - mean_perturbation: average L∞ perturbation norm

    Args:
        target_model: The black-box target model
        X_clean:      (N, 1, C, T) clean test inputs
        y_true:       (N,) ground truth labels
        X_adv:        (N, 1, C, T) adversarial inputs
        device:       Compute device

    Returns:
        Dict of evaluation metrics.
    """
    target_model.eval()

    # Get predictions on clean and adversarial inputs
    clean_preds = []
    adv_preds = []

    for start in range(0, len(X_clean), batch_size):
        end = min(start + batch_size, len(X_clean))

        with torch.no_grad():
            clean_batch = X_clean[start:end].to(device)
            adv_batch = X_adv[start:end].to(device)

            clean_logits = target_model(clean_batch)
            adv_logits = target_model(adv_batch)

            clean_preds.append(clean_logits.argmax(dim=1).cpu())
            adv_preds.append(adv_logits.argmax(dim=1).cpu())

    clean_preds = torch.cat(clean_preds)
    adv_preds = torch.cat(adv_preds)

    # Clean accuracy
    clean_correct = (clean_preds == y_true).float()
    clean_accuracy = clean_correct.mean().item()

    # Adversarial accuracy
    adv_accuracy = (adv_preds == y_true).float().mean().item()

    # Attack success rate: among samples correctly classified on clean input,
    # what fraction is misclassified on adversarial input?
    correctly_classified = clean_preds == y_true
    if correctly_classified.sum() > 0:
        attack_success = (
            (adv_preds[correctly_classified] != y_true[correctly_classified])
            .float().mean().item()
        )
    else:
        attack_success = 0.0

    # Fooling rate: any prediction change (regardless of correctness)
    fooling_rate = (clean_preds != adv_preds).float().mean().item()

    # Perturbation stats
    perturbation = (X_adv - X_clean).abs()
    mean_linf = perturbation.view(len(X_clean), -1).max(dim=1)[0].mean().item()
    mean_l2 = perturbation.view(len(X_clean), -1).norm(dim=1, p=2).mean().item()

    return {
        "clean_accuracy": clean_accuracy,
        "adversarial_accuracy": adv_accuracy,
        "attack_success_rate": attack_success,
        "fooling_rate": fooling_rate,
        "mean_linf_perturbation": mean_linf,
        "mean_l2_perturbation": mean_l2,
    }


def run_full_attack_evaluation(
    substitute_model: nn.Module,
    target_model: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    config,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Run all configured attacks and evaluate transferability.

    Args:
        substitute_model: Trained substitute model (attacks are crafted here)
        target_model:     Black-box target model (attacks are evaluated here)
        X_test:           (N, 1, C, T) test data
        y_test:           (N,) test labels
        config:           AttackConfig
        device:           Compute device

    Returns:
        Dict mapping attack_name → metrics dict
    """
    n = min(config.n_attack_samples, len(X_test))
    X_test = X_test[:n]
    y_test = y_test[:n]

    results = {}

    if "fgsm" in config.methods:
        X_adv_fgsm, _ = batch_fgsm_attack(
            substitute_model, X_test, y_test,
            epsilon=config.fgsm_epsilon, device=device,
        )
        results["fgsm"] = evaluate_attack(
            target_model, X_test, y_test, X_adv_fgsm, device=device,
        )

    if "pgd" in config.methods:
        X_adv_pgd, _ = batch_pgd_attack(
            substitute_model, X_test, y_test,
            epsilon=config.pgd_epsilon, alpha=config.pgd_alpha,
            iterations=config.pgd_iterations,
            random_start=config.pgd_random_start, device=device,
        )
        results["pgd"] = evaluate_attack(
            target_model, X_test, y_test, X_adv_pgd, device=device,
        )

    return results
