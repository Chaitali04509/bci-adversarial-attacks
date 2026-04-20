"""
Evaluation metrics for tracking experiment progress.
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
from sklearn.metrics import confusion_matrix, classification_report


def compute_agreement(
    model_a: nn.Module,
    model_b: nn.Module,
    X: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 256,
) -> float:
    """
    Compute prediction agreement between two models.
    High agreement between substitute and target = better substitute.

    Args:
        model_a, model_b: Two models to compare
        X: (N, 1, C, T) input data
        device: Compute device

    Returns:
        Agreement rate (fraction of samples with same prediction).
    """
    model_a.eval()
    model_b.eval()
    agree = 0
    total = 0

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        X_batch = X[start:end].to(device)
        with torch.no_grad():
            pred_a = model_a(X_batch).argmax(dim=1)
            pred_b = model_b(X_batch).argmax(dim=1)
        agree += (pred_a == pred_b).sum().item()
        total += len(X_batch)

    return agree / total


def compute_substitute_quality(
    substitute: nn.Module,
    target: nn.Module,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Comprehensive substitute model quality metrics.

    Returns:
        Dict with: agreement, sub_accuracy, target_accuracy, fidelity
    """
    substitute.eval()
    target.eval()

    sub_preds = []
    target_preds = []

    for start in range(0, len(X_test), 256):
        end = min(start + 256, len(X_test))
        X_batch = X_test[start:end].to(device)
        with torch.no_grad():
            sub_preds.append(substitute(X_batch).argmax(dim=1).cpu())
            target_preds.append(target(X_batch).argmax(dim=1).cpu())

    sub_preds = torch.cat(sub_preds)
    target_preds = torch.cat(target_preds)

    return {
        "substitute_accuracy": (sub_preds == y_test).float().mean().item(),
        "target_accuracy": (target_preds == y_test).float().mean().item(),
        "agreement": (sub_preds == target_preds).float().mean().item(),
        # Fidelity: how well substitute mimics target (agreement on target's predictions)
        "fidelity": (sub_preds == target_preds).float().mean().item(),
    }


def summarize_results(
    mle_history: List[Dict],
    dmle_history: List[Dict],
    mle_attacks: List[Dict],
    dmle_attacks: List[Dict],
) -> Dict:
    """
    Summarize experiment results across AL cycles and attack evaluations.

    Args:
        mle_history:  List of per-cycle metrics for MLE
        dmle_history: List of per-cycle metrics for DMLE
        mle_attacks:  List of per-cycle attack results for MLE
        dmle_attacks: List of per-cycle attack results for DMLE

    Returns:
        Summary dict for JSON export.
    """
    def _extract(history, key):
        return [h.get(key, None) for h in history]

    def _attack_extract(attacks, attack_name, metric):
        return [a.get(attack_name, {}).get(metric, None) for a in attacks]

    summary = {
        "mle": {
            "substitute_accuracy": _extract(mle_history, "substitute_accuracy"),
            "agreement": _extract(mle_history, "agreement"),
            "fgsm_success": _attack_extract(mle_attacks, "fgsm", "attack_success_rate"),
            "pgd_success": _attack_extract(mle_attacks, "pgd", "attack_success_rate"),
        },
        "dmle": {
            "substitute_accuracy": _extract(dmle_history, "substitute_accuracy"),
            "agreement": _extract(dmle_history, "agreement"),
            "fgsm_success": _attack_extract(dmle_attacks, "fgsm", "attack_success_rate"),
            "pgd_success": _attack_extract(dmle_attacks, "pgd", "attack_success_rate"),
        },
    }

    # Compute improvement metrics
    for metric in ["fgsm_success", "pgd_success"]:
        mle_vals = [v for v in summary["mle"][metric] if v is not None]
        dmle_vals = [v for v in summary["dmle"][metric] if v is not None]
        if mle_vals and dmle_vals:
            summary[f"final_{metric}_improvement"] = dmle_vals[-1] - mle_vals[-1]
            # Area under curve (higher = better across all cycles)
            summary[f"auc_{metric}_mle"] = np.trapz(mle_vals)
            summary[f"auc_{metric}_dmle"] = np.trapz(dmle_vals)

    return summary
