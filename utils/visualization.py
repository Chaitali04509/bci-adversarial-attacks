"""
Visualization utilities for experiment results.

Generates publication-quality plots comparing MLE vs DMLE across
active learning cycles and attack evaluations.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from typing import Dict, List, Optional


# Style configuration
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

# Color palette
MLE_COLOR = "#2196F3"   # blue
DMLE_COLOR = "#F44336"  # red
RANDOM_COLOR = "#9E9E9E"  # gray


def plot_accuracy_curves(
    mle_accuracies: List[float],
    dmle_accuracies: List[float],
    query_counts: List[int],
    save_path: str,
    title: str = "Substitute Model Accuracy vs. Queries",
    random_accuracies: Optional[List[float]] = None,
):
    """
    Plot substitute model accuracy over active learning cycles.

    Args:
        mle_accuracies:  Accuracy per cycle for MLE
        dmle_accuracies: Accuracy per cycle for DMLE
        query_counts:    Cumulative queries at each cycle
        save_path:       Where to save the figure
        title:           Plot title
        random_accuracies: Optional random baseline accuracies
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(query_counts, mle_accuracies, "o-", color=MLE_COLOR,
            label="MLE (baseline)", linewidth=2, markersize=5)
    ax.plot(query_counts, dmle_accuracies, "s-", color=DMLE_COLOR,
            label="DMLE (ours)", linewidth=2, markersize=5)

    if random_accuracies is not None:
        ax.plot(query_counts, random_accuracies, "^--", color=RANDOM_COLOR,
                label="Random sampling", linewidth=1.5, markersize=4)

    ax.set_xlabel("Total Queries to Target Model")
    ax.set_ylabel("Substitute Model Accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_attack_success(
    mle_success: Dict[str, List[float]],
    dmle_success: Dict[str, List[float]],
    query_counts: List[int],
    save_path: str,
    title: str = "Attack Success Rate vs. Queries",
):
    """
    Plot attack success rates for MLE vs DMLE, one subplot per attack method.

    Args:
        mle_success:  {attack_name: [success_rate_per_cycle]}
        dmle_success: {attack_name: [success_rate_per_cycle]}
        query_counts: Cumulative queries at each cycle
        save_path:    Where to save
    """
    attack_names = sorted(set(list(mle_success.keys()) + list(dmle_success.keys())))
    n_attacks = len(attack_names)

    fig, axes = plt.subplots(1, n_attacks, figsize=(6 * n_attacks, 5), squeeze=False)

    for i, atk_name in enumerate(attack_names):
        ax = axes[0, i]

        if atk_name in mle_success:
            vals = mle_success[atk_name]
            ax.plot(query_counts[:len(vals)], vals, "o-", color=MLE_COLOR,
                    label="MLE", linewidth=2, markersize=5)
        if atk_name in dmle_success:
            vals = dmle_success[atk_name]
            ax.plot(query_counts[:len(vals)], vals, "s-", color=DMLE_COLOR,
                    label="DMLE", linewidth=2, markersize=5)

        ax.set_xlabel("Total Queries")
        ax.set_ylabel("Attack Success Rate")
        ax.set_title(f"{atk_name.upper()} Attack")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    fig.suptitle(title, fontsize=15, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_agreement_curves(
    mle_agreement: List[float],
    dmle_agreement: List[float],
    query_counts: List[int],
    save_path: str,
):
    """Plot substitute-target prediction agreement over AL cycles."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(query_counts, mle_agreement, "o-", color=MLE_COLOR,
            label="MLE", linewidth=2, markersize=5)
    ax.plot(query_counts, dmle_agreement, "s-", color=DMLE_COLOR,
            label="DMLE", linewidth=2, markersize=5)

    ax.set_xlabel("Total Queries to Target Model")
    ax.set_ylabel("Substitute–Target Agreement")
    ax.set_title("Substitute Model Fidelity")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_query_efficiency(
    mle_success: List[float],
    dmle_success: List[float],
    query_counts: List[int],
    thresholds: List[float],
    save_path: str,
    attack_name: str = "FGSM",
):
    """
    Bar chart: queries needed to reach different attack success thresholds.

    Args:
        mle_success:  Attack success per cycle for MLE
        dmle_success: Attack success per cycle for DMLE
        query_counts: Cumulative queries
        thresholds:   Success rate thresholds to measure (e.g., [0.3, 0.5, 0.7])
        save_path:    Where to save
    """
    def _queries_to_threshold(success_rates, queries, threshold):
        for s, q in zip(success_rates, queries):
            if s >= threshold:
                return q
        return queries[-1]  # never reached

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(thresholds))
    width = 0.35

    mle_queries = [_queries_to_threshold(mle_success, query_counts, t) for t in thresholds]
    dmle_queries = [_queries_to_threshold(dmle_success, query_counts, t) for t in thresholds]

    bars1 = ax.bar(x - width/2, mle_queries, width, label="MLE", color=MLE_COLOR, alpha=0.85)
    bars2 = ax.bar(x + width/2, dmle_queries, width, label="DMLE", color=DMLE_COLOR, alpha=0.85)

    ax.set_xlabel("Attack Success Threshold")
    ax.set_ylabel("Queries Required")
    ax.set_title(f"Query Efficiency — {attack_name}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.0%}" for t in thresholds])
    ax.legend()

    # Add value labels on bars
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_perturbation_examples(
    X_clean: np.ndarray,
    X_adv: np.ndarray,
    y_true: np.ndarray,
    y_adv: np.ndarray,
    save_path: str,
    n_examples: int = 4,
    channel: int = 0,
    sfreq: int = 256,
):
    """
    Plot clean vs adversarial EEG signals for visual comparison.

    Shows that perturbations are small / imperceptible in the signal domain.
    """
    class_names = ["Left Hand", "Right Hand", "Feet", "Tongue"]

    fig, axes = plt.subplots(n_examples, 3, figsize=(14, 3 * n_examples))
    t = np.arange(X_clean.shape[-1]) / sfreq

    for i in range(n_examples):
        clean = X_clean[i, 0, channel, :]  # (T,)
        adv = X_adv[i, 0, channel, :]
        diff = adv - clean

        # Clean signal
        axes[i, 0].plot(t, clean, color="steelblue", linewidth=0.8)
        axes[i, 0].set_title(f"Clean (Ch {channel}) — True: {class_names[y_true[i]]}")
        axes[i, 0].set_ylabel("Amplitude")

        # Adversarial signal
        axes[i, 1].plot(t, adv, color="tomato", linewidth=0.8)
        axes[i, 1].set_title(f"Adversarial — Pred: {class_names[y_adv[i]]}")

        # Perturbation
        axes[i, 2].plot(t, diff, color="gray", linewidth=0.8)
        axes[i, 2].set_title(f"Perturbation (L∞={np.abs(diff).max():.4f})")

        for j in range(3):
            axes[i, j].set_xlim(0, t[-1])

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    axes[-1, 2].set_xlabel("Time (s)")

    fig.suptitle("Clean vs. Adversarial EEG Signals", fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
