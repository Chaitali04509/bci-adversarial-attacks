"""
Main experiment pipeline.

Runs the full comparison:
  1. Train target model on EEG data
  2. For each method (MLE, DMLE):
     a. Initialize substitute model with seed queries
     b. Run active learning loop (query target → retrain substitute)
     c. After each AL cycle, evaluate attack transferability
  3. Generate comparison plots and save results

Usage:
    python -m experiments.run_experiment
"""
import os
import sys
import json
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config import ExperimentConfig
from data.dataset import load_data
from models.eegnet import build_eegnet
from models.substitute import SubstituteModel
from active_learning.query_strategies import get_strategy
from active_learning.mle_trainer import MLETrainer
from active_learning.dmle_trainer import DMLETrainer
from attacks.evaluate import run_full_attack_evaluation
from utils.metrics import compute_substitute_quality, summarize_results
from utils.visualization import (
    plot_accuracy_curves,
    plot_attack_success,
    plot_agreement_curves,
    plot_query_efficiency,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_target_model(
    model: nn.Module,
    train_dataset,
    test_dataset,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> nn.Module:
    """
    Train the target model (simulates a deployed BCI classifier).

    Returns the trained model frozen for inference.
    """
    print("\n" + "=" * 60)
    print("TRAINING TARGET MODEL")
    print("=" * 60)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        scheduler.step()

        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                test_correct += (logits.argmax(1) == y_batch).sum().item()
                test_total += len(y_batch)

        test_acc = test_correct / test_total
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss/train_total:.4f} | "
                  f"Train Acc: {train_correct/train_total:.3f} | "
                  f"Test Acc: {test_acc:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"  Best target model accuracy: {best_acc:.3f}")
    return model


def run_active_learning(
    method: str,
    target_model: nn.Module,
    pool_dataset,
    test_dataset,
    config: ExperimentConfig,
    device: str = "cpu",
):
    """
    Run one active learning experiment (either MLE or DMLE).

    Args:
        method: "mle" or "dmle"
        target_model: Trained target model (black-box oracle)
        pool_dataset: PoolDataset with unlabeled samples
        test_dataset: EEGDataset for evaluation
        config: ExperimentConfig

    Returns:
        cycle_metrics: List of per-cycle evaluation dicts
        cycle_attacks: List of per-cycle attack result dicts
        substitute_model: Final trained substitute model
    """
    print(f"\n{'=' * 60}")
    print(f"ACTIVE LEARNING — {method.upper()}")
    print(f"{'=' * 60}")

    al_cfg = config.al
    dmle_cfg = config.dmle

    # Build fresh substitute model
    sub_model = build_eegnet(config, model_type="substitute")

    # Initialize trainer
    if method == "mle":
        trainer = MLETrainer(
            sub_model, lr=al_cfg.lr, weight_decay=al_cfg.weight_decay, device=device,
        )
    elif method == "dmle":
        trainer = DMLETrainer(
            sub_model, lr=al_cfg.lr, weight_decay=al_cfg.weight_decay,
            weight_clip=dmle_cfg.weight_clip,
            correction_start_cycle=dmle_cfg.correction_start_cycle,
            ema_decay=dmle_cfg.ema_decay, device=device,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Get test data as tensors
    X_test = test_dataset.X
    y_test = test_dataset.y

    # Get query strategy
    strategy_fn = get_strategy(al_cfg.strategy)

    # Track state
    sub_wrapper = SubstituteModel(sub_model, device=device)
    cycle_metrics = []
    cycle_attacks = []
    query_counts = []
    total_queries = 0

    # Reset pool
    pool_dataset.queried_mask[:] = False

    # ---- Seed Phase: Random initial queries ----
    print(f"\n  Seed phase: querying {al_cfg.seed_size} random samples...")
    seed_indices = np.random.choice(
        pool_dataset.unqueried_indices, size=al_cfg.seed_size, replace=False
    )
    X_seed, y_seed = pool_dataset.query(seed_indices, target_model, device=device)
    sub_wrapper.add_labeled_data(X_seed, y_seed)
    total_queries += al_cfg.seed_size

    # ---- Active Learning Cycles ----
    for cycle in range(al_cfg.n_cycles):
        print(f"\n  Cycle {cycle+1}/{al_cfg.n_cycles} "
              f"(labeled: {sub_wrapper.n_labeled}, queries: {total_queries})")

        # Train substitute model
        if method == "mle":
            history = trainer.train_cycle(
                sub_wrapper.X_labeled, sub_wrapper.y_labeled,
                epochs=al_cfg.epochs_per_cycle, batch_size=al_cfg.batch_size,
            )
        elif method == "dmle":
            history = trainer.train_cycle(
                sub_wrapper.X_labeled, sub_wrapper.y_labeled,
                acquisition_probs_history=sub_wrapper.acquisition_probs,
                n_seed=al_cfg.seed_size,
                epochs=al_cfg.epochs_per_cycle, batch_size=al_cfg.batch_size,
            )

        train_acc = history["accuracy"][-1]

        # Evaluate substitute quality
        metrics = compute_substitute_quality(
            sub_model, target_model, X_test, y_test, device=device,
        )
        metrics["cycle"] = cycle + 1
        metrics["total_queries"] = total_queries
        metrics["train_accuracy"] = train_acc
        cycle_metrics.append(metrics)

        print(f"    Train Acc: {train_acc:.3f} | "
              f"Sub Acc: {metrics['substitute_accuracy']:.3f} | "
              f"Agreement: {metrics['agreement']:.3f}")

        # Evaluate attacks at this cycle
        attack_results = run_full_attack_evaluation(
            sub_model, target_model, X_test, y_test,
            config.attack, device=device,
        )
        cycle_attacks.append(attack_results)

        for atk_name, atk_metrics in attack_results.items():
            print(f"    {atk_name.upper()} Attack Success: "
                  f"{atk_metrics['attack_success_rate']:.3f} | "
                  f"Fooling Rate: {atk_metrics['fooling_rate']:.3f}")

        # Query new samples (unless last cycle)
        if cycle < al_cfg.n_cycles - 1:
            unqueried = pool_dataset.unqueried_indices
            if len(unqueried) < al_cfg.query_size:
                print(f"    Pool exhausted ({len(unqueried)} remaining). Stopping.")
                break

            selected_indices, acq_probs = strategy_fn(
                sub_model, pool_dataset.X, unqueried,
                n_query=al_cfg.query_size, device=device,
                temperature=dmle_cfg.temperature,
            )

            X_new, y_new = pool_dataset.query(
                selected_indices, target_model, device=device,
            )
            sub_wrapper.add_labeled_data(X_new, y_new, acq_probs=acq_probs)
            total_queries += al_cfg.query_size

        query_counts.append(total_queries)

    return cycle_metrics, cycle_attacks, sub_model, query_counts


def main():
    """Run the full experiment."""
    config = ExperimentConfig()

    # Use CPU if CUDA unavailable
    device = config.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    os.makedirs(config.results_dir, exist_ok=True)

    all_results = []

    for run_idx in range(config.n_runs):
        print(f"\n{'#' * 60}")
        print(f"# RUN {run_idx + 1}/{config.n_runs}")
        print(f"{'#' * 60}")

        set_seed(config.seed + run_idx)

        # Load data
        datasets = load_data(config.data)

        # Train target model
        target_model = build_eegnet(config, model_type="target")
        target_model = train_target_model(
            target_model, datasets["train"], datasets["test"],
            epochs=100, batch_size=config.al.batch_size,
            lr=config.al.lr, device=device,
        )

        # Run MLE baseline
        pool_mle = copy.deepcopy(datasets["pool"])
        mle_metrics, mle_attacks, mle_sub, mle_queries = run_active_learning(
            "mle", target_model, pool_mle, datasets["test"], config, device=device,
        )

        # Run DMLE (ours)
        pool_dmle = copy.deepcopy(datasets["pool"])
        dmle_metrics, dmle_attacks, dmle_sub, dmle_queries = run_active_learning(
            "dmle", target_model, pool_dmle, datasets["test"], config, device=device,
        )

        all_results.append({
            "mle_metrics": mle_metrics,
            "dmle_metrics": dmle_metrics,
            "mle_attacks": mle_attacks,
            "dmle_attacks": dmle_attacks,
            "mle_queries": mle_queries,
            "dmle_queries": dmle_queries,
        })

    # ---- Aggregate and Plot Results ----
    print(f"\n{'=' * 60}")
    print("GENERATING PLOTS")
    print(f"{'=' * 60}")

    # Use last run for plots (or average across runs for multi-run)
    result = all_results[-1]
    query_counts = result["mle_queries"]

    mle_accs = [m["substitute_accuracy"] for m in result["mle_metrics"]]
    dmle_accs = [m["substitute_accuracy"] for m in result["dmle_metrics"]]
    mle_agree = [m["agreement"] for m in result["mle_metrics"]]
    dmle_agree = [m["agreement"] for m in result["dmle_metrics"]]

    # 1. Accuracy curves
    plot_accuracy_curves(
        mle_accs, dmle_accs, query_counts,
        save_path=os.path.join(config.results_dir, "accuracy_curves.png"),
    )

    # 2. Agreement curves
    plot_agreement_curves(
        mle_agree, dmle_agree, query_counts,
        save_path=os.path.join(config.results_dir, "agreement_curves.png"),
    )

    # 3. Attack success rates
    mle_success = {}
    dmle_success = {}
    for atk_name in config.attack.methods:
        mle_success[atk_name] = [
            a.get(atk_name, {}).get("attack_success_rate", 0)
            for a in result["mle_attacks"]
        ]
        dmle_success[atk_name] = [
            a.get(atk_name, {}).get("attack_success_rate", 0)
            for a in result["dmle_attacks"]
        ]

    plot_attack_success(
        mle_success, dmle_success, query_counts,
        save_path=os.path.join(config.results_dir, "attack_success.png"),
    )

    # 4. Query efficiency (for FGSM)
    if "fgsm" in mle_success:
        plot_query_efficiency(
            mle_success["fgsm"], dmle_success["fgsm"], query_counts,
            thresholds=[0.2, 0.4, 0.6],
            save_path=os.path.join(config.results_dir, "query_efficiency_fgsm.png"),
            attack_name="FGSM",
        )

    if "pgd" in mle_success:
        plot_query_efficiency(
            mle_success["pgd"], dmle_success["pgd"], query_counts,
            thresholds=[0.2, 0.4, 0.6],
            save_path=os.path.join(config.results_dir, "query_efficiency_pgd.png"),
            attack_name="PGD",
        )

    # 5. Save numerical results
    summary = summarize_results(
        result["mle_metrics"], result["dmle_metrics"],
        result["mle_attacks"], result["dmle_attacks"],
    )
    summary["config"] = {
        "n_cycles": config.al.n_cycles,
        "seed_size": config.al.seed_size,
        "query_size": config.al.query_size,
        "strategy": config.al.strategy,
        "fgsm_epsilon": config.attack.fgsm_epsilon,
        "pgd_epsilon": config.attack.pgd_epsilon,
        "n_runs": config.n_runs,
    }

    results_path = os.path.join(config.results_dir, "metrics.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {results_path}")

    # Print final summary
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Target Model Accuracy: {result['mle_metrics'][-1]['target_accuracy']:.3f}")
    print(f"  --- MLE (Baseline) ---")
    print(f"    Final Sub Accuracy: {mle_accs[-1]:.3f}")
    print(f"    Final Agreement:    {mle_agree[-1]:.3f}")
    for atk in config.attack.methods:
        val = mle_success.get(atk, [0])[-1]
        print(f"    {atk.upper()} Success Rate: {val:.3f}")
    print(f"  --- DMLE (Ours) ---")
    print(f"    Final Sub Accuracy: {dmle_accs[-1]:.3f}")
    print(f"    Final Agreement:    {dmle_agree[-1]:.3f}")
    for atk in config.attack.methods:
        val = dmle_success.get(atk, [0])[-1]
        print(f"    {atk.upper()} Success Rate: {val:.3f}")

    print(f"\nAll results saved to: {config.results_dir}")


if __name__ == "__main__":
    main()
