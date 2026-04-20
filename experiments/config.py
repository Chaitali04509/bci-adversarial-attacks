"""
Experiment configuration — all hyperparameters in one place.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Dataset settings."""
    dataset: str = "synthetic"          # "synthetic" or "bci_comp_iv_2a"
    data_dir: str = "data/raw/"
    n_subjects: int = 9                 # BCI Comp IV 2a has 9 subjects
    subject_ids: List[int] = field(default_factory=lambda: [1])  # which subjects to use
    n_classes: int = 4                  # 4-class motor imagery
    n_channels: int = 22               # EEG channels
    n_timepoints: int = 256            # samples per trial (1s at 256 Hz)
    sfreq: int = 256                   # sampling frequency
    # Synthetic data settings
    synthetic_n_train: int = 2000
    synthetic_n_test: int = 500
    synthetic_n_pool: int = 5000       # unlabeled pool for active learning


@dataclass
class ModelConfig:
    """EEGNet architecture settings."""
    # EEGNet hyperparameters (Lawhern et al. 2018)
    F1: int = 8                         # number of temporal filters
    D: int = 2                          # depth multiplier for depthwise conv
    F2: int = 16                        # F1 * D, number of pointwise filters
    kernel_length: int = 64             # temporal conv kernel size (0.25s at 256Hz)
    dropout_rate: float = 0.5
    # Substitute model (smaller)
    sub_F1: int = 4
    sub_D: int = 2
    sub_F2: int = 8
    sub_kernel_length: int = 32
    sub_dropout_rate: float = 0.25


@dataclass
class ActiveLearningConfig:
    """Active learning loop settings."""
    n_cycles: int = 15                  # number of AL cycles
    seed_size: int = 100                # initial labeled seed set
    query_size: int = 50                # samples queried per cycle
    strategy: str = "uncertainty"       # "uncertainty", "random", "entropy", "margin"
    # Training per cycle
    epochs_per_cycle: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class DMLEConfig:
    """DMLE-specific settings."""
    temperature: float = 1.0            # softmax temperature for acquisition probs
    ema_decay: float = 0.99             # EMA for running weight normalization
    weight_clip: float = 10.0           # max importance weight to prevent instability
    correction_start_cycle: int = 2     # start DMLE correction after this many cycles


@dataclass
class AttackConfig:
    """Adversarial attack settings."""
    methods: List[str] = field(default_factory=lambda: ["fgsm", "pgd"])
    # FGSM
    fgsm_epsilon: float = 0.05          # perturbation magnitude (fraction of input range)
    # PGD
    pgd_epsilon: float = 0.05
    pgd_alpha: float = 0.01             # step size
    pgd_iterations: int = 40
    pgd_random_start: bool = True
    # Evaluation
    n_attack_samples: int = 500         # how many test samples to attack


@dataclass
class ExperimentConfig:
    """Top-level experiment config."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    al: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    dmle: DMLEConfig = field(default_factory=DMLEConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    # General
    seed: int = 42
    device: str = "cuda"                # "cuda" or "cpu"
    results_dir: str = "results/"
    n_runs: int = 3                     # repeat experiment for confidence intervals
