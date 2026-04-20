"""
Experiment configuration — all hyperparameters in one place.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    """Dataset settings."""
    dataset: str = "synthetic"
    data_dir: str = "data/raw/"
    n_subjects: int = 9
    subject_ids: List[int] = field(default_factory=lambda: [1])
    n_classes: int = 4
    n_channels: int = 22
    n_timepoints: int = 256
    sfreq: int = 256
    synthetic_n_train: int = 2000
    synthetic_n_test: int = 500
    synthetic_n_pool: int = 5000


@dataclass
class ModelConfig:
    """EEGNet architecture settings."""
    # Target model (full EEGNet)
    F1: int = 16
    D: int = 2
    F2: int = 32
    kernel_length: int = 64
    dropout_rate: float = 0.5
    # Substitute model (intentionally weaker — different architecture)
    sub_F1: int = 4
    sub_D: int = 1
    sub_F2: int = 4
    sub_kernel_length: int = 32
    sub_dropout_rate: float = 0.25


@dataclass
class ActiveLearningConfig:
    """Active learning loop settings."""
    n_cycles: int = 20
    seed_size: int = 80
    query_size: int = 60
    strategy: str = "uncertainty"
    epochs_per_cycle: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class DMLEConfig:
    """DMLE-specific settings."""
    temperature: float = 1.0
    ema_decay: float = 0.99
    weight_clip: float = 10.0
    correction_start_cycle: int = 2


@dataclass
class AttackConfig:
    """Adversarial attack settings."""
    methods: List[str] = field(default_factory=lambda: ["fgsm", "pgd"])
    fgsm_epsilon: float = 0.3
    pgd_epsilon: float = 0.3
    pgd_alpha: float = 0.05
    pgd_iterations: int = 40
    pgd_random_start: bool = True
    n_attack_samples: int = 500


@dataclass
class ExperimentConfig:
    """Top-level experiment config."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    al: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    dmle: DMLEConfig = field(default_factory=DMLEConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    seed: int = 42
    device: str = "cuda"
    results_dir: str = "results/"
    n_runs: int = 1
