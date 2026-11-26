# Black-Box Adversarial Attacks on EEG-Based BCIs: Evaluating Dependency-Aware vs. Standard Active Learning

## Authors
- Alexandros Kerwick
- Nadia Maryam
- Chaitali Nimse

Northeastern University — Electrical and Computer Engineering

---

## Overview

This project implements black-box adversarial attacks on EEG-based Brain-Computer Interfaces (BCIs), comparing conventional active learning (MLE) with Dependency-Aware Maximum Likelihood Estimation (DMLE) for substitute model training. The goal is to test whether DMLE, by modeling the sample dependencies that active learning introduces, enables more query-efficient attacks than standard MLE.

**Current findings (synthetic data, 1 run, ~1,200 queries):** the MLE-trained substitute reached 86.6% accuracy and 100% PGD transfer attack success, while the DMLE-trained substitute reached 41.8% accuracy and ~77% PGD success. In this setup, DMLE did not improve query efficiency over standard MLE.

## Project Structure

```
bci-adversarial/
├── data/
│   ├── __init__.py
│   ├── dataset.py          # EEG dataset loading (BCI Comp IV 2a + synthetic)
│   └── preprocessing.py    # Signal preprocessing & augmentation
├── models/
│   ├── __init__.py
│   ├── eegnet.py           # EEGNet architecture (target model)
│   └── substitute.py       # Substitute model (smaller EEGNet variant)
├── active_learning/
│   ├── __init__.py
│   ├── query_strategies.py # Uncertainty sampling, random, entropy
│   ├── mle_trainer.py      # Standard MLE training loop
│   └── dmle_trainer.py     # DMLE training with dependency correction
├── attacks/
│   ├── __init__.py
│   ├── fgsm.py             # Fast Gradient Sign Method
│   ├── pgd.py              # Projected Gradient Descent
│   └── evaluate.py         # Attack success rate evaluation
├── experiments/
│   ├── __init__.py
│   ├── run_experiment.py   # Main experiment pipeline
│   └── config.py           # Hyperparameters and experiment config
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # Evaluation metrics
│   └── visualization.py    # Plotting results
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset (BCI Competition IV 2a)
#    - Go to: https://www.bbci.de/competition/iv/
#    - Download Dataset 2a (9 subjects, 4-class motor imagery)
#    - Place .gdf files in data/raw/bci_comp_iv_2a/
#
#    OR use the synthetic data generator for quick testing:
python -c "from data.dataset import generate_synthetic_eeg; generate_synthetic_eeg('data/synthetic/')"

# 4. Run the full experiment
python -m experiments.run_experiment
```

## Key Concepts

### Pipeline
1. **Train target model** (EEGNet) on real/synthetic EEG data
2. **Initialize substitute model** with small seed set queried from target
3. **Active learning loop**: select informative samples → query target → retrain substitute
   - **MLE baseline**: standard cross-entropy training
   - **DMLE**: corrected loss accounting for sample selection dependencies
4. **Generate adversarial examples** on substitute model (FGSM/PGD)
5. **Evaluate transferability** of adversarial examples to target model

### DMLE Correction
Standard MLE assumes i.i.d. samples, but active learning selects samples sequentially
based on model state, creating dependencies. DMLE corrects for this by reweighting
the loss function using importance weights derived from the acquisition probability
of each sample.

## Experiment Configuration

Edit `experiments/config.py` to adjust:
- Number of active learning cycles
- Query budget per cycle
- Attack parameters (epsilon, iterations)
- Model architecture settings
- Dataset selection (synthetic vs real)

## Expected Outputs

The experiment produces:
- `results/accuracy_curves.png` — Substitute model accuracy over AL cycles
- `results/attack_success.png` — Attack success rate comparison (MLE vs DMLE)
- `results/query_efficiency.png` — Queries needed to reach attack thresholds
- `results/metrics.json` — All numerical results
