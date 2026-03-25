# Federated Learning Image Classification

This project runs **centralized (single-machine) training** and **federated learning (FL)** for image classification, and compares performance across multiple experiments (e.g. different **data distributions** such as Dirichlet \(\alpha\)).

## Setup

- **Python**: use a Python 3 environment
- **Data**: place/download datasets under `data/` (see scripts for expected structure)
- **Outputs**: plots and artifacts are written under `results/` (and some example images may be in the repo root)

## Experiments

- **Centralized ML**: train on the full dataset centrally, then evaluate.
- **Federated Learning**: split data across clients (optionally non-IID via Dirichlet \(\alpha\)), train locally, and aggregate.
- **Comparisons**: run multiple settings (e.g. different \(\alpha\)) and plot accuracy/loss and other overlays.

## Code organization

- `src/Centralized_Training/`: centralized training scripts
- `src/Federated_Learning/`: FL training / aggregation logic
- `src/client_training.py`: client-side training routine used by FL runs
- `src/data_loading.py`: dataset loading + partitioning utilities (incl. distribution logic)
- `src/plotting.py`: plotting utilities (accuracy/loss curves, etc.)
- `src/overlay/`: scripts to generate comparison overlays across experiments
- `src/model_implementation.py`: model definition / training helpers

## Quick start (entry points)

Typical entry points live in:

- `src/Centralized_Training/centralised_ML.py`
- `src/Federated_Learning/FL.py`
