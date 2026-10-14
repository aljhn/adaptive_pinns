# Adaptive PINNs


### Things to consider
* Maybe move from jax to pytorch
* Checkpointing with orbax
* Hyperparameter tuning with optuna
* Maybe replace the data loading stuff with grain
* Consider unit testing with pytest


### Setup
1. Clone the repo
2. Setup virtual environment with uv:
  * CPU only: uv sync
  * GPU support: uv sync --extra gpu


### Commands
* uv run ruff format
* uv run ruff check --fix


### Current Usage
* uv run experiments/1_basic.py
* uv run experiments/2_pinn.py
