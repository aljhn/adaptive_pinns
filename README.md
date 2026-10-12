# Adaptive PINNs

### Todos before starting actual work:
* Maybe move from jax to pytorch. Not sure
* Establish proper collaborative git workflow
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
* uv run ruff check --fix
* uv run ruff format


### Current Usage
* uv run experiments/1_basic.py
