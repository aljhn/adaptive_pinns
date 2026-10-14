from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

from adaptive_pinns.collocation import UniformCollocationSampler
from adaptive_pinns.dataloading import TrainLoader, ValidationLoader
from adaptive_pinns.models import MLP
from adaptive_pinns.problems import Heat1D
from adaptive_pinns.training import train_pinn


@dataclass
class Config:
    seed: int = 42069

    input_size: int = 2
    output_size: int = 1
    hidden_size: int = 20
    num_hidden_layers: int = 5

    k: float = 1.0
    t0: float = 0.0
    t1: float = 0.2
    x0: float = 0.0
    x1: float = 1.0

    n_initial: int = 100
    n_boundary: int = 100
    n_validation: int = 1000
    n_collocation: int = 1000
    batch_size: int = 32

    learning_rate: float = 1e-3
    epochs: int = 500
    print_every: int = 10
    physics_weight: float = 1e-1


def train_model(config, problem, key):
    key, subkey = jax.random.split(key, 2)
    model = MLP(config.input_size, config.output_size, config.hidden_size, config.num_hidden_layers, subkey)

    key, subkey = jax.random.split(key, 2)
    tx_ib, u_ib = problem.sample_ib(subkey, config.n_initial, config.n_boundary)

    key, subkey = jax.random.split(key, 2)
    tx_val, u_val = problem.sample_validation(subkey, config.n_validation)

    trainloader = TrainLoader(tx_ib, u_ib, config.batch_size)
    validationloader = ValidationLoader(tx_val, u_val, config.batch_size)
    collocation_sampler = UniformCollocationSampler(config.n_collocation, problem.domain_boundary)

    def physics_residual(model, positions):
        return jax.vmap(problem.diffeq, in_axes=(None, 0))(model, positions)

    optimizer = optax.adamw(config.learning_rate)

    key, subkey = jax.random.split(key, 2)
    model = train_pinn(
        model,
        optimizer,
        config.epochs,
        config.print_every,
        trainloader,
        validationloader,
        subkey,
        collocation_sampler,
        physics_residual,
        config.physics_weight,
    )

    return model


def evaluate_model(model, problem):
    T0 = problem.domain_boundary[0, 0]
    T1 = problem.domain_boundary[0, 1]
    X0 = problem.domain_boundary[1, 0]
    X1 = problem.domain_boundary[1, 1]

    n = 100
    t = jnp.linspace(T0, T1, n)
    x = jnp.linspace(X0, X1, n)
    tt, xx = jnp.meshgrid(t, x, indexing="xy")
    ttxx = jnp.stack((tt.ravel(), xx.ravel()), axis=1)
    uu = jax.vmap(model)(ttxx).reshape(tt.shape)
    uu_true = problem.get_true(ttxx).reshape(tt.shape)

    print("MSE:", jnp.mean((uu - uu_true) ** 2))

    plt.figure()
    plt.pcolormesh(t, x, uu, vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.pcolormesh(t, x, uu_true, vmin=0.0, vmax=1.0, cmap="rainbow")
    plt.colorbar()
    plt.xlabel(r"$t$")
    plt.ylabel(r"$x$")
    plt.tight_layout()
    plt.show()


def main():
    config = Config()

    domain_boundary = jnp.array([[config.t0, config.t1], [config.x0, config.x1]])
    problem = Heat1D(config.k, domain_boundary)

    key = jax.random.PRNGKey(config.seed)
    model = train_model(config, problem, key)

    evaluate_model(model, problem)


if __name__ == "__main__":
    main()
