import random

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from adaptive_pinns.dataloading import TrainLoader, ValidationLoader
from adaptive_pinns.models import MLP
from adaptive_pinns.training import train_pinn
from adaptive_pinns.collocation import UniformCollocationSampler
from adaptive_pinns.problems import Heat1D


def main():
    seed = 42069
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    input_size = 2
    output_size = 1
    hidden_size = 20
    num_hidden_layers = 5

    key, subkey = jax.random.split(key, 2)
    model = MLP(input_size, output_size, hidden_size, num_hidden_layers, subkey)

    k = 1.0
    T0 = 0.0
    T1 = 0.2
    X0 = 0.0
    X1 = 1.0
    domain_boundary = jnp.array([[T0, T1], [X0, X1]])
    problem = Heat1D(k, domain_boundary)

    n_i = 100
    n_b = 100
    key, subkey = jax.random.split(key, 2)
    tx_ib, u_ib = problem.sample_ib(subkey, n_i, n_b)

    batch_size = 32
    trainloader = TrainLoader(tx_ib, u_ib, batch_size)

    n_val = 1000
    key, subkey = jax.random.split(key, 2)
    tx_val, u_val = problem.sample_domain(subkey, n_val)
    validationloader = ValidationLoader(tx_val, u_val, batch_size)

    n_c = 1000
    collocation_sampler = UniformCollocationSampler(n_c, domain_boundary)

    def physics_residual(model, positions):
        diffeq = jax.vmap(lambda position: problem.diffeq(model, position))
        return diffeq(positions)

    initial_learning_rate = 1e-3
    optimizer = optax.adamw(initial_learning_rate)

    epochs = 500
    print_every = 10
    physics_weight = 1e-1
    key, subkey = jax.random.split(key, 2)
    model = train_pinn(
        model,
        optimizer,
        epochs,
        print_every,
        trainloader,
        validationloader,
        subkey,
        collocation_sampler,
        physics_residual,
        physics_weight,
    )

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


if __name__ == "__main__":
    main()
