import random

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from adaptive_pinns.dataloading import Trainloader, Validationloader
from adaptive_pinns.models import MLP
from adaptive_pinns.training import train_pinn


def heat_true(t, x):
    return jnp.sin(jnp.pi * x) * jnp.exp(-(jnp.pi**2.0) * t)


def heat_initial_condition(key, n, T0, X0, X1):
    t_i = jnp.ones((n,)) * T0
    x_i = jax.random.uniform(key, shape=(n,), minval=X0, maxval=X1)
    u_i = heat_true(t_i, x_i)
    u_i = u_i[:, None]
    tx_i = jnp.stack((t_i, x_i), axis=1)
    return tx_i, u_i


def heat_boundary_condition(key, n, T0, T1, X0, X1):
    t_b = jax.random.uniform(key, shape=(n,), minval=T0, maxval=T1)
    x_b0 = jnp.ones((n // 2,)) * X0
    x_b1 = jnp.ones((n // 2,)) * X1
    x_b = jnp.concatenate((x_b0, x_b1), axis=0)
    tx_b = jnp.stack((t_b, x_b), axis=1)
    u_b = heat_true(t_b, x_b)
    u_b = u_b[:, None]
    return tx_b, u_b


def heat_validation(key, n, T0, T1, X0, X1):
    key1, key2 = jax.random.split(key, 2)
    t = jax.random.uniform(key1, shape=(n,), minval=T0, maxval=T1)
    x = jax.random.uniform(key2, shape=(n,), minval=X0, maxval=X1)
    u = heat_true(t, x)
    u = u[:, None]
    tx = jnp.stack((t, x), axis=1)
    return tx, u


def collocation_sample(key, n, T0, T1, X0, X1):
    t_pinn = jax.random.uniform(key, shape=(n,), minval=T0, maxval=T1)
    x_pinn = jax.random.uniform(key, shape=(n,), minval=X0, maxval=X1)
    tx_pinn = jnp.stack((t_pinn, x_pinn), axis=1)
    return tx_pinn


def heat_dynamics(model, state, k):
    def u(tx):
        return model(tx)[0]

    u_t = jax.grad(u)(state)[0]
    u_xx = jax.grad(lambda tx: jax.grad(u)(tx)[1])(state)[1]

    return u_t - k * u_xx


class CollocationSampler:
    def __init__(self, key, n, T0, T1, X0, X1):
        self.key = key
        self.n = n
        self.T0 = T0
        self.T1 = T1
        self.X0 = X0
        self.X1 = X1

    def __call__(self):
        key, subkey = jax.random.split(self.key)
        self.key = key
        return collocation_sample(subkey, self.n, self.T0, self.T1, self.X0, self.X1)


def main():
    seed = 42069
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    input_size = 2
    output_size = 1
    hidden_size = 100
    num_hidden_layers = 5

    key, subkey = jax.random.split(key, 2)
    model = MLP(input_size, output_size, hidden_size, num_hidden_layers, subkey)

    T0 = 0.0
    T1 = 0.2
    X0 = 0.0
    X1 = 1.0

    n_i = 100
    key, subkey = jax.random.split(key, 2)
    tx_i, u_i = heat_initial_condition(subkey, n_i, T0, X0, X1)

    n_b = 100
    key, subkey = jax.random.split(key, 2)
    tx_b, u_b = heat_boundary_condition(subkey, n_b, T0, T1, X0, X1)

    tx_ib = jnp.concatenate((tx_i, tx_b), axis=0)
    u_ib = jnp.concatenate((u_i, u_b), axis=0)

    batch_size = 32
    key, subkey = jax.random.split(key, 2)
    trainloader = Trainloader(tx_ib, u_ib, batch_size, subkey)

    n_val = 1000
    key, subkey = jax.random.split(key, 2)
    tx_val, u_val = heat_validation(subkey, n_val, T0, T1, X0, X1)
    validationloader = Validationloader(tx_val, u_val, batch_size)

    n_c = 1000
    key, subkey = jax.random.split(key, 2)
    collocation_sampler = CollocationSampler(subkey, n_c, T0, T1, X0, X1)

    k = 1.0
    physics_residual = lambda model, states: jax.vmap(lambda state: heat_dynamics(model, state, k))(states)

    initial_learning_rate = 1e-4
    optimizer = optax.adamw(initial_learning_rate)

    epochs = 100
    print_every = 10
    model = train_pinn(model, optimizer, epochs, print_every, trainloader, validationloader, collocation_sampler, physics_residual, 1e-1)

    n = 100
    t = jnp.linspace(T0, T1, n)
    x = jnp.linspace(X0, X1, n)
    tt, xx = jnp.meshgrid(t, x, indexing="xy")
    ttxx = jnp.stack((tt.flatten(), xx.flatten()), axis=1)
    uu = jax.vmap(model)(ttxx)
    un = jnp.reshape(uu, tt.shape)
    uu_true = heat_true(tt, xx)

    print("True difference:", jnp.mean((un - uu_true)**2))

    plt.figure()
    plt.pcolormesh(t, x, un, vmin=0.0, vmax=1.0, cmap="rainbow")
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
