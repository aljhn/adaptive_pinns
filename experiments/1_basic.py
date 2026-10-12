import random

import jax
import jax.numpy as jnp
import numpy as np
import optax

from adaptive_pinns.dataloading import Trainloader, Validationloader
from adaptive_pinns.models import MLP
from adaptive_pinns.training import train


def main():
    seed = 42069
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)

    input_size = 5
    output_size = 3
    hidden_size = 10
    num_hidden_layers = 4

    key, subkey = jax.random.split(key, 2)
    model = MLP(input_size, output_size, hidden_size, num_hidden_layers, subkey)

    key, subkey1 = jax.random.split(key, 2)
    data_amount = 500
    x_data = jax.random.normal(subkey1, shape=(data_amount, input_size))
    y_data = jnp.sin(x_data[:, :output_size])

    key, subkey = jax.random.split(key)
    indexes = jax.random.permutation(subkey, data_amount)

    train_split = 80
    train_amount = (data_amount * train_split) // 100

    indexes_train = indexes[:train_amount]
    x_train = x_data[indexes_train]
    y_train = y_data[indexes_train]

    indexes_val = indexes[train_amount:]
    x_val = x_data[indexes_val]
    y_val = y_data[indexes_val]

    batch_size = 32
    key, subkey = jax.random.split(key, 2)
    trainloader = Trainloader(x_train, y_train, batch_size, subkey)

    validationloader = Validationloader(x_val, y_val, batch_size)

    initial_learning_rate = 1e-4
    optimizer = optax.adamw(initial_learning_rate)

    epochs = 100
    print_every = 10
    model = train(model, optimizer, epochs, print_every, trainloader, validationloader)


if __name__ == "__main__":
    main()
