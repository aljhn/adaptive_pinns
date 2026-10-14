from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from adaptive_pinns.dataloading import TrainLoader, ValidationLoader, split_data
from adaptive_pinns.models import MLP
from adaptive_pinns.training import train


@dataclass
class Config:
    seed: int = 42069

    input_size: int = 5
    output_size: int = 3
    hidden_size: int = 10
    num_hidden_layers: int = 4

    n_data: int = 500
    batch_size: int = 32

    learning_rate: float = 1e-3
    epochs: int = 100
    print_every: int = 10


def main():
    config = Config()

    key = jax.random.PRNGKey(config.seed)

    key, subkey = jax.random.split(key, 2)
    model = MLP(config.input_size, config.output_size, config.hidden_size, config.num_hidden_layers, subkey)

    key, subkey1 = jax.random.split(key, 2)
    x_data = jax.random.normal(subkey1, shape=(config.n_data, config.input_size))
    y_data = jnp.sin(x_data[:, : config.output_size])

    key, subkey = jax.random.split(key)
    (x_train, y_train), (x_val, y_val) = split_data(x_data, y_data, subkey)

    trainloader = TrainLoader(x_train, y_train, config.batch_size)
    validationloader = ValidationLoader(x_val, y_val, config.batch_size)

    optimizer = optax.adamw(config.learning_rate)

    key, subkey = jax.random.split(key, 2)
    model = train(model, optimizer, config.epochs, config.print_every, trainloader, validationloader, subkey)


if __name__ == "__main__":
    main()
