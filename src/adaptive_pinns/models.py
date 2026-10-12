import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class MLP(eqx.Module):
    layers: list

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, num_hidden_layers + 1)

        self.layers = []
        self.layers.append(eqx.nn.Linear(input_size, hidden_size, key=keys[0]))
        self.layers.append(jnp.tanh)

        for i in range(1, num_hidden_layers):
            self.layers.append(eqx.nn.Linear(hidden_size, hidden_size, key=keys[i]))
            self.layers.append(jnp.tanh)

        self.layers.append(eqx.nn.Linear(hidden_size, output_size, key=keys[-1]))

    def __call__(self, x: Float[Array, " input_size"]) -> Float[Array, " output_size"]:
        for layer in self.layers:
            x = layer(x)
        return x
