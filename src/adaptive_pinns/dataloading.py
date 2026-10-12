import jax
from jaxtyping import Array, Float, PRNGKeyArray


class Trainloader:
    def __init__(
        self,
        x: Float[Array, "data_size input_size"],
        y: Float[Array, "data_size output_size"],
        batch_size: int,
        key: PRNGKeyArray,
    ):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.key = key

    def __iter__(self):
        data_size = self.x.shape[0]

        key, subkey = jax.random.split(self.key)
        self.key = key
        indexes = jax.random.permutation(subkey, data_size)

        iterations = (data_size + self.batch_size - 1) // self.batch_size
        for i in range(iterations):
            index_start = i * self.batch_size
            index_end = min((i + 1) * self.batch_size, data_size)
            batch_indexes = indexes[index_start:index_end]
            yield self.x[batch_indexes], self.y[batch_indexes]


class Validationloader:
    def __init__(
        self,
        x: Float[Array, "data_size input_size"],
        y: Float[Array, "data_size output_size"],
        batch_size: int,
    ):
        self.x = x
        self.y = y
        self.batch_size = batch_size

    def __iter__(self):
        data_size = self.x.shape[0]

        iterations = (data_size + self.batch_size - 1) // self.batch_size
        for i in range(iterations):
            index_start = i * self.batch_size
            index_end = min((i + 1) * self.batch_size, data_size)
            yield self.x[index_start:index_end], self.y[index_start:index_end]
