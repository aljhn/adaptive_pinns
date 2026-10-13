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


def split_data(
    x_data: Float[Array, "data_size input_size"],
    y_data: Float[Array, "data_size output_size"],
    key: PRNGKeyArray,
    train_split: int = 80,
) -> tuple[
    tuple[Float[Array, "train_size input_size"], Float[Array, "train_size output_size"]],
    tuple[Float[Array, "val_size input_size"], Float[Array, "val_size output_size"]],
]:
    data_amount = x_data.shape[0]
    indexes = jax.random.permutation(key, data_amount)

    train_amount = (data_amount * train_split) // 100
    indexes_train = indexes[:train_amount]
    x_train = x_data[indexes_train]
    y_train = y_data[indexes_train]

    indexes_val = indexes[train_amount:]
    x_val = x_data[indexes_val]
    y_val = y_data[indexes_val]

    return (x_train, y_train), (x_val, y_val)
