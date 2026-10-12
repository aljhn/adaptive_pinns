import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PyTree

from adaptive_pinns.dataloading import Trainloader, Validationloader
from adaptive_pinns.models import MLP


def loss(
    model: MLP,
    x: Float[Array, "batch input_size"],
    y: Float[Array, "batch output_size"],
) -> Float[Array, ""]:
    y_pred = jax.vmap(model)(x)
    return jnp.mean(optax.losses.l2_loss(y_pred, y))


def evaluate(
    model: MLP,
    validationloader: Validationloader,
) -> Float[Array, ""]:
    loss_total = jnp.array(0.0)
    samples = 0

    for x_batch, y_batch in validationloader:
        batch_loss = loss(model, x_batch, y_batch)
        batch_size = x_batch.shape[0]

        loss_total += batch_loss * batch_size
        samples += batch_size

    return loss_total / samples


def train(
    model: MLP,
    optimizer: optax.GradientTransformation,
    epochs: int,
    print_every: int,
    trainloader: Trainloader,
    validationloader: Validationloader,
) -> MLP:
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(
        model: MLP,
        opt_state: PyTree,
        x: Float[Array, "batch input_size"],
        y: Float[Array, "batch output_size"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for epoch in range(1, epochs + 1):
        loss_train = jnp.array(0.0)
        samples_train = 0

        for x_batch, y_batch in trainloader:
            model, opt_state, batch_loss = train_step(model, opt_state, x_batch, y_batch)

            batch_size = x_batch.shape[0]
            loss_train += batch_loss * batch_size
            samples_train += batch_size

        loss_train /= samples_train

        if epoch % print_every == 0:
            loss_validation = evaluate(model, validationloader)
            print(f"epoch={epoch:4d}, loss_train={loss_train.item():.4f}, loss_validation={loss_validation.item():.4f}")

    return model
