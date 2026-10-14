from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

from adaptive_pinns.collocation import CollocationSampler
from adaptive_pinns.dataloading import TrainLoader, ValidationLoader
from adaptive_pinns.models import MLP


def loss_data(
    model: MLP,
    x: Float[Array, "batch input_size"],
    y: Float[Array, "batch output_size"],
) -> Float[Array, ""]:
    y_pred = jax.vmap(model)(x)
    return jnp.mean((y_pred - y) ** 2.0)


def loss_physics(
    model: MLP,
    x_c: Float[Array, "size dim"],
    physics_residual: Callable[[MLP, Float[Array, "size dim"]], Float[Array, " size"]],
) -> Float[Array, ""]:
    residual = physics_residual(model, x_c)
    return jnp.mean(residual**2.0)


def loss_pinn(
    model: MLP,
    x: Float[Array, "batch input_size"],
    y: Float[Array, "batch output_size"],
    x_c: Float[Array, "size dim"],
    physics_residual: Callable[[MLP, Float[Array, "size dim"]], Float[Array, " size"]],
    physics_weight: float,
) -> Float[Array, ""]:
    loss = loss_data(model, x, y)
    loss += physics_weight * loss_physics(model, x_c, physics_residual)
    return loss


def evaluate(
    model: MLP,
    validationloader: ValidationLoader,
) -> Float[Array, ""]:
    loss_total = jnp.array(0.0)
    samples = 0

    for x_batch, y_batch in validationloader.iter():
        batch_loss = loss_data(model, x_batch, y_batch)
        batch_size = x_batch.shape[0]

        loss_total += batch_loss * batch_size
        samples += batch_size

    return loss_total / samples


def train(
    model: MLP,
    optimizer: optax.GradientTransformation,
    epochs: int,
    print_every: int,
    trainloader: TrainLoader,
    validationloader: ValidationLoader,
    key: PRNGKeyArray,
) -> MLP:
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(
        model: MLP,
        opt_state: PyTree,
        x: Float[Array, "batch input_size"],
        y: Float[Array, "batch output_size"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss_data)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for epoch in range(1, epochs + 1):
        loss_train = jnp.array(0.0)
        samples_train = 0

        key, subkey = jax.random.split(key)
        for x_batch, y_batch in trainloader.iter(subkey):
            model, opt_state, batch_loss = train_step(model, opt_state, x_batch, y_batch)

            batch_size = x_batch.shape[0]
            loss_train += batch_loss * batch_size
            samples_train += batch_size

        loss_train /= samples_train

        if epoch % print_every == 0:
            loss_validation = evaluate(model, validationloader)
            print(f"epoch={epoch:4d}, loss_train={loss_train.item():.4f}, loss_validation={loss_validation.item():.4f}")

    return model


def train_pinn(
    model: MLP,
    optimizer: optax.GradientTransformation,
    epochs: int,
    print_every: int,
    trainloader: TrainLoader,
    validationloader: ValidationLoader,
    key: PRNGKeyArray,
    collocation_sampler: CollocationSampler,
    physics_residual: Callable[[MLP, Float[Array, "size dim"]], Float[Array, " size"]],
    physics_weight: float,
) -> MLP:
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(
        model: MLP,
        opt_state: PyTree,
        x: Float[Array, "batch input_size"],
        y: Float[Array, "batch output_size"],
        x_collocation: Float[Array, "size dim"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss_pinn)(
            model, x, y, x_collocation, physics_residual, physics_weight
        )
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    for epoch in range(1, epochs + 1):
        loss_train = jnp.array(0.0)
        samples_train = 0

        key, subkey = jax.random.split(key)
        for x_batch, y_batch in trainloader.iter(subkey):
            key, subkey = jax.random.split(key)
            x_c = collocation_sampler.sample(subkey)

            model, opt_state, batch_loss = train_step(model, opt_state, x_batch, y_batch, x_c)

            batch_size = x_batch.shape[0]
            loss_train += batch_loss * batch_size
            samples_train += batch_size

        loss_train /= samples_train

        if epoch % print_every == 0:
            loss_validation = evaluate(model, validationloader)
            print(f"epoch={epoch:4d}, loss_train={loss_train.item():.4f}, loss_validation={loss_validation.item():.4f}")

    return model
