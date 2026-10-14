from abc import ABC, abstractmethod
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class InitialBoundaryValueProblem(ABC):
    domain_boundary: Float[Array, "dims 2"]

    def __init__(self, domain_boundary: Float[Array, "dims 2"]):
        self.domain_boundary = domain_boundary

    @abstractmethod
    def get_true(self, position: Float[Array, "batch input_dim"]) -> Float[Array, "batch output_dim"]:
        pass

    @abstractmethod
    def sample_initial_conditions(
        self, key: PRNGKeyArray, n: int
    ) -> tuple[Float[Array, "batch input_dim"], Float[Array, "batch output_dim"]]:
        pass

    @abstractmethod
    def sample_boundary_conditions(
        self, key: PRNGKeyArray, n: int
    ) -> tuple[Float[Array, "batch input_dim"], Float[Array, "batch output_dim"]]:
        pass

    def sample_ib(
        self, key: PRNGKeyArray, n_i: int, n_b: int
    ) -> tuple[Float[Array, "batch input_dim"], Float[Array, "batch output_dim"]]:
        key1, key2 = jax.random.split(key, 2)
        x_i, u_i = self.sample_initial_conditions(key1, n_i)
        x_b, u_b = self.sample_boundary_conditions(key2, n_b)
        x_ib = jnp.concatenate((x_i, x_b), axis=0)
        u_ib = jnp.concatenate((u_i, u_b), axis=0)
        return x_ib, u_ib

    def sample_validation(
        self, key: PRNGKeyArray, n: int
    ) -> tuple[Float[Array, "batch input_dim"], Float[Array, "batch output_dim"]]:
        dims = self.domain_boundary.shape[0]
        domain_boundary_min = self.domain_boundary[:, 0]
        domain_boundary_max = self.domain_boundary[:, 1]
        x = jax.random.uniform(key, shape=(n, dims), minval=domain_boundary_min, maxval=domain_boundary_max)
        u = self.get_true(x)
        return x, u

    @abstractmethod
    def diffeq(
        self,
        model: Callable[[Float[Array, " input_dim"]], Float[Array, " output_dim"]],
        position: Float[Array, " input_dim"],
    ) -> Float[Array, " output_dim"]:
        pass


class Heat1D(InitialBoundaryValueProblem):
    k: float
    t0: float
    t1: float
    x0: float
    x1: float

    def __init__(self, k: float, domain_boundary: Float[Array, "dims 2"]):
        super().__init__(domain_boundary)
        self.k = k
        self.t0 = self.domain_boundary[0, 0].item()
        self.t1 = self.domain_boundary[0, 1].item()
        self.x0 = self.domain_boundary[1, 0].item()
        self.x1 = self.domain_boundary[1, 1].item()

    def get_true(self, position: Float[Array, "batch input_dim"]) -> Float[Array, "batch output_dim"]:
        t = position[:, 0, None]
        x = position[:, 1, None]

        L = self.x1 - self.x0
        F = jnp.sin(jnp.pi * (x - self.x0) / L)
        G = jnp.exp(-((jnp.pi / L) ** 2.0) * self.k * (t - self.t0))
        return F * G

    def sample_initial_conditions(
        self, key: PRNGKeyArray, n: int
    ) -> tuple[Float[Array, "batch input_dim"], Float[Array, "batch output_dim"]]:
        t = jnp.full((n,), self.t0)
        x = jax.random.uniform(key, shape=(n,), minval=self.x0, maxval=self.x1)
        tx = jnp.stack((t, x), axis=1)
        u = self.get_true(tx)
        return tx, u

    def sample_boundary_conditions(
        self, key: PRNGKeyArray, n: int
    ) -> tuple[Float[Array, "batch input_dim"], Float[Array, "batch output_dim"]]:
        t = jax.random.uniform(key, shape=(n,), minval=self.t0, maxval=self.t1)
        n0 = n // 2
        n1 = n - n0
        x0 = jnp.full((n0,), self.x0)
        x1 = jnp.full((n1,), self.x1)
        x = jnp.concatenate((x0, x1), axis=0)
        tx = jnp.stack((t, x), axis=1)
        u = self.get_true(tx)
        return tx, u

    def diffeq(
        self,
        model: Callable[[Float[Array, " input_dim"]], Float[Array, " output_dim"]],
        position: Float[Array, " input_dim"],
    ) -> Float[Array, ""]:
        def u(position):
            return model(position)[0]

        u_grad = jax.grad(u)(position)
        u_hessian = jax.hessian(u)(position)

        u_t = u_grad[0]
        u_xx = u_hessian[1, 1]
        return u_t - self.k * u_xx
