from abc import ABC, abstractmethod

import jax
from jaxtyping import Array, Float, PRNGKeyArray


class CollocationSampler(ABC):
    @abstractmethod
    def sample(self, key: PRNGKeyArray) -> Float[Array, "points dims"]:
        pass


class UniformCollocationSampler(CollocationSampler):
    n: int
    domain_boundary: Float[Array, "dims 2"]

    def __init__(
        self,
        n: int,
        domain_boundary: Float[Array, "dims 2"],
    ):
        self.n = n
        self.dims = domain_boundary.shape[0]
        self.boundary_min = domain_boundary[:, 0]
        self.boundary_max = domain_boundary[:, 1]

    def sample(self, key: PRNGKeyArray) -> Float[Array, "points dims"]:
        return jax.random.uniform(key, shape=(self.n, self.dims), minval=self.boundary_min, maxval=self.boundary_max)
