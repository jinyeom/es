import numpy as np

from .optim import Optimizer

__all__ = ["EvolutionStrategy"]


class EvolutionStrategy:
    """
    OpenAI Evolution Strategy.

    Parameters:
        optimizer (Optimizer): Parameter optimizer.
        sigma (float): Standard deviation for sampling a population of solution
            candidates (default: 0.1).
        sigma_decay (float): Decay rate of the standard deviation of the sample
            distribution (default: 0.001).
        lr_decay (float): Decay rate of the learning rate (default: 0.001).
        antithetic (bool): If set to True, use antithetic sampling (default: False).
        fitness_shaping (bool): If set to True, use rank-based fitness shaping
            (default: True).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        sigma: float = 0.1,
        sigma_decay: float = 0.0,
        lr_decay: float = 0.0,
        antithetic: bool = False,
        fitness_shaping: bool = True,
    ) -> None:
        self.optimizer = optimizer
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.lr_decay = lr_decay
        self.antithetic = antithetic
        self.fitness_shaping = fitness_shaping
        self.epsilon = None

    def sample(self, size: int) -> np.ndarray:
        """
        Mirror-sample a population of solution candidates with the argument size, and
        store the sampled noise for computing gradient in `update()`.

        Parameters:
            size (int): Number of solution candidates; this value must be divisible by
                2 for mirror-sampling.

        Returns:
            A NumPy array of solution candidates.
        """
        num_params = len(self.optimizer.theta)
        dtype = self.optimizer.theta.dtype
        if self.antithetic:
            if size % 2 != 0:
                raise ValueError("sample size must be divisible by 2")
            eps_split = np.random.randn(size, num_params).astype(np.float32)
            self.epsilon = np.concatenate([eps_split, -eps_split], axis=0)
        else:
            self.epsilon = np.random.randn(size, num_params).astype(np.float32)
        return self.optimizer.theta + self.sigma * self.epsilon

    def update(self, fitness: np.ndarray) -> None:
        """
        Update the mean parameter vector given the fitness vector that corresponds to
        each of the noise vector created by `sample()`. Update the standard deviation
        of the sample distribution.

        Parameters:
            fitness (np.ndarray): Fitness vector whose values correspond to the fitness
                of each noise vector acquired in `sample()`.

        Raises:
            ValueError if the size of the fitness vector doesn't match that of the
            current solution candidates.
        """
        if len(fitness) != self.epsilon.shape[0]:
            raise ValueError("invalid size of the fitness vector")

        # Rank-based fitness shaping
        if self.fitness_shaping:
            rank = np.empty_like(fitness, dtype=int)
            rank[np.argsort(fitness)] = np.arange(len(fitness))
            fitness = rank.astype(fitness.dtype) / (len(fitness) - 1) - 0.5
            fitness = (fitness - np.mean(fitness)) / (np.std(fitness) + 1e-8)

        # Update theta
        # Note that the gradient vector is negated, as we assume gradient ascent while
        # using gradient descent optimizers
        grad = -1 / (len(fitness) * self.sigma) * (self.epsilon.T @ fitness)
        self.optimizer.update(grad)

        # Update sigma
        self.sigma = (1 - self.sigma_decay) * self.sigma

        # Update the learning rate
        self.optimizer.lr = (1 - self.lr_decay) * self.optimizer.lr
