import numpy as np


__all__ = ["Optimizer", "SGD", "Adam"]


class Optimizer:
    """
    A base class for iterative gradient-based optimizers. This `Optimizer` class assumes
    global optimization, i.e., the entire parameter vector is optimized together.

    Implement `_step()` and optionally `reset()` to inherit from this class. Each time 
    `update()` is called, the output of the `_step()` is added to `self.theta`, i.e.,
    `Optimizer` implements gradient ascent by default.

    Note that `Optimizer` does not copy the argument parameter vector; it updates the
    parameter vector directly in-place.

    Parameters
    ----------
    theta
        Parameter vector as a 1-D NumPy array
    """

    def __init__(self, theta: np.ndarray) -> None:
        self.theta = theta
        self.iters = 0

    def reset(self) -> None:
        """
        Reset the state of the optimizer. Optional, but useful for optimizers with
        momentum, e.g., SGD with momentum, Adam, etc.
        """
        self.iters = 0

    def update(self, grad: np.ndarray) -> None:
        """
        Update the parameter vector by computing the next step from the argument
        gradient.

        Parameters
        ----------
        grad
            A gradient vector

        Raises
        ------
        ValueError when `grad` doesn't have the same shape as `self.theta`.
        """
        if grad.shape != self.theta.shape:
            raise ValueError(
                f"the argument gradient vector must have the same shape as the target"
                f"parameters: {grad.shape} != {self.theta.shape}"
            )
        self.iters += 1
        self.theta += self._step(grad)

    def _step(self, grad: np.ndarray) -> None:
        """
        Compute the next step vector, given the argument gradient vector. Implement this
        function to inherit from this class.

        Parameters
        ----------
        grad
            A gradient vector

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic gradient descent (SGD) optimizer with momentum.

    Parameters
    ----------
    theta
        Parameter vector as a 1-D NumPy array
    lr
        Learning rate
    momentum
        Momentum factor (default: 0.9)
    """

    def __init__(self, theta: np.ndarray, lr: float, momentum: float = 0.9) -> None:
        super().__init__(theta)
        self.lr = lr
        self.momentum = momentum
        self.v = np.zeros_like(theta)

    def reset(self) -> None:
        """
        Reset the state of the optimizer, including the momentum vector.
        """
        super().reset()
        self.v.fill(0.0)

    def _step(self, grad: np.ndarray) -> np.ndarray:
        """
        Update the momentum vector and compute the next step vector, given the argument
        gradient vector. Note that the step vector is negated in order to perform
        gradient descent.

        Parameters
        ----------
        grad
            A gradient vector

        Returns
        -------
        The next step vector for updating the parameters.
        """
        self.v = self.momentum * self.v + (1 - self.momentum) * grad
        return -self.lr * self.v


class Adam(Optimizer):
    """
    Adam optimizer.

    Parameters
    ----------
    theta
        Parameter vector as a 1-D NumPy array
    lr
        Learning rate
    beta1
        Beta coefficient for computing running average of gradient (default: 0.9)
    beta2
        Beta coefficient for computing running average of square of gradient
        (default: 0.999)
    epsilon
        Term added to the denominator to improve numerical stability (default: 1e-8)
    """

    def __init__(
        self,
        theta: np.ndarray,
        lr: float,
        beta1: float = 0.99,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(theta)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros_like(theta)
        self.v = np.zeros_like(theta)

    def reset(self) -> None:
        """
        Reset the state of the optimizer, including the running average of gradient
        and that of its square.
        """
        super().reset()
        self.m.fill(0)
        self.v.fill(0)

    def _step(self, grad: np.ndarray) -> np.ndarray:
        """
        Update the running averages of gradient and its square, and compute the next
        step vector, given the argument gradient vector. Note that the step vector is
        negated in order to perform gradient descent.

        Parameters
        ----------
        grad
            A gradient vector

        Returns
        -------
        The next step vector for updating the parameters.
        """
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_corr = 1 - self.beta1 ** self.iters
        v_corr = np.sqrt(1 - self.beta2 ** self.iters)
        lr = self.lr * v_corr / m_corr
        return -lr * self.m / (np.sqrt(self.v) + self.epsilon)
