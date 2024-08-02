import numpy as np


class Optimizer:
    def __init__(self, lr: float) -> None:
        self.lr = lr

    def update(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class GradientDescent(Optimizer):
    def update(self, x: np.ndarray, dx: np.ndarray) -> np.ndarray:
        # Compute the new value for 'x' and return the result
        return x - self.lr * dx
