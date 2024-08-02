import numpy as np

from numpy_nn.optim import Optimizer


class Layer:
    def __init__(self, inp_dim: int = None, out_dim: int = None) -> None:
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.inp = None
        self.out = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def step(self, optimizer: Optimizer) -> None:
        pass


class Linear(Layer):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(in_dim, out_dim)

        # Initialize the layer's weights and biases
        self.w = 0.10 * np.random.randn(in_dim, out_dim)
        self.b = np.zeros((1, out_dim))

        self.dw = None
        self.db = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Compute linear layer's output and save the value(s) required for the backward phase.
        self.inp = inp
        self.out = np.dot(inp, self.w) + self.b
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        # Calculate the gradient with respect to the weights and biases and save the results.
        self.dw = np.dot(self.inp.T, up_grad)
        self.db = np.sum(up_grad, axis=0, keepdims=True)
        down_grad = np.dot(up_grad, self.w.T)
        return down_grad

    def step(self, optimizer: Optimizer) -> None:
        # Update the layer's weights and biases
        self.w = optimizer.update(self.w, self.dw)
        self.b = optimizer.update(self.b, self.db)


class ReLU(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Write the forward pass for ReLU and save the value(s) required for the backward pass.
        self.inp = inp
        self.out = np.maximum(0, inp)
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        down_grad = up_grad.copy()
        down_grad[self.inp <= 0] = 0
        return down_grad


class Sigmoid(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.out = 1 / (1 + np.exp(-inp))
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        down_grad = self.out * (1 - self.out) * up_grad
        return down_grad


class Softmax(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.inp = inp
        exp_values = np.exp(inp - np.max(inp, axis=1, keepdims=True))
        self.out = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        down_grad = np.empty_like(up_grad)
        for index in range(up_grad.shape[0]):
            single_output, single_grad = self.out[index], up_grad[index]
            single_output = single_output.reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            down_grad[index] = np.dot(jacobian, single_grad)
        return down_grad
