import numpy as np
from tqdm import trange
from numpy_nn.layers import Layer
from numpy_nn.loss import Loss
from numpy_nn.optim import Optimizer


class MLP:
    def __init__(self, layers: list[Layer], loss_fn: Loss, optimizer: Optimizer) -> None:
        self.layers = layers
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Pass `inp` to all the layers sequentially and return the result.
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp

    def loss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        loss = self.loss_fn.forward(prediction, target)
        return loss

    def backward(self):
        # Start with loss function's gradient and do the backward pass on all the layers.
        up_grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            up_grad = layer.backward(up_grad)

    def update(self):
        for layer in self.layers:
            layer.step(self.optimizer)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int) -> np.ndarray:
        losses = np.empty(epochs)
        for epoch in (pbar := trange(epochs)):
            running_loss = 0.0
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                prediction = self.forward(x_batch)
                running_loss += self.loss(prediction, y_batch) * batch_size
                self.backward()
                self.update()
            running_loss /= len(x_train)
            pbar.set_description(f"Loss: {running_loss:.3f}")
            losses[epoch] = running_loss
        return losses
