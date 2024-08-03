import numpy as np


class Loss:
    def __init__(self):
        self.prediction = None
        self.target = None
        self.loss = None

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        return self.forward(prediction, target)

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class CrossEntropy(Loss):
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        self.prediction = prediction
        self.target = target
        # Compute and return the loss
        self.loss = np.mean(- np.log(np.sum(prediction * target, axis=1, keepdims=True)))
        return self.loss

    def backward(self) -> np.ndarray:
        grad = - self.target / (self.prediction * self.target.shape[0])
        return grad
