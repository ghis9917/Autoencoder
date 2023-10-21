from typing import List

import numpy as np


class Layer:
    def __init__(self, n_neurons, n_inputs):
        self.n_neurons: int = n_neurons
        self.n_inputs: int = n_inputs
        self.weights = 0.1 * self.sigmoid(np.random.randn(n_neurons, n_inputs))
        self.biases = np.ones(n_neurons)

    def forward_prop(self, inputs) -> List[float]:
        self.output = self.sigmoid(np.dot(inputs, self.weights.T) + self.biases)
        return self.output

    def compute_errors(self, target) -> List[float]:
        self.errors = -(np.subtract(target, self.output))
        return self.errors

    def update_weights(self, w_d) -> None:
        self.weights = self.weights - w_d

    def update_biases(self, d) -> None:
        self.biases = self.biases - d

    @staticmethod
    def sigmoid(x) -> float:
        return 1 / (1 + np.exp(-x))
