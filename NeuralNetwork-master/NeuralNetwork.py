from typing import List

from Layer import Layer
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    train_data = [[1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1]]

    learning_rate = 15
    weight_decay = 0
    training_iterations = 50000

    def __init__(self):
        self.network = []

        # Hidden Layer
        self.network.append(Layer(3, 8))

        # Output Layer
        self.network.append(Layer(8, 3))

    def forward_propagation(self, input: List[float]) -> None:
        inputs = input

        for layer in self.network:
            inputs = layer.forward_prop(inputs)

        # The last "inputs" are the actual outputs
        self.outputs = inputs

    def backward_propagation(self, input: List[float]) -> None:
        target = input
        errors = self.network[-1].compute_errors(target)

        for layer in reversed(range(len(self.network))):
            # d_sigmoid(activation value) * errors vector
            gradient = self.d_sigmoid(self.network[layer].output) * errors

            # Check if previous layer is the input layer
            if layer - 1 < 0:
                w_d = np.reshape(gradient, [3, 1]) * input
            else:
                w_d = np.reshape(gradient, [8, 1]) * np.reshape(self.network[layer - 1].output, [1, 3])

            # Update weights and biases
            self.network[layer].update_weights(self.learning_rate * (
                        (1 / self.network[layer].n_inputs) * w_d + self.weight_decay * self.network[layer].weights))
            self.network[layer].update_biases(self.learning_rate * (1 / self.network[layer].n_inputs) * gradient)

            # Backpropagate the errors for next iteration
            errors = np.dot(self.network[layer].weights.T, errors)

    def train(self) -> None:
        # Initialize list of cost per iteration used in the graph
        cost_per_example: List[List[float]] = [[], [], [], [], [], [], [], []]
        average_cost_per_iteration: List[float] = []

        # Training loop
        for i in range(self.training_iterations):
            for j in self.train_data:
                self.forward_propagation(j)
                self.backward_propagation(j)

            tot_cost: float = 0
            for j in range(len(self.train_data)):
                cost_j: float = self.cost(self.train_data[j])
                cost_per_example[j].append(cost_j)
                tot_cost += cost_j

            average_cost_per_iteration.append(tot_cost / len(self.train_data))

        self.plot_cost_per_example(cost_per_example)
        self.plot_average_cost(average_cost_per_iteration)

    def cost(self, input) -> float:
        self.forward_propagation(input)
        return float(np.sum(1 / 2 * np.square(self.network[len(self.network) - 1].compute_errors(input))))

    @staticmethod
    def plot_cost_per_example(self, cost_list: List[List[float]]) -> None:
        # Create and show cost graph
        for j in range(len(self.train_data)):
            plt.plot(range(len(cost_list[j])), cost_list[j])
        plt.show()

    @staticmethod
    def plot_average_cost(self, cost_list: List[float]) -> None:
        plt.plot(range(len(cost_list)), cost_list)
        plt.show()

    @staticmethod
    def d_sigmoid(x: float) -> float:
        return x * (1 - x)
