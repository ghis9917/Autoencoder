import numpy as np
from NeuralNetwork import NeuralNetwork

NN = NeuralNetwork()

NN.train()

NN.forward_propagation([1, 0, 0, 0, 0, 0, 0, 0])
print(np.around(NN.outputs))

NN.forward_propagation([0, 1, 0, 0, 0, 0, 0, 0])
print(np.around(NN.outputs, decimals=3))

NN.forward_propagation([0, 0, 1, 0, 0, 0, 0, 0])
print(np.around(NN.outputs, decimals=3))

NN.forward_propagation([0, 0, 0, 1, 0, 0, 0, 0])
print(np.around(NN.outputs, decimals=3))

NN.forward_propagation([0, 0, 0, 0, 1, 0, 0, 0])
print(np.around(NN.outputs, decimals=3))

NN.forward_propagation([0, 0, 0, 0, 0, 1, 0, 0])
print(np.around(NN.outputs, decimals=3))

NN.forward_propagation([0, 0, 0, 0, 0, 0, 1, 0])
print(np.around(NN.outputs, decimals=3))

NN.forward_propagation([0, 0, 0, 0, 0, 0, 0, 1])
print(np.around(NN.outputs, decimals=3))
