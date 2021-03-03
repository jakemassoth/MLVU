import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    def __init__(self, input_layer, output_layer, num_hidden_layers, nodes_per_hidden_layer):
        # These must be numpy arrays
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.num_hidden_layers = num_hidden_layers
        self.nodes_per_hidden_layer = nodes_per_hidden_layer
        self.weights = np.array([np.random.rand(self.input_layer.shape[1], self.nodes_per_hidden_layer),
                                 np.full(num_hidden_layers, np.random.rand(self.nodes_per_hidden_layer,
                                                                           self.nodes_per_hidden_layer)),
                                 np.random.rand(nodes_per_hidden_layer, output_layer.shape)])
        self.output_layer = np.zeros(output_layer.shape)
