import numpy as np


class NeuralNetwork:
    def __init__(self, num_input_nodes, num_output_nodes, num_hidden_layers, nodes_per_hidden_layer):
        assert num_hidden_layers > 0, "Number of hidden layers must be more than 0."

        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        self.num_hidden_layers = num_hidden_layers
        self.nodes_per_hidden_layer = nodes_per_hidden_layer
        self.output_layer = np.zeros(num_output_nodes)
        self.input_layer = np.zeros(num_input_nodes)
        self.is_trained = True
        self.weights = []
        self.biases = []

        # set up weights: starting with the weights from the input layer to the first hidden layer
        self.weights.append(np.random.randn(self.nodes_per_hidden_layer, self.num_input_nodes))
        if self.num_hidden_layers > 1:
            for _ in range(self.num_hidden_layers):
                self.weights.append(np.random.randn(self.nodes_per_hidden_layer,
                                                    self.nodes_per_hidden_layer))
        # weights from the last hidden layer to the output layer
        self.weights.append(np.random.randn(num_output_nodes, nodes_per_hidden_layer))

        # set up biases
        self.biases.append(np.random.randn(nodes_per_hidden_layer))
        if self.num_hidden_layers > 1:
            for _ in range(self.num_hidden_layers):
                self.biases.append(np.random.randn(self.nodes_per_hidden_layer))
        self.biases.append(np.random.randn(num_output_nodes))

    def train(self, learning_rate, examples):
        pass

    def __backward_pass(self):
        pass

    def __forward_pass(self, x):
        a = [np.copy(x)]
        z = []
        for i in range(len(self.weights)):
            z.append(self.weights[i].dot(a[-1]) + self.biases[i])
            a.append(self.__sigmoid(z[-1]))
        return z, a

    def __back_prop(self, x):
        pass

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        assert self.is_trained, "Model is not trained. Please provide some input examples and run the train() function"
        _, y = self.__forward_pass(x)
        return y[-1]
