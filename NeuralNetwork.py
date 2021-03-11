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
        self.is_trained = False
        self.weights = []
        self.biases = []

        # set up weights: starting with the weights from the input layer to the first hidden layer
        self.weights.append(np.random.randn(self.nodes_per_hidden_layer, self.num_input_nodes))
        if self.num_hidden_layers > 1:
            for _ in range(self.num_hidden_layers):
                self.weights.append(np.random.randn(self.nodes_per_hidden_layer,
                                                    self.nodes_per_hidden_layer))
        # weights from the last hidden layer to the output layer
        self.weights.append(np.random.randn(self.num_output_nodes, self.nodes_per_hidden_layer))

        # set up biases
        self.biases.append(np.random.randn(self.nodes_per_hidden_layer, 1))
        if self.num_hidden_layers > 1:
            for _ in range(self.num_hidden_layers):
                self.biases.append(np.random.randn(self.nodes_per_hidden_layer, 1))
        self.biases.append(np.random.randn(self.num_output_nodes, 1))

    def train(self, input_examples, output_examples, learning_rate):
        self.is_trained = True

    def __backward_pass(self):
        pass

    def __forward_pass(self, x):
        # A : the activated layers, should this be sigmoid???
        a = [self.__sigmoid(np.copy(x))]
        # Z : the linear layers
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
        return np.array(y[-1]).flatten()  # this is probably not the way to do this
