import numpy as np


class NeuralNetwork:
    def __init__(self, num_input_nodes, num_output_nodes, num_hidden_layers, nodes_per_hidden_layer):
        """
        :param num_input_nodes: The number of input nodes, should be the same as the number of features
        :param num_output_nodes: number of output nodes, also the number of classes
        :param num_hidden_layers: the number of hidden layers. Must be more than 0
        :param nodes_per_hidden_layer: Nodes per hidden layer
        """
        assert num_hidden_layers > 0, "Number of hidden layers must be more than 0."

        np.random.seed(42)

        self.__num_input_nodes = num_input_nodes
        self.__num_output_nodes = num_output_nodes
        self.__num_hidden_layers = num_hidden_layers
        self.__nodes_per_hidden_layer = nodes_per_hidden_layer
        self.__is_trained = False
        self.__weights = []
        self.__biases = []

        # set up weights: starting with the weights from the input layer to the first hidden layer
        self.__weights.append(np.random.randn(self.__nodes_per_hidden_layer, self.__num_input_nodes) * 0.01)
        if self.__num_hidden_layers > 1:
            for _ in range(self.__num_hidden_layers):
                self.__weights.append(np.random.randn(self.__nodes_per_hidden_layer,
                                                      self.__nodes_per_hidden_layer) * 0.01)
        # weights from the last hidden layer to the output layer
        self.__weights.append(np.random.randn(self.__num_output_nodes, self.__nodes_per_hidden_layer) * 0.01)
        # TODO biases
        # # set up biases
        # self.__biases.append(np.random.randn(self.__nodes_per_hidden_layer, 1) * 0.01)
        # if self.__num_hidden_layers > 1:
        #     for _ in range(self.__num_hidden_layers):
        #         self.__biases.append(np.random.randn(self.__nodes_per_hidden_layer, 1) * 0.01)
        # self.__biases.append(np.random.randn(self.__num_output_nodes, 1) * 0.01)

    def train(self, x_train, y_train, learning_rate, epochs):
        """

        Perform gradient descent on the loss (MSE) to optimize the weights of the neural network to predict values.

        :param epochs: Number of rounds of optimizing the loss to perform.
        :param x_train: A list of input vectors. Should have shape (number of training examples, num_input_nodes)
        :param y_train: The desired output for each input.
        :param learning_rate: The coefficient to multiply each gradient descent step by
        :return: An array of the loss over epochs, an array of the weights updated at each step.
        """

        loss_history = []
        weights_history = []
        biases_history = []

        # Perform gradient descent
        for epoch in range(epochs):
            epoch_loss = []
            for x, y in zip(x_train, y_train):
                z_hat, y_hat = self.__forward_pass(x)
                loss = self.__mean_squared_loss(np.array(y_hat[-1]).flatten(), y)
                epoch_loss.append(loss)

                gradients_w = self.__backward_pass(y_hat, z_hat, y)
                new_weights, new_biases = self.__update_weights(gradients_w, learning_rate)
                weights_history.append(new_weights)
                biases_history.append(new_biases)
            loss_history.append(epoch_loss[-1])
            print("Epoch: {0}, Most recent Loss: {1}".format(epoch, loss_history[-1]))


        self.__is_trained = True
        return loss_history, weights_history

    def predict(self, x):
        """

        Use a trained neural network to predict the output of a value x.

        :param x: The value you want to predict y for.
        :return: The predicted value. Should be of shape (num_output_nodes, 1)
        """
        assert self.__is_trained, "Model is not trained. Please provide some input examples and run the train() " \
                                  "function "
        _, y = self.__forward_pass(x)
        return np.array(y[-1]).flatten()  # this is probably not the way to do this

    def evaluate(self, x, y):
        correct = 0
        for example, target in zip(x, y):
            y_pred = self.predict(example)
            maxi = np.max(y_pred)
            max_index = np.where(y_pred == maxi)
            if target[max_index] == 1:
                correct += 1
        return correct / len(y)

    def __backward_pass(self, y_hat, z_hat, y_true):
        w_gradient = []
        predicted = y_hat[-1]
        # calculate the error for the output layer
        error = 2 * (predicted - y_true) / predicted.shape[0] * self.__relu_derivative(z_hat[-1])
        delta_error = np.outer(error, y_hat[-2])
        w_gradient.append(delta_error)
        for i in reversed(range(len(self.__weights) - 1)):
            d_w = np.dot(self.__weights[i + 1].T, error)
            error = d_w * self.__relu_derivative(z_hat[i])
            delta_error = np.outer(error, y_hat[i])
            w_gradient.append(delta_error)

        w_gradient.reverse()
        return w_gradient

    def __forward_pass(self, x):
        """

        Perform a forward pass on an input x.

        :param x: The value you want to calculate y for

        :return: the weighted sums on the linear layers, the weighted sums on the activated layers
        """
        # A : the activated layers
        a = [x]
        # Z : the linear layers
        z = []
        for i in range(len(self.__weights)):
            z.append(self.__weights[i].dot(a[-1]))
            a.append(self.__relu(z[-1]))
        return z, a

    def __update_weights(self, gradients_w, learning_rate):
        """

        Modify the weights of the neural network according to the backpropogation.

        :param gradients_w: An array representing the derivatives of the weights with respect to the loss.
        :param learning_rate: The coefficient to multiply these weights by.
        :return:
        """
        for i in range(len(self.__weights)):
            delta = learning_rate * gradients_w[i]
            self.__weights[i] -= delta

        return self.__weights, self.__biases

    @staticmethod
    def __mean_squared_loss(y_pred, y_true):
        return np.average((y_true - y_pred) ** 2, axis=0)

    @staticmethod
    def __relu(x):
        return np.maximum(0, x)

    @staticmethod
    def __relu_derivative(x):
        return np.heaviside(x, 0)
