from NeuralNetwork import NeuralNetwork
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import csv


def prepare_data():
    """

    Prepare the data for testing/training. Normalizes the input data and reshapes it to be accepted by our ANN, also
    performs one-hot-encoding on the output data for training.

    :return: 2 values, A tuple of the x training and y training, and a tuple of the x testing and y testing.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_validate, x_test = np.split(x_test, 2)
    y_validate, y_test = np.split(y_test, 2)

    # we need to build a new input vector from the 28x28 pixels, to one that is 1x784
    x_train = x_train.reshape((-1, 784))
    x_test = x_test.reshape((-1, 784))
    x_train = x_train.astype(float)
    x_test = x_test.astype(float)
    x_validate = x_validate.reshape((-1, 784))
    x_validate = x_validate.astype(float)
    # normalize to greyscale values between 0 and 1
    x_train /= 255
    x_test /= 255
    x_validate /= 255

    # convert output vectors to one-hot encoding
    num_classes = 10
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    y_validate = np_utils.to_categorical(y_validate, num_classes)
    return (x_train, y_train), (x_validate, y_validate), (x_test, y_test)


if __name__ == '__main__':
    (x_train, y_train), (x_validate, y_validate), (x_test, y_test) = prepare_data()

    hidden_layer_trials = [1, 5, 10]
    num_node_trials = [256, 512, 1024]

    LEARNING_RATE = 0.01
    EPOCHS = 50

    res = []
    headers = []

    for num_hidden_layers in hidden_layer_trials:
        for num_nodes in num_node_trials:
            print("--- NEXT TRIAL STARTING ---")
            print(f'Number of hidden nodes per layer : {num_nodes}, number of hidden layers: {num_hidden_layers}')
            nn = NeuralNetwork(num_input_nodes=784, num_output_nodes=10, num_hidden_layers=num_hidden_layers,
                               nodes_per_hidden_layer=num_nodes)
            loss, _ = nn.train(x_train, y_train, LEARNING_RATE, EPOCHS)
            accuracy = nn.evaluate(x_validate, y_validate)
            print("Training finished")
            print(f'Accuracy: {accuracy}')
            loss_label = f'{num_nodes} {num_hidden_layers} LOSS'
            accuracy_label = f'{num_nodes} {num_hidden_layers} ACCURACY'

            res.append({loss_label: loss, accuracy_label: accuracy})

            headers.append(loss_label)
            headers.append(accuracy_label)

        with open('res.csv', mode='w') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for data in res:
                writer.writerow(data)
