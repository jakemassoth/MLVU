from NeuralNetwork import NeuralNetwork
import matplotlib
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np


def prepare_data():
    """

    Prepare the data for testing/training. Normalizes the input data and reshapes it to be accepted by our ANN, also
    performs one-hot-encoding on the output data for training.

    :return: 2 values, A tuple of the x training and y training, and a tuple of the x testing and y testing.
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # we need to build a new input vector from the 28x28 pixels, to one that is 1x784
    x_train = x_train.reshape((-1, 784))
    x_test = x_test.reshape((-1, 784))
    x_train = x_train.astype(float)
    x_test = x_test.astype(float)
    # normalize to greyscale values between 0 and 1
    x_train /= 255
    x_test /= 255

    # convert output vectors to one-hot encoding
    num_classes = 10
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = prepare_data()

    nn = NeuralNetwork(num_input_nodes=784, num_output_nodes=10, num_hidden_layers=1, nodes_per_hidden_layer=1024)
    LEARNING_RATE = 0.01
    EPOCHS = 20
    loss, _ = nn.train(x_train, y_train, LEARNING_RATE, EPOCHS)
    x_2 = np.linspace(0, EPOCHS, len(loss))
    matplotlib.use("qt5agg")
    fig_2, ax_2 = plt.subplots()
    ax_2.plot(x_2, loss, label="loss")
    ax_2.set_xlabel('epochs')
    ax_2.set_ylabel('loss')
    ax_2.set_title("loss over epochs")
    ax_2.legend()

    plt.show()

