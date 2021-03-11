from NeuralNetwork import NeuralNetwork
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    nn = NeuralNetwork(1, 1, 3, 10)
    x = np.linspace(-4, 4, 100)
    y = []
    y_true = np.sin(x)
    nn.train(x, y, 0.05)
    for x_i in x:
        y.append(nn.predict(x_i))

    matplotlib.use("qt5agg")
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x), label="sin(x)")
    ax.plot(x, y, label="nn predicted")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("does our nn suck?")
    ax.legend()

    plt.show()
