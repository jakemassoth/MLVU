from NeuralNetwork import NeuralNetwork
import numpy as np

if __name__ == '__main__':
    nn = NeuralNetwork(9, 9, 100, 10)
    y = nn.predict([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(y)
