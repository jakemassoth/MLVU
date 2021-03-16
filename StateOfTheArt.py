from test import prepare_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from keras.callbacks import History
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = prepare_data()

    hist = History()
    EPOCH = 20
    LEARNING_RATE = 0.01

    model = Sequential()
    model.add(Dense(128, input_dim=784, activation='sigmoid'))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid')) #or, activation='softmax'

    optimizer = Adam(lr=LEARNING_RATE)  # Adam uses stochastic gradient descent
    model.compile(optimizer=optimizer, loss='MSE', metrics=['accuracy'])
    model.summary()

    model.fit(x_train, y_train, epochs=EPOCH, callbacks=[hist])
    _, accuracy = model.evaluate(x_test, y_test)

    print(accuracy)

    y_loss = hist.history['loss']
    x_epoch = np.arange(0, EPOCH, 1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss value')
    plt.scatter(x_epoch, y_loss)
    plt.show()


