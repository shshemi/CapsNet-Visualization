import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from keras.models import *
from keras.losses import *
from keras.metrics import *
from keras.layers import *
import keras.backend as K
from capsulelayers import *

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)


def generate_two_class_linear_dataset():
    x_1 = np.random.multivariate_normal([5, 0], [[1, 0], [0, 1]], 100)
    x_2 = np.random.multivariate_normal([-5, 0], [[1, 0], [0, 1]], 100)
    y = np.zeros((200, 2))
    y[:100, 0] = 1
    y[100:, 1] = 1
    return np.concatenate([x_1, x_2]), y


def generate_two_class_linear_dataset_asym():
    x_1 = np.random.multivariate_normal([5, 0], [[1, 0], [0, 1]], 100)
    x_2 = np.random.multivariate_normal([0, 5], [[1, 0], [0, 1]], 100)
    y = np.zeros((200, 2))
    y[:100, 0] = 1
    y[100:, 1] = 1
    return np.concatenate([x_1, x_2]), y


def generate_two_class_xor_dataset():
    x_11 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 100)
    x_12 = np.random.multivariate_normal([10, 10], [[1, 0], [0, 1]], 100)
    x_21 = np.random.multivariate_normal([0, 10], [[1, 0], [0, 1]], 100)
    x_22 = np.random.multivariate_normal([10, 0], [[1, 0], [0, 1]], 100)
    y = np.zeros((400, 2))
    y[:200, 0] = 1
    y[200:, 1] = 1
    return np.concatenate([x_11, x_12, x_21, x_22]), y


def generate_two_class_xor_centered_dataset():
    x_11 = np.random.multivariate_normal([-5, -5], [[1, 0], [0, 1]], 100)
    x_12 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 100)
    x_21 = np.random.multivariate_normal([-5, 5], [[1, 0], [0, 1]], 100)
    x_22 = np.random.multivariate_normal([5, -5], [[1, 0], [0, 1]], 100)
    y = np.zeros((400, 2))
    y[:200, 0] = 1
    y[200:, 1] = 1
    return np.concatenate([x_11, x_12, x_21, x_22]), y


def plot_dataset(X, Y, lims=None):
    plt.cla()
    if lims is None:
        plt.xlim(auto=True)
        plt.ylim(auto=True)
    else:
        plt.xlim(lims)
        plt.ylim(lims)
    plt.scatter(X[:, 0], X[:, 1], c=["red" if y.argmax() == 0 else "blue" for y in Y], cmap=plt.cm.Spectral)
    plt.show()


def plot_decision_boundary(pred_func, X, Y, lims=None):
    plt.cla()
    # Set min and max values and give it some padding
    if lims is None:
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    else:
        x_min, x_max = lims
        y_min, y_max = lims

    h = 0.1
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z[:, 0].reshape(xx.shape)
    # Z1 = np.round(Z1)
    Z2 = Z[:, 1].reshape(xx.shape)
    # Z2 = np.round(Z2)
    # Plot the contour and training examples
    fig = plt.figure(1)
    ax = fig.subplots(1, 2)

    ax[0].contourf(xx, yy, Z1, cmap="Reds")
    ax[0].scatter(X[:, 0], X[:, 1], c=["red" if y.argmax() == 0 else "blue" for y in Y], cmap=plt.cm.Spectral)

    ax[1].contourf(xx, yy, Z2, cmap="Blues")
    ax[1].scatter(X[:, 0], X[:, 1], c=["red" if y.argmax() == 0 else "blue" for y in Y], cmap=plt.cm.Spectral)


def generate_three_class_linear():
    x_1 = np.random.multivariate_normal([6, 6], [[1, 0], [0, 1]], 100)
    x_2 = np.random.multivariate_normal([-6, 6], [[1, 0], [0, 1]], 100)
    x_3 = np.random.multivariate_normal([0, -12], [[1, 0], [0, 1]], 100)
    y = np.zeros((300, 3))
    y[:100, 0] = 1
    y[100:200, 1] = 1
    y[200:, 2] = 1
    return np.concatenate([x_1, x_2, x_3]), y


def plot_three_class_dataset(x, y):
    plt.cla()
    plt.scatter(x[:, 0], x[:, 1], c=y.argmax(axis=-1))
    plt.show()


def plot_three_class_decision_boundary(pred_func, X, Y, lims=None):
    plt.cla()
    # Set min and max values and give it some padding
    if lims is None:
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    else:
        x_min, x_max = lims
        y_min, y_max = lims

    h = 0.1
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z[:, 0].reshape(xx.shape)
    # Z1 = np.round(Z1)
    Z2 = Z[:, 1].reshape(xx.shape)
    # Z2 = np.round(Z2)
    Z3 = Z[:, 2].reshape(xx.shape)
    # Plot the contour and training examples
    fig = plt.figure(1)
    ax = fig.subplots(2, 2)

    ax[0, 0].contourf(xx, yy, Z1, cmap="Purples")
    ax[0, 0].scatter(X[:, 0], X[:, 1], c=Y.argmax(axis=-1))

    ax[0, 1].contourf(xx, yy, Z2, cmap="Greens")
    ax[0, 1].scatter(X[:, 0], X[:, 1], c=Y.argmax(axis=-1))

    ax[1, 0].contourf(xx, yy, Z3, cmap="YlOrBr")
    ax[1, 0].scatter(X[:, 0], X[:, 1], c=Y.argmax(axis=-1))


if __name__ == "__main__":
    x, y = generate_three_class_linear()
    model = Sequential()
    model.add(Lambda(lambda x: K.expand_dims(x), input_shape=(2,)))
    model.add(CapsuleLayer(3, 3, 5))
    model.add(Length())
    model.compile("adam", mean_squared_error)
    model.fit([x], [y], epochs=256, verbose=0)
    plot_three_class_decision_boundary(model.predict, x, y)
