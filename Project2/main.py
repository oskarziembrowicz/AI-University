import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
import os

class Perceptron(object):
    def __init__(self, learning_parameter=0.01, number_of_iterations=50, random_seed=1):
        self.learning_parameter = learning_parameter
        self.number_of_iterations = number_of_iterations
        self.random_seed = random_seed
        self.number_of_errors = []

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_seed)
        self.wages = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        for _ in range(self.number_of_iterations):
            errors = 0
            for xi, target in zip(X, y):
                # print(xi)
                update = self.learning_parameter * (target - self.predict_function(xi))
                self.wages[1:] += update * xi
                self.wages[0] += update * 1
                # print(self.wages[0:])
                errors += int(update != 0.0)
            self.number_of_errors.append(errors)
        # for i in range(0,4):
        # print(self.predict_function(X[i]))
        return self

    def fit_manual(self, X, y, w):
        self.wages = w

        for _ in range(self.number_of_iterations):
            errors = 0
            for xi, target in zip(X, y):
                # print(xi)
                update = self.learning_parameter * (target - self.predict_function(xi))
                self.wages[1:] += update * xi
                self.wages[0] += update * 1
                # (self.wages[0:])
                errors += int(update != 0.0)
            self.number_of_errors.append(errors)
        return self

    def net_input(self, X):
        # print(np.dot(X, self.wages[1:]) + self.wages[0])
        return np.dot(X, self.wages[1:]) + self.wages[0]

    def predict_function(self, X):
        # print(np.where(self.net_input(X) >= 0.0, 1, -1))
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class SLP(object):
    def __init__(self, eta=0.05, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = 1
        # self.ppn = []

    def fit(self, X, y):
        #utworzyć 10 perceptronów - po jednym dla każdej litery
        self.ppn = np.zeros(10,10)
        for i in X:
            self.ppn[i] = Perceptron(self.eta, self.n_iter, self.random_state)
            self.ppn[i].fit(X[i], y[i]);
        # self.ppn = np.array(self.ppn)

    def predict(self, X):
        # array = np.zeros(shape=(10,10))
        for i in X:
            print(self.ppn[i].predict_function(X[i]))
        # return array

    # def misclassified(self, X):

    def show(self, X):
        for i in X:
            matrix = np.reshape(i, (7, 5))
            matrix = np.where(matrix == -1, 0, matrix)
            plt.imshow(matrix, cmap="Greys")
            # plt.show()
        # plt.tight_layout()
        plt.show()

# code: 1 5 8 9 10 11 13 14 17 21
net = SLP()

#uploading file in google collab
# from google.colab import files
#
# uploaded = files.upload()
# df = pd.read_csv(io.BytesIO(uploaded['letters.data']))

df = pd.read_csv('letters.data',
                 header=None,
                 encoding='utf-8')

X = df.iloc[[1,5,8,9,10,11,13,14,17,21], :35].values
y = df.iloc[[1,5,8,9,10,11,13,14,17,21], 35:].values

net.show(X)

net.fit(X, y)

# print(net.predict(X))

# %matplotlib inline
# import matplotlib.pyplot as plt
# import os

# y = df.iloc[0:100, 4].values
# y = np.where(y == 'Iris-setosa', -1, 1)
# X = df.iloc[0:100, [0, 2]].values

# plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc='upper left')
# plt.show()

# ppn = Perceptron(learning_parameter=0.1, number_of_iterations=10)
# ppn.fit(X, y)
# plt.plot(range(0, len(ppn.number_of_errors) ), ppn.number_of_errors, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of updates')
# plt.show()