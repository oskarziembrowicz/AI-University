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
                update = self.learning_parameter * (target - self.predict_function(xi))
                self.wages[1:] += update * xi
                self.wages[0] += update * 1
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

    def fit(self, X, y):
        #utworzyć 10 perceptronów - po jednym dla każdej litery
        self.perceptrons = np.array([Perceptron(self.eta, self.n_iter, self.random_state) for _ in range(len(X))])
        self.errors_ = [0 for _ in range(len(X))]
        for i in range(len(X)):
            self.perceptrons[i].fit(X, y[i])
            self.errors_ = list(map(sum,zip(self.errors_, self.perceptrons[i].number_of_errors)))
            # print(self.errors_)

    def predict(self, X):
        result = np.array([[0 for _ in range(len(X))] for _ in range(len(X))])
        for i in range(len(X)):
            result[i] = self.perceptrons[i].predict_function(X)
            # print(self.perceptrons[i].predict_function(X))
        return result

    def misclassified(self, X, y):
        # compare results of predict and y
        counter = 0
        for i in range(len(X)):
            counter += np.sum(self.predict(X)[i] != y[i])
        return counter

    def show(self, X):
        for i in range(len(X)):
            matrix = np.reshape(X[i], (7, 5))
            matrix = np.where(matrix == -1, 0, matrix)
            plt.subplot(2, 5, i+1)
            plt.imshow(matrix, cmap="Greys")
            # plt.show()
        plt.show()

def damage(X, percent, seed=1):
    rgen = np.random.RandomState(seed)
    result = np.array(X)
    count = int(X.shape[1] * percent/100)

    for index_example in range(len(X)):
        order = np.sort(rgen.choice(X.shape[1], count, replace=False))
        for index_pixel in order:
            result[index_example][index_pixel] *= -1
    return result


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

# my set: 1,5,8,9,10,11,13,14,17,21
# test set: 10,11,12,13,14,15,16,17,18,19

X = df.iloc[[10,11,12,13,14,15,16,17,18,19], :35].values
# y = df.iloc[[1,5,8,9,10,11,13,14,17,21], 35:].values
y = np.array([[-1 for _ in range(10)] for _ in range(10)])
np.fill_diagonal(y, 1)

net.show(X)

net.fit(X, y)

print(net.predict(X))

print(net.errors_)

print(net.misclassified(X, y))

damaged5 = damage(X, 5)
damaged15 = damage(X, 15)
damaged40 = damage(X, 40)

# 5% damaged:
net.show(damaged5)
print(net.predict(damaged5))
print(net.misclassified(damaged5, y))

# 15% damaged:
net.show(damaged15)
print(net.predict(damaged15))
print(net.misclassified(damaged15, y))

# 40% damaged:
net.show(damaged40)
print(net.predict(damaged40))
print(net.misclassified(damaged40, y))