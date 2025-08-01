import numpy as np

class Perceptron(object):
    def __init__(self, learning_parameter=0.01, number_of_iterations=50, random_seed=1):
        self.learning_parameter = learning_parameter
        self.number_of_iterations = number_of_iterations
        self.random_seed = random_seed
        self.number_of_errors = []

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_seed)
        self.wages = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])

        self.saved_weights = np.zeros((self.number_of_iterations + 1, len(self.wages)))
        self.saved_weights[0, :] = self.wages

        for i in range(self.number_of_iterations):
            errors = 0
            for xi, target in zip(X, y):
                # print(xi)
                update = self.learning_parameter * (target - self.predict_function(xi))
                self.wages[1:] += update * xi
                self.wages[0] += update * 1
                # print(self.wages[0:])
                errors += int(update != 0.0)
            self.number_of_errors.append(errors)
            self.saved_weights[i + 1, :] = self.wages
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

# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#end of perceptron class
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==

#%matplotlib inline
import matplotlib.pyplot as plt
import os
import pandas as pd
# import numpy as np

s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
print('URL:', s)

df = pd.read_csv(s, header=None, encoding='utf-8')
#df.tail(5)

# y = df.iloc[0:100, 4].values
# y = np.where(y == 'Iris-setosa', -1, 1)
# X = df.iloc[0:100, [0, 2]].values

y = df.iloc[0:, 4].values
y1 = np.where(y == 'Iris-setosa', -1, 1)
y2 = np.where(y == 'Iris-versicolor', -1, 1)
y3 = np.where(y == 'Iris-virginica', -1, 1)
X = df.iloc[0:, [0,2]].values
X[100:] = np.array([x+2 for x in X[100:]])

#=================================================================================
#plot points
#=================================================================================

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.scatter(X[100:, 0], X[100:, 1], color='green', marker='*', label='virginica')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

# == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
#plot lines based on perceptrons
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==

# ppn = Perceptron(learning_parameter=0.1, number_of_iterations=10)
# ppn.fit(X, y)
# plt.plot(range(0, len(ppn.number_of_errors)), ppn.number_of_errors, marker='o')

ppn1 = Perceptron(learning_parameter=0.1, number_of_iterations=10)
ppn2 = Perceptron(learning_parameter=0.1, number_of_iterations=10)
ppn3 = Perceptron(learning_parameter=0.1, number_of_iterations=10)

ppn1.fit(X, y1)
ppn2.fit(X, y2)
ppn3.fit(X, y3)

plt.plot(range(0, len(ppn1.number_of_errors)), ppn1.number_of_errors, marker='o', label='setosa', color='red')
plt.plot(range(0, len(ppn2.number_of_errors)), ppn2.number_of_errors, marker='o', label='versicolor', color='blue')
plt.plot(range(0, len(ppn3.number_of_errors)), ppn3.number_of_errors, marker='o', label='virginica', color='green')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.legend(loc='lower left')
plt.show()

plt.plot(range(0, ppn1.saved_weights.shape[0]), ppn1.saved_weights[:,0], label="w0", color="red")
plt.plot(range(0, ppn1.saved_weights.shape[0]), ppn1.saved_weights[:,1], label="w1", color="green")
plt.plot(range(0, ppn1.saved_weights.shape[0]), ppn1.saved_weights[:,2], label="w2", color="blue")
# plt.plot(range(0, ppn1.saved_weights.shape[0]), ppn1.saved_weights[:,3], label="w3", color="yellow")
plt.xlabel("Changes")
plt.ylabel("Weights")
plt.legend(loc="lower left")
plt.show()