import numpy as np
import random

class Chromosom(object):
    def __init__(self, wages, diff):
        self.wages = wages
        self.diff = diff

    def __repr__(self):
        return self.wages + " : " + str(self.diff)

    def __str__(self):
        return self.wages + " : " + str(self.diff)


class Evolution(object):
    def __init__(self, target, entry_data):
        self.target = target
        self.entry_data = entry_data
        self.chromosom_size = 9
        # self.chars = string.ascii_letters + string.digits + string.punctuation
        self.chromosoms_list = np.array(
            [Chromosom(wages=np.array([random.choice(range(10)) for _ in range(self.chromosom_size)]), diff=450)
             for _ in range(100)])  # zmienić range

    def start_evolution(self, number_of_generations):
        for _ in range(number_of_generations):
            # ---1GENERATION---
            self.count_differences()
            # print(self.chromosoms_list[i].diff)

            self.chromosoms_list = sorted(self.chromosoms_list, key=lambda x: x.diff)
            # print(self.chromosoms_list)
            self.next_chromosoms_generation = np.array(
                [Chromosom(wages=np.array([0 for _ in self.chromosom_size]), diff=450) for _ in range(100)])
            self.next_chromosoms_generation[:10] = self.chromosoms_list[:10]
            for chrom in self.next_chromosoms_generation[10:]:
                # for each chromosome(after first 10)
                daddy = np.random.choice(self.chromosoms_list[:50])
                mommy = np.random.choice(self.chromosoms_list[:50])
                for i in range(self.chromosom_size):
                    # for each element
                    rand = np.random.random()
                    if rand < 0.45:
                        chrom.wages[i] = daddy.wages[i]
                    elif rand < 0.9:
                        chrom.wages[i] = mommy.wages[i]
                    else:
                        chrom.wages[i] = np.random.choice(range(10))  # zmienić range
            # print(self.next_chromosoms_generation[:10])
            self.chromosoms_list = self.next_chromosoms_generation

        self.count_differences()
        self.chromosoms_list = sorted(self.chromosoms_list, key=lambda x: x.diff)
        # print(self.chromosoms_list[0].text)

    def count_differences(self):
        for i in range(100):
            wages = self.chromosoms_list[i].wages
            counter = 0

            for j in range(150):
                enter = self.entry_data[j]  # - dane wejściowe
                # enter należy podać na trzy neurony
                n1_input = np.array([enter[0] * wages[0], enter[1] * wages[1], wages[2]])
                n2_input = np.array([enter[0] * wages[3], enter[1] * wages[4], wages[5]])
                n3_input = np.array([enter[0] * wages[6], enter[1] * wages[7], wages[8]])
                n1_output = np.where(sum(n1_input) >= 0.0, 1, -1)  # zamień na if
                n2_output = np.where(sum(n2_input) >= 0.0, 1, -1)
                n3_output = np.where(sum(n3_input) >= 0.0, 1, -1)
                output = np.array([n1_output, n2_output, n3_output])
                # porównaj output z target i różnice dodaj do counter
                counter += np.sum(output == self.target[j])

            self.chromosoms_list[i].diff = counter

    # def predict(self, X):
    # return np.where(self.net_input(X) >= 0.0, 1, -1)

    # def number_of_differences(self, chromosom):
    #     counter = 0
    #     # 
    #     return counter

import pandas as pd
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(s, header=None, encoding='utf-8')

# y = df.iloc[0:, 4].values
# y = np.array([[1, -1, -1] for x in y if x == 'Iris-setosa' elif x == 'Iris-versicolor'])
y1 = np.array([[1, -1, -1] for _ in range(50)])
y2 = np.array([[-1, 1, -1] for _ in range(50)])
y3 = np.array([[-1, -1, 1] for _ in range(50)])
y = np.concatenate((y1, y2, y3))
# print(y)

X = df.iloc[0:, [0,2]].values
X[100:] = np.array([x+2 for x in X[100:]])

import matplotlib.pyplot as plt

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.scatter(X[100:, 0], X[100:, 1], color='green', marker='*', label='virginica')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()