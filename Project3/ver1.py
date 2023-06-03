import numpy as np
import random

class Chromosome(object):
    def __init__(self, wages, diff):
        self.wages = wages
        self.diff = diff

    def __repr__(self):
        return self.wages + " : " + str(self.diff)

    def __str__(self):
        return self.wages + " : " + str(self.diff)

import random
class Evolution(object):
    def __init__(self, entry_data, target):
        self.target = target
        self.entry_data = entry_data
        self.input_wages_length = entry_data.shape[1] * target.shape[1]
        self.chromosome_size = (entry_data.shape[1]+1) * target.shape[1]
        self.max_differences = self.target.shape[0] * self.target.shape[1]
        self.chromosomes_list = np.array(
            [Chromosome(wages=np.array([random.choice(np.arange(-10.0, 10.0, 0.5)) for _ in range(self.chromosome_size)]), diff=self.max_differences)
             for _ in range(100)])

    def start_evolution(self, number_of_generations):
        self.differeces_list = np.array([self.max_differences for _ in range(number_of_generations)])
        self.average_differences_list = np.array([self.max_differences for _ in range(number_of_generations)])
        for i in range(number_of_generations):
            # ---1GENERATION---
            self.count_differences()

            self.chromosomes_list = sorted(self.chromosomes_list, key=lambda x: x.diff)
            self.differeces_list[i] = self.chromosomes_list[0].diff
            self.average_differences_list[i] = sum([x.diff for x in self.chromosomes_list]) / len(self.chromosomes_list)
            if self.chromosomes_list[0].diff == 0:
                return self.chromosomes_list[0]
            self.next_chromosomes_generation = np.array(
                [Chromosome(wages=np.array([0 for _ in range(self.chromosome_size)]), diff=self.max_differences) for _ in range(100)])
            self.next_chromosomes_generation[:10] = self.chromosomes_list[:10]
            for chrom in self.next_chromosomes_generation[10:]:
                # dla każdego chromosomu(po 10)
                daddy = np.random.choice(self.chromosomes_list[:50])
                mommy = np.random.choice(self.chromosomes_list[:50])
                for i in range(self.chromosome_size):
                    # dla każdego elementu
                    rand = np.random.random()
                    if rand < 0.45:
                        chrom.wages[i] = daddy.wages[i]
                    elif rand < 0.9:
                        chrom.wages[i] = mommy.wages[i]
                    else:
                        chrom.wages[i] = random.choice(np.arange(-10.0, 10.0, 0.5))
            self.chromosomes_list = self.next_chromosomes_generation

        self.count_differences()
        self.chromosomes_list = sorted(self.chromosomes_list, key=lambda x: x.diff)
        return self.chromosomes_list[0]

    def count_differences(self):
        for i in range(100):
            self.chromosomes_list[i].diff = self.predict(self.entry_data, self.chromosomes_list[i])

    def predict(self, input_data, chromosome):
        counter = 0
        for j in range(self.target.shape[0]):
            enter = input_data[j]  # - dane wejściowe
            output = np.zeros(self.target.shape[1]) # tablica o długości jednego rzędu danych wynikowych
            for i in range(len(output)):
                # dla każdego elementu y lub dla każdego zestawu danych z 1 chromosomu
                wages = chromosome.wages[i*len(enter):(1+i)*len(enter)]
                bias = chromosome.wages[self.input_wages_length+i]
                output[i] = 1 if (sum(wages * enter) + bias) >= 0 else -1
            counter += np.sum(output == self.target[j])
        return counter

import pandas as pd
s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(s, header=None, encoding='utf-8')

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

evo = Evolution(X, y)
generations = 150
# max ~300/400
best_chromosome = evo.start_evolution(generations)

plt.plot(range(generations), evo.differeces_list)
plt.xlabel('Generations')
plt.ylabel('Errors')
plt.show()

plt.plot(range(generations), evo.average_differences_list)
plt.xlabel('Generations')
plt.ylabel('Average errors')
plt.show()

print(evo.predict(X, best_chromosome))