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

import random

class Evolution(object):
    def __init__(self, entry_data, target):
        self.target = target
        self.entry_data = entry_data
        self.input_wages_length = entry_data.shape[1] * target.shape[1]
        self.chromosom_size = (entry_data.shape[1]+1) * target.shape[1]
        self.max_differences = self.target.shape[0] * self.target.shape[1]
        self.chromosoms_list = np.array(
            [Chromosom(wages=np.array([random.choice(np.arange(-10.0, 10.0, 0.5)) for _ in range(self.chromosom_size)]), diff=self.max_differences)
             for _ in range(100)])

    def start_evolution(self, number_of_generations):
        self.differeces_list = np.array([self.max_differences for _ in range(number_of_generations)])
        self.average_differences_list = np.array([self.max_differences for _ in range(number_of_generations)])
        for i in range(number_of_generations):
            # ---1GENERATION---
            self.count_differences()
            # print(self.chromosoms_list[i].diff)

            self.chromosoms_list = sorted(self.chromosoms_list, key=lambda x: x.diff)
            # print(self.chromosoms_list[0].diff)
            self.differeces_list[i] = self.chromosoms_list[0].diff
            self.average_differences_list[i] = sum([x.diff for x in self.chromosoms_list]) / len(self.chromosoms_list)
            if self.chromosoms_list[0].diff == 0:
                return self.chromosoms_list[0]
            # print(self.chromosoms_list)
            self.next_chromosoms_generation = np.array(
                [Chromosom(wages=np.array([0 for _ in range(self.chromosom_size)]), diff=self.max_differences) for _ in range(100)])
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
                        chrom.wages[i] = random.choice(np.arange(-10.0, 10.0, 0.5))
            # print(self.next_chromosoms_generation[:10])
            self.chromosoms_list = self.next_chromosoms_generation

        self.count_differences()
        self.chromosoms_list = sorted(self.chromosoms_list, key=lambda x: x.diff)
        # print(self.chromosoms_list[0].text)
        return self.chromosoms_list[0]

    def count_differences(self):
        for i in range(100):
            self.chromosoms_list[i].diff = self.predict(self.entry_data, self.chromosoms_list[i])

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
            # print(output)
            counter += np.sum(output == self.target[j])
        return counter

import matplotlib.pyplot as plt
def show(X):
    for i in range(len(X)):
        matrix = np.reshape(X[i], (7, 5))
        matrix = np.where(matrix == -1, 0, matrix)
        plt.subplot(2, 5, i+1)
        plt.imshow(matrix, cmap="Greys")
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

import pandas as pd
# s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
# df = pd.read_csv(s, header=None, encoding='utf-8')

df = pd.read_csv('letters.data',
                 header=None,
                 encoding='utf-8')

X = df.iloc[[1, 5, 8, 9, 10, 11, 13, 14, 17, 21], :35].values
y = np.array([[-1 for _ in range(10)] for _ in range(10)])
np.fill_diagonal(y, 1)


show(X)
evo = Evolution(X, y)
generations = 150
best_chromoseme = evo.start_evolution(generations)

plt.plot(range(generations), evo.differeces_list)
plt.xlabel('Generations')
plt.ylabel('Errors')
plt.show()

plt.plot(range(generations), evo.average_differences_list)
plt.xlabel('Generations')
plt.ylabel('Average errors')
plt.show()

print(evo.predict(X, best_chromoseme))

damaged5 = damage(X, 5)
damaged15 = damage(X, 15)
damaged40 = damage(X, 40)

print(evo.predict(damaged5, best_chromoseme))
print(evo.predict(damaged15, best_chromoseme))
print(evo.predict(damaged40, best_chromoseme))