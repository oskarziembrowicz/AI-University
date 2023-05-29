import numpy as np
import random
import string


class Chromosom(object):
    def __init__(self, text, diff):
        self.text = text
        self.diff = diff

    def __repr__(self):
        return self.text + " : " + str(self.diff)

    def __str__(self):
        return self.text + " : " + str(self.diff)


class Evolution(object):
    def __init__(self, text=""):
        self.target = text
        self.chromosom_size = len(self.target)
        self.chars = string.ascii_letters + string.digits + string.punctuation
        self.chromosoms_list = np.array(
            [Chromosom(text=''.join(random.choice(self.chars) for _ in range(self.chromosom_size)), diff=100) for _ in
             range(100)])

    def start_evolution(self, number_of_generations):
        for _ in range(number_of_generations):
            # ---1GENERATION---
            self.count_differences()
            # print(self.chromosoms_list[i].diff)

            self.chromosoms_list = sorted(self.chromosoms_list, key=lambda x: x.diff)
            # print(self.chromosoms_list)
            self.next_chromosoms_generation = np.array([Chromosom(text="", diff=100) for _ in range(100)])
            self.next_chromosoms_generation[:10] = self.chromosoms_list[:10]
            for chrom in self.next_chromosoms_generation[10:]:
                # for each element(after first 10)
                daddy = np.random.choice(self.chromosoms_list[:50])
                mommy = np.random.choice(self.chromosoms_list[:50])
                for i in range(self.chromosom_size):
                    # for each letter
                    rand = np.random.random()
                    if rand < 0.45:
                        chrom.text += daddy.text[i]
                    elif rand < 0.9:
                        chrom.text += mommy.text[i]
                    else:
                        chrom.text += np.random.choice(list(self.chars))
            # print(self.next_chromosoms_generation)
            self.chromosoms_list = self.next_chromosoms_generation

        self.count_differences()
        print(self.chromosoms_list[0].text)

    def count_differences(self):
        for i in range(100):
            self.chromosoms_list[i].diff = self.number_of_differences(self.chromosoms_list[i])

    def number_of_differences(self, chromosom):
        counter = 0
        for i in range(self.chromosom_size):
            if chromosom.text[i] != self.target[i]:
                counter += 1
        return counter


evo = Evolution("OskarZiembrowicz")
evo.start_evolution(200)