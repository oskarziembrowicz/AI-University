import numpy as np
import random
import string


class Chromosom(object):
    def __init__(self, text, diff):
        self.text = text
        self.diff = diff


class Evolution(object):
    def __init__(self, text=""):
        self.target = text
        self.chromosom_size = len(self.target)
        self.chars = string.ascii_letters + string.digits
        self.chromosoms_list = np.array(
            [Chromosom(text=''.join(random.choice(self.chars) for _ in range(self.chromosom_size)), diff=100) for _ in
             range(100)])

        # for c in self.chromosoms_list:
        #   print(c.text)

        # print(self.chromosoms_list)

    def start_evolution(self):
        # ---GENERATION---
        for i in range(100):
            self.chromosoms_list[i].diff = self.number_of_differences(self.chromosoms_list[i])
            # print(self.chromosoms_list[i].diff)

    def number_of_differences(self, chromosom):
        counter = 0
        for i in range(self.chromosom_size):
            if chromosom.text[i] != self.target[i]:
                counter += 1
        return counter


evo = Evolution("OskarZiembrowicz")
evo.start_evolution()