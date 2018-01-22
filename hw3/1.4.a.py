#! /usr/bin/env python3

"""
Generate a linearly separable data set of size 20 as indicated in Exercise 1.4. Plot the examples {(x_n, y_n)} as well
as the target function f on a plane. Be sure to mark the examples from different classes differently and add labels to
the axes of the plot.
"""

import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class Data(object):
    def __init__(self):
        self.gen_data()

    def gen_data(self):
        points = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(20)]
        labels = [1 if p[0] < p[1] else 0 for p in points]
        self._class1 = [points[i] for i in range(len(labels)) if labels[i] == 1]
        self._class2 = [points[i] for i in range(len(labels)) if labels[i] == 0]

    def plot_data(self):
        x = [i[0] for i in self._class1]
        y = [i[1] for i in self._class1]
        plt.scatter(x, y)
        x = [i[0] for i in self._class2]
        y = [i[1] for i in self._class2]
        plt.scatter(x, y, marker='x')
        plt.plot([-1, 1], [-1, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linearly separable data')
        plt.show()

if __name__ == '__main__':
    d = Data()
    d.plot_data()
