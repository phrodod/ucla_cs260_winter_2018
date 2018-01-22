#! /usr/bin/env python3

"""
Generate a linearly separable data set of size 20 as indicated in Exercise 1.4. Plot the examples {(x_n, y_n)} as well
as the target function f on a plane. Be sure to mark the examples from different classes differently and add labels to
the axes of the plot.
"""

import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set()

class Data(object):
    def __init__(self):
        self.gen_data()

    def gen_data(self):
        x = [random.uniform(-1, 1) for _ in range(20)]
        y = [random.uniform(-1, 1) for _ in range(len(x))]
        labels = [1 if x[i] < y[i] else 0 for i in range(len(x))]
        self._df = pd.DataFrame(data={'x':x, 'y':y, 'label':labels})

    def plot_data(self):
        markers = {0:'x', 1:'o'}
        for label in [0, 1]:
            d = self._df[self._df.label == label]
            plt.scatter(d.x, d.y, marker=markers[label])
        plt.plot([-1, 1], [-1, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Linearly separable data')
        plt.show()

if __name__ == '__main__':
    d = Data()
    d.plot_data()
