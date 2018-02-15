#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import perceptron
from sklearn.linear_model import LinearRegression

class SemiCircle(object):
    def __init__(self, sep=5, rad=10, thickness=5):
        self._sep = sep
        self._rad = rad
        self._thk = thickness

    def generate_data(self, n=2000):
        r=np.random.rand(n)
        theta=np.random.rand(n)
        r = self._rad + self._thk * r
        theta = theta * 2 * np.pi
        print('r:')
        print(r)
        print('theta:')
        print(theta)
        x = np.cos(theta)*r
        y = np.sin(theta)*r
        print('pts:')
        print(zip(x,y))
        c=np.zeros(n)
        for i in range(n):
            if y[i] < 0:
                y[i] = y[i] - self._sep
                x[i] = x[i] + self._rad + 0.5*self._thk
                c[i] = 1
        pts = np.array([x,y])
        print(pts)
        return (x,y,c)

s = SemiCircle(sep=-5)
x,y,c = s.generate_data(3000)
plt.scatter(x, y, c=c, marker='.')
plt.axis('equal')
plt.show()

X = pd.DataFrame({'x':x, 'y':y})
y = pd.Series(c)

p = perceptron.Perceptron(max_iter=1)
f = p.fit(X.values, y)
w_hat = p.coef_
n = 1000
scores = np.zeros(n)
best_scores = np.zeros(n)
scores[0] = 1.0 - p.score(X.values, y.values)
best_score = scores[0]
best_scores[0] = best_score
for i in range(1,n):
    p.fit(X.values, y.values, coef_init=p.coef_, intercept_init=p.intercept_)
    scores[i] = 1.0 - p.score(X.values, y.values)
    if scores[i] < best_score:
        w_hat = p.coef_
        best_score = scores[i]
        print('iteration:',i, ', best_score:', best_score)
    best_scores[i] = best_score

plt.scatter(X['x'], X['y'], c=y, marker='.')
plt.axis('equal')

plt.title(f'Best fit line in {n} iterations')
plt.xlabel('x')
plt.ylabel('y')
ymin, ymax = plt.ylim()
xmin, xmax = plt.xlim()
w = w_hat[0]
a = -w[0] / w[1]
if (a < 1):
    xx = np.linspace(ymin, ymax)
    yy = a * xx - p.intercept_[0] / w[1]
else:
    yy = np.linspace(xmin, xmax)
    xx = (yy + p.intercept_[0] / w[1]) / a
plt.plot(yy, xx, 'k-')
plt.show()

plt.plot(scores)
plt.title("Error scores by iteration")
plt.xlabel('iteration')
plt.ylabel('error rate')
plt.show()

plt.plot(best_scores)
plt.title("Best error scores by iteration")
plt.xlabel('iteration')
plt.ylabel('error rate')
plt.show()

lr = LinearRegression()
lr.fit(X.values, y.values)
plt.scatter(X['x'], X['y'], c=y, marker='.')
plt.axis('equal')
plt.title(f'Best fit linear regression line')
plt.xlabel('x')
plt.ylabel('y')
ymin, ymax = plt.ylim()
xmin, xmax = plt.xlim()
w = w_hat[0]
a = -w[0] / w[1]
if (a < 1):
    xx = np.linspace(ymin, ymax)
    yy = a * xx - p.intercept_[0] / w[1]
else:
    yy = np.linspace(xmin, xmax)
    xx = (yy + p.intercept_[0] / w[1]) / a
plt.plot(yy, xx, 'k-')
plt.show()

