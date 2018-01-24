#!/bin/bash/env python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import perceptron
import p1_4_a as dg       # data generator

d = dg.Data(n=1000)
p = perceptron.Perceptron(tol=1e-9)

X = d.frame[['x', 'y']]
y = d.frame['label']
f = p.fit(X.values, y)
f = p.fit(X.values, y)
predictions = p.predict(X)

print('iterations:',p.n_iter_)
print('accuracy', str(p.score(X.values, y.values)*100) + '%')

cls1 = X[y == 1.0]
cls2 = X[y == 0.0]

plt.scatter(cls1['y'].values, cls1['x'].values, marker='o')
plt.scatter(cls2['y'].values, cls2['x'].values, marker='x')
ymin, ymax = plt.ylim()
xmin, xmax = plt.xlim()
w = p.coef_[0]
a = -w[0] / w[1]
if (a < 1):
    xx = np.linspace(ymin, ymax)
    yy = a * xx - p.intercept_[0] / w[1]
else:
    yy = np.linspace(xmin, xmax)
    xx = (yy + p.intercept_[0] / w[1]) / a
plt.plot([-1,1], [-1,1], 'b-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original separation function in blue')
plt.show()
plt.plot([-1,1], [-1,1], 'b-')
plt.plot(yy, xx, 'k-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Machine separation in black, original in blue')
plt.scatter(cls1['y'].values, cls1['x'].values, marker='o')
plt.scatter(cls2['y'].values, cls2['x'].values, marker='x')
plt.show()
