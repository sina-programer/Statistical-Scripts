from scipy.optimize import leastsq
from functools import partial
import numpy as np

def loss(c, k=2):
    return np.abs((X-c)**k).sum()

X = np.random.randint(1, 10, size=100)

mean = np.mean(X)
median = np.median(X)
print('mean:', mean)
print('median:', median)

for o in range(1, 5):
    loss_func = partial(loss, k=o)

    print()
    print('order {}'.format(o).center(30, '-'))
    print('mean loss:', loss_func(mean))
    print('median loss:', loss_func(median))

    optimal = leastsq(loss_func, mean)[0][0]
    print('optimal loss:', loss_func(optimal), '({})'.format(optimal))
