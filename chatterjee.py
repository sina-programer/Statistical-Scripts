import numpy as np

def chatterjee(x, y):
    n = np.shape(x)[0]
    y = np.take(y, np.argsort(x, kind='quicksort'))
    u, i, c = np.unique(y, return_inverse=True, return_counts=True)
    j = len(u)-i-1
    r = np.cumsum(c)[i]
    l = np.cumsum(c[::-1])[j]
    return 1 - n * np.abs(np.diff(r)).sum() / (2 * np.sum(l * (n-l)))


if __name__ == '__main__':
    X = np.linspace(-1, 1, 101)
    Y = np.square(X)

    print('Y = X^2')
    print('Chatterjee Correlation:', chatterjee(X, Y))
    print('Pearson Correlation   :', np.corrcoef(X, Y)[0, 1])
