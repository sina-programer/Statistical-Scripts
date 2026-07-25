import itertools as it
import numpy as np

series = list[float | int]


def loss(y1: series, y2: series):
    return np.square(
        np.subtract(
            y1,
            y2
        )
    ).mean()


def least_squared_error(X: series, Y: series, func, params_values):
    '''
    find the best parameters of <func> based on LSE method regarding func(X) = Y

    X: series of independent values
    Y: series of dependent values
    func: function to be fit on X to Y
    params_values: dictionary of parameters and their possible values to be explored
    '''

    losses = {}
    keys = list(params_values.keys())
    grid = list(map(list, params_values.values()))
    for params in it.product(*grid):
        YP = func(X, **dict(zip(keys, params)))
        losses[params] = loss(Y, YP)

    optimals = min(losses, key=losses.get)
    return dict(zip(keys, optimals)), losses[optimals]


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    def linear(x, a, b):
        return a + b*x

    n = 20
    a, b = 0.5, 1.5
    E = np.random.normal(0, 1/4, (n,))
    X = np.linspace(0, 1, n)
    Y = linear(X, a, b) + E

    P = {
        'a': [-1, -0.5, 0, 0.5, 1, 1.25],
        'b': [-1, -0.5, 0, 0.5, 1, 1.25, 1.5, 1.75, 2]
    }

    params, ls = least_squared_error(X, Y, linear, P)
    print(params)

    YP = linear(X, a=params['a'], b=params['b'])

    plt.scatter(X, Y)
    plt.plot(X, YP, color='red')
    plt.show()
