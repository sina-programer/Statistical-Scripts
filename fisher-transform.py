import numpy as np

def minmax_scale(array):
    maximum, minimum = np.max(array), np.min(array)
    return np.subtract(array, minimum) / (maximum - minimum)

def fisher_transform(array):
    series = 2*minmax_scale(array) - 1  # [-1, +1]
    series = np.clip(series, -0.999, 0.999)
    return np.log10((1+series) / (1-series))  # np.arctanh

def fisher_transform_inv(array, minimum=0, maximum=1):
    expon = np.exp(np.multiply(array, 2))
    series = (expon - 1) / (expon + 1)  # np.tanh
    series = (series + 1) / 2
    domain = maximum - minimum
    return series*domain + minimum


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from scipy.stats import norm

    X = np.random.exponential(2, size=1000)
    Y = fisher_transform(X)
    Z = fisher_transform_inv(Y, minimum=np.min(X), maximum=np.max(X))

    plt.hist(X, bins=100, density=True, alpha=0.8, label='original')
    plt.hist(Y, bins=100, density=True, alpha=0.8, label='transformed')
    plt.hist(Z, bins=100, density=True, alpha=0.6, label='detransformed')

    span = np.linspace(np.min(Y), np.max(Y), 101)
    plt.plot(span, norm.pdf(span, np.mean(Y), np.std(Y)), color='red', label='normal')

    plt.legend()
    plt.show()
