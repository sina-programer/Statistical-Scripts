import numpy as np

# any non-negative function
def kernel(x):
    return np.exp(-np.square(x)/2) / np.sqrt(2*np.pi)

def get_estimator(X, kernel, h=0.5):
    assert h > 0

    n = len(X)
    denom = n * h

    def estimator(x):
        y = np.subtract(x, X) / h
        return np.sum(kernel(y)) / denom

    return estimator


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    X = np.random.exponential(size=1000)
    f_hat = get_estimator(X, kernel, h=0.4)  # bigger values of h makes the estimation more smooth

    t = np.arange(np.min(X), np.max(X), 0.01)
    y = np.array(list(map(f_hat, t)))
    plt.plot(t, y, color='red')

    plt.hist(X, density=True)
    plt.show()