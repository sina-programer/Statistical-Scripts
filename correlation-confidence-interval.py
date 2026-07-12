from scipy import stats
import numpy as np

def _pearson_ci(r, n, alpha=0.05):
    z = np.log((1+r) / (1-r)) / 2
    se = 1 / np.sqrt(n-3)
    q = stats.norm(0, 1).ppf(1-alpha/2)
    d = q * se
    zl, zu = z-d, z+d
    rl, ru = np.tanh(zl), np.tanh(zu)
    return rl, ru

def pearson_ci(X, Y, alpha=0.05):
    r = np.corrcoef(X, Y)[0, 1]
    n = len(X)
    return r, _pearson_ci(r, n, alpha=alpha)


if __name__ == '__main__':
    n = 100
    n1 = n // 2
    n2 = n - n1
    U1 = np.random.random(n1)
    U2 = np.random.random(n2)
    U = np.concatenate([U1, U1])
    V = np.concatenate([U1, U2])
    X = -np.log(U)
    Y = -np.log(V)

    print(pearson_ci(X, Y))
    r = stats.pearsonr(X, Y)
    print(r)
    print(r.confidence_interval())
