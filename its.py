from scipy.stats.sampling import NumericalInversePolynomial
from matplotlib import pyplot as plt
from abc import ABC, abstractmethod
import numpy as np
import sys

INF = sys.maxsize


class Distribution(ABC):
    '''
    Base class for distributions.
    Each child class should implement below items
    - `pdf` method which indicates the <probability density function>
    - `support` method which determines the domain of `pdf`
    '''

    @abstractmethod
    def pdf(self, x: float) -> float: return

    def support(self):
        return -INF, INF


class Exponential(Distribution):
    def __init__(self, landa):
        self.landa = landa

    def pdf(self, x):
        return self.landa * np.exp(-self.landa * x)

    def support(self, k=8):
        return 0, k / self.landa


class Uniform(Distribution):
    def __init__(self, a, b):
        assert a != b

        if a > b:
            a, b = b, a

        self.a = a
        self.b = b

    @property
    def d(self):
        return self.b - self.a

    def pdf(self, x):
        return 1 / self.d

    def support(self):
        return self.a, self.b


def monte_carlo(dist: Distribution, N=1000):
    samples = []
    n = 0

    while (d := N-n) > 0:
        R = np.random.uniform(0, 1, size=d)
        X = np.random.uniform(*dist.support(), size=d)
        Y = np.array(list(map(dist.pdf, X)))
        ceil = np.max(Y) * 1.5
        mask = (R * ceil) < Y
        samples.append(X[mask])
        n += mask.sum()

    return np.concatenate(samples)


def probability_transform(array, **kwargs):
    total = np.trapz(array, **kwargs)
    return np.divide(array, total)



if __name__ == '__main__':
    distribution = Exponential(1/2)
    N = 10_000

    sample_mc = monte_carlo(distribution, N)

    generator = NumericalInversePolynomial(distribution)
    sample_poly = generator.rvs(N)

    x_min = min(np.min(sample_mc), np.min(sample_poly))
    x_max = max(np.max(sample_mc), np.max(sample_poly))
    X = np.linspace(x_min, x_max, 1000)
    Y = np.array(list(map(distribution.pdf, X)))
    Y = probability_transform(Y, x=X)

    plt.plot(X, Y, color='red')
    plt.hist(sample_mc, density=True, bins=100, color='blue', alpha=0.4, label='Monte Carlo')
    plt.hist(sample_poly, density=True, bins=100, color='green', alpha=0.4, label='Inverse Poly')
    plt.tight_layout()
    plt.grid(True)
    plt.legend()
    plt.show()
