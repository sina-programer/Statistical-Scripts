from abc import ABC, abstractmethod
from typing import Callable, Any
from functools import wraps

import numpy as np


def attribute_required(attr, validator: Callable[[Any], bool] | None = None, exception=None):
    if validator is None or not callable(validator):
        validator = bool

    if exception is None or not issubclass(exception, Exception):
        exception = RuntimeError

    def decorator(function):

        @wraps(function)
        def wrapper(*args, **kwargs):
            self = args[0]
            if validator(getattr(self, attr, None)):
                return function(*args, **kwargs)
            raise exception(f"<{type(self).__name__}> object does not qualify <{attr}> for <{function.__name__}>.")

        return wrapper

    return decorator


class NotFittedError(Exception):
    '''raised when a transformer is used before fitting.'''


fit_required = attribute_required('fitted', exception=NotFittedError)

class BaseTransform(ABC):

    def __post_init__(self): pass

    def __init__(self, *args, **kwargs):
        self.__fitted = False
        self.__post_init__(*args, **kwargs)

    def fit(self, X):
        self._fit(X)
        self.__fitted = True
        return self

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def fitted(self):
        return self.__fitted

    @abstractmethod
    def _fit(self, X): return

    @abstractmethod
    def transform(self, X): return

    @abstractmethod
    def inverse_transform(self, X): return


class MinMaxTransform(BaseTransform):
    """Projects data onto the closed interval [0, 1]."""

    def __post_init__(self):
        self.minimum = None
        self.maximum = None

    def _fit(self, X):
        self.minimum = np.min(X)
        self.maximum = np.max(X)

    @fit_required
    def transform(self, X):
        return np.subtract(X, self.minimum) / (self.maximum - self.minimum)

    @fit_required
    def inverse_transform(self, X):
        return np.multiply(X, self.maximum-self.minimum) + self.minimum


class ABTransform(BaseTransform):
    """Projects data onto an arbitrary closed interval [low, high]."""

    def __post_init__(self, low=0.0, high=1.0):
        if low >= high:
            raise ValueError("low must be less than high")

        self.low = low
        self.high = high
        self._minmax = MinMaxTransform()

    def _fit(self, X):
        self._minmax.fit(X)

    @fit_required
    def transform(self, X):
        unit = self._minmax.transform(X)
        return unit * (self.high - self.low) + self.low

    @fit_required
    def inverse_transform(self, X):
        unit = np.subtract(X, self.low) / (self.high - self.low)
        return self._minmax.inverse_transform(unit)


class FisherZTransform(BaseTransform):

    def __post_init__(self, eps=1e-6):
        self.eps = float(eps)
        self._ab = ABTransform(low=-1.0, high=1.0)

    def _fit(self, X):
        self._ab.fit(X)

    @fit_required
    def transform(self, X):
        r = self._ab.transform(X) * (1 - self.eps)
        return np.log((1 + r) / (1 - r)) / 2  # np.arctanh

    @fit_required
    def inverse_transform(self, Z):
        e = np.exp(np.multiply(Z, 2))
        r = (e - 1) / (e + 1)  # np.tanh
        r /= (1 - self.eps)
        return self._ab.inverse_transform(r)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from scipy.stats import norm

    X = np.random.exponential(2, size=1000)

    transformer = FisherZTransform()
    Z = transformer.fit_transform(X)
    X_reconstructed = transformer.inverse_transform(Z)

    rmse = np.sqrt(np.mean(np.square(X - X_reconstructed)))
    print('RMSE of reconstruction:', rmse)

    props = dict(bins=100, density=True, alpha=0.5)
    plt.hist(X, label='original', **props)
    plt.hist(Z, label='transformed', **props)
    plt.hist(X_reconstructed, label='reconstructed', **props)

    span = np.linspace(np.min(Z), np.max(Z), 1001)
    plt.plot(span, norm.pdf(span, np.mean(Z), np.std(Z)), color='red', label='normal pdf')
    plt.legend()
    plt.show()
