from collections import namedtuple
from scipy import stats
import numpy as np

ChatterjeeResult = namedtuple('ChatterjeeResult', ['statistic', 'pvalue'])


def validate_1d_inputs(x, y):
    try:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError('x and y must be numeric vectors') from exc

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional.")

    if x.size != y.size:
        raise ValueError("x and y must have the same length.")

    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        raise ValueError("x and y must contain at least 2 valid observations.")

    return x, y


def randomized_order(x, *, random_state=None):
    '''Sort and return indices of input array by breaking ties at random.'''
    rng = np.random.default_rng(random_state)
    permutation = rng.permutation(np.size(x))
    order = permutation[np.argsort(x[permutation], kind='stable')]
    return order


def randomized_order_alt(x, *, random_state=None):
    rng = np.random.default_rng(random_state)
    random_order = rng.random(np.shape(x))
    return np.lexsort((random_order, x))


def bidirectional_ranks(array):
    """
    Returns two arrays as follow
        R_i = #{j : array_j <= array_i}
        L_i = #{j : array_j >= array_i}
    """
    u, i, c = np.unique(array, return_inverse=True, return_counts=True)
    j = len(u)-i-1
    r = np.cumsum(c)[i]
    l = np.cumsum(c[::-1])[j]
    return r, l


def bidirectional_ranks_alt(array):
    array_sorted = np.sort(array)
    r = np.searchsorted(array_sorted, array, side='right')
    l = np.size(array) - np.searchsorted(array_sorted, array, side='left')
    return r.astype(int), l.astype(int)


def null_variance(r, l):
    """
    Estimate tau^2 in `sqrt(n) * xi_n -> N(0, tau^2)`.
    `r` and `l` are bidirectional-ranks regarding to X and Y.
    under null hypothesis: X independent of Y.
    For continuous Y it converges to 2/5 (0.4).
    """

    n = np.size(r)
    u = np.sort(r).astype(float)
    v = np.cumsum(u)
    i = np.arange(n) + 1.0
    w = 2*n - 2*i + 1.0

    a = np.sum(w * np.square(u)) / n**4
    b = np.sum(np.square(v + (n - i) * u)) / n**5
    c = np.sum(w * u) / n**3
    d = np.sum(l * (n-l)) / n**3

    tau2 = (a - 2*b + c) / (d**2)

    if tau2 <= 0:
        raise ValueError("Estimated null variance is not positive.")

    return float(tau2)


def point_estimate(r, l):
    numerator = n * np.abs(np.diff(r)).sum()
    denominator = 2 * np.sum(l * (n-l))

    if denominator == 0:
        raise ValueError('Degenerated Y: the denominator of chatterjee coefficient is zero!')

    return float(1 - numerator / denominator)


def chatterjee_xi(x, y, *, random_state=None):
    """
    Chatterjee's rank correlation coefficient ξ_n(X, Y).

    Parameters
    ----------
    x, y : array-like
        One-dimensional observations of equal length.
    random_state : int, np.random.Generator, or None
        Random state used when ties occur in X.

    Returns
    -------
    float
        Chatterjee's directional correlation ξ_n(X, Y).

    Notes
    -----
    ξ_n(X, Y) measures how well Y can be regarded as a function of X.

    It is directional:
        chatterjee(x, y) != chatterjee(y, x)

    The implementation follows equations (1.1) and
    the tied-data definition in Chatterjee (2020).
    """

    x, y = validate_1d_inputs(x, y)
    n = np.size(x)

    if np.all(y == y[0]):
        raise ValueError('Y must not be constant!')

    indices = randomized_order(x, random_state=random_state)
    x, y = x[indices], y[indices]
    r, l = bidirectional_ranks(y)

    xi = point_estimate(r, l)
    tau2 = null_variance(r, l)

    zstatistic = float(np.sqrt(n) * xi / np.sqrt(tau2))
    pvalue = float(stats.norm.sf(zstatistic))

    return ChatterjeeResult(xi, pvalue)


if __name__ == '__main__':
    n = 1000
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, n)
    Y = np.sin(8 * np.pi * X) + rng.normal(size=n)/4

    r = stats.pearsonr(X, Y)
    xi = chatterjee_xi(X, Y)

    print('Chatterjee:', xi)
    print('Pearson   :', r)
