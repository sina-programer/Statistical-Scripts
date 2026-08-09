import numpy as np


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
    array_sorted = np.sort(array, kind='stable')
    r = np.searchsorted(array_sorted, array, side='right')
    l = np.size(array) - np.searchsorted(array_sorted, array, side='left')
    return r.astype(int), l.astype(int)


def chatterjee(x, y, *, random_state=None):
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
    numerator = n * np.abs(np.diff(r)).sum()
    denominator = 2 * np.sum(l * (n-l))

    if denominator == 0:
        raise ValueError('Degenerated Y: the denominator of chatterjee coefficient is zero!')

    xi = 1 - numerator / denominator
    return float(xi)


if __name__ == '__main__':
    n = 1000
    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, n)
    Y = np.sin(8 * np.pi * X) + rng.normal(size=n)/4

    print('Chatterjee Correlation:', chatterjee(X, Y))
    print('Pearson Correlation   :', np.corrcoef(X, Y)[0, 1])
