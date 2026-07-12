import numpy as np

__doc__ = "different methods to avoid catastrophic cancellation in computing variance and covariance"

def mean_deviation(array):
    return np.subtract(array, np.mean(array))

def naive_variance(series, sample=False):
    assert np.ndim(series) == 1
    n = np.shape(series)[0]
    s = np.sum(series)
    ss = np.square(series).sum()
    var = ss/n - (s/n)**2
    if sample:
        var *= n / (n-1)
    return var

def shifted_variance(series, sample=False):
    '''
    According to Var(X-k)=Var(X) for any constant k we can avoid cancellation
    by any k in the range of X; the closer k to mean, the better.
    '''

    assert np.ndim(series) == 1

    n = np.shape(series)[0]
    if n < 2:
        return 0.0

    k = series[0]
    s = np.subtract(series, k).sum()
    ss = np.square(np.subtract(series, k)).sum()
    var = (ss - s**2/n) / (n-1 if sample else n)
    return var

def two_pass_variance(series, sample=False):
    '''most stable method for variance computation'''
    assert np.ndim(series) == 1
    df = np.shape(series)[0]
    if sample:
        df -= 1
    return np.square(mean_deviation(series)).sum() / df

def naive_covariance(A, B, sample=False):
    assert np.ndim(A) == np.ndim(B) == 1
    n, m = np.shape(A)[0], np.shape(B)[0]
    assert n == m, n>0
    s1 = np.sum(A)
    s2 = np.sum(B)
    s12 = np.multiply(A, B).sum()
    return (s12 - s1*s2/n) / (n-1 if sample else n)

def shifted_covariance(A, B, sample=False):
    assert np.ndim(A) == np.ndim(B) == 1
    n, m = np.shape(A)[0], np.shape(B)[0]
    assert n == m, n>0

    k1 = A[0]
    k2 = B[0]

    d1 = np.subtract(A, k1)
    d2 = np.subtract(B, k2)
    d12 = d1 * d2

    s1 = d1.sum()
    s2 = d2.sum()
    s12 = d12.sum()

    return (s12 - s1*s2/n) / (n-1 if sample else n)

def two_pass_covariance(A, B, sample=False):
    assert np.ndim(A) == np.ndim(B) == 1
    n, m = np.shape(A)[0], np.shape(B)[0]
    assert n == m, n>0
    df = n-1 if sample else n
    return np.sum(mean_deviation(A) * mean_deviation(B)) / df


if __name__ == '__main__':
    array = np.random.random(size=1000)
    print(np.var(array, ddof=1))
    print(naive_variance(array, True))
    print(shifted_variance(array, True))
    print(two_pass_variance(array, True))
    print(two_pass_covariance(array, array, True))
