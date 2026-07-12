from functools import partial
from scipy import stats
import pandas as pd
import numpy as np

def discretize_manual(array, break_points: list[int|float]):
    """𝛗(x)=i; b_{i-1}<x<=b_i"""
    return np.sum([np.greater(array, point).astype(int) for point in break_points], axis=0)

def discretize_equal_width(array, bins=5):
    """divide numbers into equal-width intervals"""
    return pd.cut(array, bins=bins, include_lowest=True).codes

def discretize_equal_freq(array, bins=5):
    """divide numebrs into bins with approximately equal frequencies"""
    return pd.qcut(array, q=bins).codes

def discretize_distribution(array, bins=5, dist=stats.norm):
    """
    divide numbers based on a statistical distribution quantiles.
    usefule if we know the underlying distribution of taken sample.
    """
    quantiles = np.arange(1, bins) / bins
    points = dist.ppf(quantiles)
    return discretize_manual(array, points)

def discretize_jenks(array, bins=5):
    """
    Jenks natural breaks optimization.
    Minimize within-group and maximize between-group variance.
    """

    sorted_array = np.sort(array)
    n = sorted_array.shape[0]

    SS = np.zeros((n, n))  # Sum of Squared deviations matrix
    for i in range(n):
        for j in range(i+1, n):
            subarray = sorted_array[i:j+1]
            SS[i, j] = np.sum(np.square(subarray - np.mean(subarray)))

    DP = np.full((n, bins), np.inf)  # Dynamic Programming
    BP = np.zeros((n, bins), dtype=int)  # Break Points

    for i in range(n):
        DP[i, 0] = SS[0, i]

    for k in range(1, bins):
        for i in range(k, n):
            for j in range(k-1, i):
                cost = DP[j, k-1] + SS[j+1, i]
                if cost < DP[i, k]:
                    DP[i, k] = cost
                    BP[i, k] = j

    breaks = {sorted_array[0]-1, sorted_array[-1]}
    k = bins - 1
    i = n - 1
    while k > 0:
        j = BP[i, k]
        i = j - 1
        k -= 1
        breaks.add(sorted_array[j])
    breaks = sorted(breaks)

    SDAM = SS[0, n-1]
    SDCM = DP[n-1, bins-1]
    GVF = (SDAM - SDCM) / SDAM  # goodness of variance fit
    # GVF ranges between 0 (worst) and 1 (perfect fit)

    return pd.cut(array, breaks).codes

def loss(array, encoded):
    """information loss from discretization; entropy is useful too"""

    means = pd.DataFrame({
        'original': array,
        'encoded': encoded
    }).groupby('encoded')['original'].transform('mean')

    ss_total = np.sum(np.square(np.subtract(array, np.mean(array))))
    ss_residual = np.sum(np.square(np.subtract(array, means)))
    return ss_residual / ss_total


METHODS = {
    'Equal Width': discretize_equal_width,
    'Equal Frequency': discretize_equal_freq,
    'Normal Distribution': partial(discretize_distribution, dist=stats.norm),
    'Uniform Distribution': partial(discretize_distribution, dist=stats.uniform),
    'Jenks Natural Breaks': discretize_jenks
}

if __name__ == '__main__':
    X = np.random.random(10).round(2)
    k = 3

    print(X)
    for name, method in METHODS.items():
        encoded = method(X, k)
        print('-' * 25)
        print(name)
        print('Codes:', encoded)
        print('Loss:', loss(X, encoded))
