from functools import partial
import operator as op
import numpy as np


def length(array):
    assert np.ndim(array) == 1
    return np.shape(array)[0]

def all_positive(array):
    return all(map(partial(op.lt, 0), array))

def power(array, p):
    output = np.power(array, abs(p))
    if p < 0:
        output = np.divide(1, output)
    return output

def outliers(array, k=1.5, return_index=False):
    array = np.array(array)
    q1 = np.quantile(array, .25)
    q3 = np.quantile(array, .75)
    iq = q3 - q1
    iqk = iq * k
    mask = (array > q3+iqk) | (array < q1-iqk)
    if return_index:
        return np.where(mask)[0]
    return mask


def arithmetic(array):
    return np.sum(array) / length(array)

def weighted(array, weights=None):
    if weights is None:
        weights = np.ones_like(array)
    assert np.shape(array) == np.shape(weights)
    return np.sum(np.multiply(array, weights)) / np.sum(weights)

def geometric(array):
    assert all_positive(array)
    return np.power(np.prod(array), 1/length(array))

def harmonic(array):
    assert all_positive(array)
    return length(array) / np.sum(np.divide(1, array))

def contraharmonic(array):
    return np.sum(np.power(array, 2)) / np.sum(array)

def lehmer(array, p=1):
    return np.sum(power(array, p)) / np.sum(power(array, p-1))

def quadratic(array):  # RMS
    return np.sqrt(arithmetic(np.square(array)))

def generalized(array, p=1):
    return power(arithmetic(power(array, p)), 1/p)

def quasi(array, f, fi):
    return fi(arithmetic(list(map(f, array))))


def midrange(array):
    return (np.max(array) - np.min(array)) / 2

def median(array):
    array = np.sort(array)
    n = length(array)
    m = n // 2
    if n % 2:
        return array[m]
    return arithmetic(array[m-1:m+1])

def mode(array):
    return max(np.unique(array), key=list(array).count)

def trimmed_mean(array, k=3):
    k = min(k, (length(array)-1)//2)
    return arithmetic(np.sort(array)[k:-k])

def nonoutlier(array):
    mask = outliers(array)
    return arithmetic(array[~mask])

def winsorized(array):
    array = np.array(array)
    mask = outliers(array)
    valid_array = array[~mask]
    lower = np.min(valid_array)
    upper = np.max(valid_array)
    array[array > upper] = upper
    array[array < lower] = lower
    return arithmetic(array)


def weighted_(array, weights):
    assert np.shape(array) == np.shape(weights)
    ws = np.divide(weights, np.sum(weights))
    return np.sum(np.multiply(ws, array))

def harmonic_(a, b):
    return (2 * a * b) / (a + b)

def harmonic__(a, b):
    arr = (a, b)
    return np.square(geometric(arr)) / arithmetic(arr)



if __name__ == '__main__':
    arr = np.random.randint(1, 10, (10,))
    arr[-1] += 75
    print('Array:', arr)
    print()

    print('Arithmetic:', arithmetic(arr))
    print('Geometric:', geometric(arr))
    print('Harmonic:', harmonic(arr))
    print()

    print('Mid-Range:', midrange(arr))
    print('Median:', median(arr))
    print('Mode:', mode(arr))
    print('Trimmed-Mean:', trimmed_mean(arr))
    print('Non-Outlier:', nonoutlier(arr))
    print('Winsorized:', winsorized(arr))
    print()

    print('Contra Harmonic:', contraharmonic(arr))
    print('Quadratic:', quadratic(arr))
    print()

    print('Weighted (indexes as weights):', weighted(arr, range(len(arr))))
    print('Quasi_square_sqrt:', quasi(arr, np.square, np.sqrt))
    print('Quasi_sqrt_square:', quasi(arr, np.sqrt, np.square))
    print()

    for p in [0, .5, 1, 2, 3]:
        print(f"Lehmer_{p}:", lehmer(arr, p))
    print()

    for p in [1, 2, 3]:
        print(f"Generalized_{p}:", generalized(arr, p))
