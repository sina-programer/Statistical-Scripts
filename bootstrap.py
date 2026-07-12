import numpy as np

def bootstrap(array, k=1000):
    shape = np.shape(array)
    for _ in range(k):
        yield np.random.choice(array, shape, replace=True)


if __name__ == '__main__':
    n = 100
    arr = np.random.random(n)
    mu = np.mean(arr)
    print('Estimate of mu:', mu)
    print('Variance of Estimate ( theory  ):', np.var(arr)/n)

    # means = list(map(np.mean, bootstrap(arr)))
    means = [np.mean(x) for x in bootstrap(arr)]
    print('Variance of Estimate (bootstrap):', np.var(means))
