def elasticity(x, *coefs):
    '''compute elasticity at point `x` for coefficients `C` in ascending order'''
    numerator = 0
    denomerator = 0
    for i, c in enumerate(coefs):
        denomerator += c * x ** i
        numerator += i * c * x ** i
    return numerator / denomerator

def elasticity_vectorized(x, *coefs):
    '''only works for scalar `x` values'''
    C = np.array(coefs)
    I = np.arange(len(coefs))
    A = C * x ** I
    return np.sum(I*A) / np.sum(A)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import numpy as np

    C = (a, b) = (10, 5)
    print(elasticity(3, *C))
    print(elasticity_vectorized(3, *C))

    mx = -a / b
    X = np.arange(0, np.abs(mx), 0.01) * np.sign(mx)

    plt.plot(X, elasticity(X, a, b))
    plt.grid()
    plt.show()
