import numpy as np

def row_reduction(matrix, verbose=False):
    matrix = np.array(matrix)
    coeffs = np.zeros_like(matrix)
    r, c = matrix.shape

    for i in range(1, r):
        for j in range(min(i, c)):
            pivot = matrix[i, j]
            k = -pivot / matrix[j, j]
            matrix[i] += k * matrix[j]
            coeffs[i, j] = k

            if verbose:
                print(matrix)

    return matrix, coeffs

def apply_coefficients(coeffs, array):
    array = np.array(array)
    r, c = coeffs.shape
    for i in range(r):
        for j in range(min(i, c)):
            array[i] += array[j]*coeffs[i, j]
    return array

def solve_equations(X, y):
    A, C = row_reduction(X)
    b = apply_coefficients(C, y)

    V = np.empty_like(b, dtype=float)
    for i in range(V.shape[0]-1, -1, -1):
        V[i] = (b[i] - A[i, i+1:]@V[i+1:]) / A[i, i]

    return V


if __name__ == '__main__':
    A = np.random.randint(1, 10, size=(3, 3)).astype(float)
    X = np.array([1, 2, -1])
    b = A @ X

    x = solve_equations(A, b)
    assert np.isclose(X, x).all()
