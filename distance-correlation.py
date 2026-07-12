import numpy as np

def distance_matrix(X):
    return np.abs(X[:, None] - X[None, :])

def align_matrix(X):  # double center
    return X - X.mean(axis=1) - X.mean(axis=0).reshape(-1, 1) + X.mean()

def distance_covariance(X, Y):
    A = align_matrix(distance_matrix(X))
    B = align_matrix(distance_matrix(Y))
    dot = np.sum(A.flatten() * B.flatten())
    n = len(X)
    return dot / n**2

# NOTE: also canonical-correlation can be used on distance matrices
def distance_correlation(X, Y):
    d_xx = distance_covariance(X, X)
    d_yy = distance_covariance(Y, Y)
    d_xy = distance_covariance(X, Y)
    return d_xy / np.sqrt(d_xx * d_yy)

def distance_correlation_efficient(X, Y):
    A = align_matrix(distance_matrix(X)).flatten()
    B = align_matrix(distance_matrix(Y)).flatten()
    return np.mean(A*B) / np.sqrt(np.mean(A*A) * np.mean(B*B))


if __name__ == '__main__':
    X = np.linspace(-1, 1, 101)
    Y = np.square(X) + np.random.randn(len(X))/5
    print(distance_correlation_efficient(X, Y))
    print(distance_correlation(X, Y))
