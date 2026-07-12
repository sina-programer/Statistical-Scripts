from scipy import stats, linalg
import numpy as np

def canonical_correlation(X: np.ndarray, Y: np.ndarray, regularization: float=0.0):
    '''
    Canonical Correlation Analysis
    extract canonical pairs for two sets of variables so they are maximal correlated.
    significance tests using Bartlett's chi-square approximation are conducted.
    the null hypothesis states that all canonical correlations onward are zero.
    variance decomposition and loadings are computable and returned.

    Parameters
    X: a 2D numpy array of float type (variables in columns)
    Y: a 2D numpy array of float type
    regularization: a float number for ill-conditioned covariance matrices to avoid singularity
    '''

    assert isinstance(X, np.ndarray) and X.dtype==float and X.ndim==2
    assert isinstance(Y, np.ndarray) and Y.dtype==float and Y.ndim==2

    n, p = X.shape
    ny, q = Y.shape
    assert n == ny

    X_means = np.mean(X, axis=0)
    Y_means = np.mean(Y, axis=0)

    X -= X_means
    Y -= Y_means

    S_xx = (X.T @ X) / (n-1)
    S_yy = (Y.T @ Y) / (n-1)
    S_xy = (X.T @ Y) / (n-1)

    S_xx += np.eye(p) * regularization
    S_yy += np.eye(q) * regularization

    S_xx_inv = linalg.inv(S_xx)
    S_yy_inv = linalg.inv(S_yy)

    M = S_xx_inv @ S_xy @ S_yy_inv @ S_xy.T
    eigenvalues, eigenvectors = linalg.eig(M)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    indexes = np.argsort(np.abs(eigenvalues))[::-1]
    correlations = np.sqrt(np.maximum(eigenvalues[indexes], 0))
    A = eigenvectors[:, indexes]
    B = S_yy_inv @ S_xy.T @ A * np.where(np.isclose(correlations, 0), 0, 1/correlations)

    X_scores = X @ A
    Y_scores = Y @ B

    info = {}
    for i, ci in enumerate(correlations):
        wilks_lambda = np.prod(1 - correlations[i:]**2)
        dof = (p - i) * (q - i)
        if dof == 0:
            chi_square = 0.0
            pvalue = 1.0
        else:
            chi_square = -((n-1) - (p + q + 1)/2) * np.log(wilks_lambda)
            pvalue = 1 - stats.chi2(dof).cdf(chi_square)

        X_loadings = np.corrcoef(X_scores[:, i], X.T)[0, 1:]
        Y_loadings = np.corrcoef(Y_scores[:, i], Y.T)[0, 1:]

        info[i] = {
            'correlation': ci,
            'eigenvalue': ci**2,
            'wilks_lambda': wilks_lambda,
            'chi_square': chi_square,
            'dof': dof,
            'p_value': pvalue,
            'var_x': np.sum(X_loadings**2) / p,
            'var_y': np.sum(Y_loadings**2) / q
        }

    return info, A, B


if __name__ == "__main__":
    import pandas as pd

    n = 500
    p, q = 4, 5
    true_correlations = [0.9, 0.7, 0.4, 0.1]

    k = min(p, q)  # len(true_correlations)
    latent = np.random.randn(n, k)

    X = np.random.randn(n, p) / 2
    Y = np.random.randn(n, q) / 2

    for i, rho in enumerate(true_correlations):
        rho = np.sqrt(rho)

        x_loading = np.random.randn(1, p) * 2
        X += latent[:, [i]] @ x_loading * rho
    
        y_loading = np.random.randn(1, q) * 2
        Y += latent[:, [i]] @ y_loading * rho

    info, A, B = canonical_correlation(X, Y, regularization=0.001)
    print(pd.DataFrame(info))
