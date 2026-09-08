import numpy as np


def variance_inflation_factor(X: np.ndarray, add_intercept: bool = True) -> np.ndarray:
    """
    Compute the Variance Inflation Factor (VIF) for each column of X.

    VIF_j = 1 / (1 - R^2_j)

    where R^2_j comes from regressing column j on all other columns.
    VIF_j > 10 typically signals problematic collinearity between predictor j and the rest.

    X should NOT already contain an intercept column;
    set add_intercept=False only if it does, to avoid a singular matrix.
    """

    n, p = X.shape
    intercept = np.ones((n, 1))
    vif = np.zeros(p)

    for j in range(p):
        y = X[:, j]
        x = np.delete(X, j, axis=1)
        if add_intercept:
            x = np.column_stack([intercept, x])

        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        residuals = y - x @ beta
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        vif[j] = np.inf if np.isclose(r2, 1) else 1 / (1 - r2)

    return vif


def condition_number(X: np.ndarray) -> float:
    """
    Compute the condition index of X based on its correlation matrix.

    kappa > 30 is generally considered problematic for multicollinearity.
    Note that the linear dependence measured by condition number may be
    more complicated than what can be captured by VIF scores per variable.
    """

    R = np.corrcoef(X, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(R)
    l1 = abs(np.max(eigenvalues))
    ln = abs(np.min(eigenvalues))
    if ln == 0:
        raise ValueError('Singular Matrix Error')

    return np.sqrt(l1 / ln)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200

    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)

    matrices = {
        'well-conditioned': np.column_stack([x1, x2, x3]),
        'ill-conditioned': np.column_stack([x1, x2, 2*x1 + x3/10])
    }

    for label, matrix in matrices.items():
        print('-' * 30)
        print(label, 'matrix')
        print('VIF:', variance_inflation_factor(matrix))
        print('condition number:', condition_number(matrix))
