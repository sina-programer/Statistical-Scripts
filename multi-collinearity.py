import numpy as np

def variance_inflation_factor(X: np.ndarray, add_intercept=True):
    '''
    VIF for each column in matrix X is computed.
    The intercept shall be ommited from input matrix but if it has,
    set `add_intercept=False` to avoid singular matrix error.
    VIF_j = 1 / (1 - R^2_j)
    in which VIF_j is the value of VIF for j_th column in X
    and R^2_j is the R-squared of regressing j_th column on other columns.
    usually a VIF value greater than 10 concerns us about collinearity.
    '''

    n, p = X.shape
    J = np.ones(n)
    VIF = np.zeros(p)
    for i in range(p):
        y = X[:, i]
        x = np.delete(X, i, axis=1)  # X[:, list(set(range(p))-{i})]
        if add_intercept:
            x = np.column_stack([J, x])
        beta = np.linalg.inv(x.T @ x) @ x.T @ y
        y_pred = x @ beta
        ssr = np.sum(np.square(y - y_pred))
        sst = np.sum(np.square(y - np.mean(y)))
        r2 = 1 - ssr/sst
        VIF[i] = 1 / (1-r2)
    return VIF

def condition_number(X: np.ndarray):
    '''a condition number greater than 30 is problematic'''
    R = np.corrcoef(X.T)
    eigenvalues = np.abs(np.linalg.eigvalsh(R))
    kappa = np.max(eigenvalues) / np.min(eigenvalues)
    return kappa
