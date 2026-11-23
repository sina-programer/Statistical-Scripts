import numpy as np

def distance_matrix(X):
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)

def normalized_stress(D1, D2):
    assert np.shape(D1) == np.shape(D2)
    ui = np.triu_indices_from(D1, k=1)
    return np.sqrt(
        np.sum((D1[ui] - D2[ui]) ** 2)
        / np.sum(D1[ui] ** 2)
    )

def mds(D, k=2):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    top = np.argsort(eigvals)[::-1][:k]
    return eigvecs[:, top] * np.sqrt(np.clip(eigvals[top], 0, None))

def smacof(D, k=2, X0=None, n_iter=300, tol=1e-9):
    n = D.shape[0]
    X = mds(D, k) if X0 is None else X0.copy()
    s_prev = normalized_stress(D, distance_matrix(X))
    for _ in range(n_iter):
        Dx = distance_matrix(X)
        B = -np.divide(D, Dx, out=np.zeros_like(D), where=Dx > 0)
        np.fill_diagonal(B, -B.sum(axis=1))
        X = B @ X / n
        s = normalized_stress(D, distance_matrix(X))
        if s_prev - s < tol:
            break
        s_prev = s
    return X


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    n = 500
    rng = np.random.default_rng(0)
    X_raw = rng.normal(size=(n, 2), scale=0.5)

    norms = np.linalg.norm(X_raw, axis=1)
    colors = (
        np.arange(n) % 2,
        (X_raw[:, 0] < X_raw[:, 1]**2).astype(int),
        (norms >= np.mean(norms)).astype(int)
    )

    n_colors = len(colors)
    fig, axes = plt.subplots(ncols=3, nrows=n_colors, figsize=(10, n_colors*3))

    for i, color in enumerate(colors):
        X = np.column_stack([X_raw, color])
        D = distance_matrix(X)

        Y = mds(D, k=2)
        Z = smacof(D, 2, X0=Y)

        s_y = normalized_stress(D, distance_matrix(Y))
        s_z = normalized_stress(D, distance_matrix(Z))

        axes[i, 0].scatter(X[:, 0], X[:, 1], c=color)
        axes[i, 1].scatter(Y[:, 0], Y[:, 1], c=color)
        axes[i, 2].scatter(Z[:, 0], Z[:, 1], c=color)

        axes[i, 0].set_title("Original")
        axes[i, 1].set_title("Classical MDS \nnormalized stress: {:.3f}".format(s_y))
        axes[i, 2].set_title("SMACOF \nnormalized stress: {:.3f}".format(s_z))

    plt.tight_layout()
    plt.show()
