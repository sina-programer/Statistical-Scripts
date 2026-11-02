# the plot has been derived from scikit-learn

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np


def plot_silhouette(scores, cluster_ids, gap=10):
    scores = np.asarray(scores)
    cluster_ids = np.asarray(cluster_ids)
    clusters = np.unique(cluster_ids)
    n_clusters = len(clusters)
    y_lower = gap

    fig, ax = plt.subplots(figsize=(9, 7))
    for rank, cluster in enumerate(clusters):
        values = np.sort(scores[cluster_ids == cluster])
        n_values = len(values)
        y_upper = y_lower + n_values
        color = cm.nipy_spectral(rank / n_clusters)

        ax.fill_betweenx(
            np.arange(y_lower, y_upper), 0, values,
            facecolor=color, edgecolor=color, alpha=0.7,
        )
        ax.text(scores.min() - 0.05, y_lower + 0.5 * n_values, str(cluster))
        y_lower = y_upper + gap

    ax.axvline(scores.mean(), color="red", linestyle="--")
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax.set(
        xlim=(min(scores.min() - 0.1, -0.1), 1),
        ylim=(0, y_lower),
        yticks=[],
        xlabel="Silhouette Score",
        ylabel="Cluster"
    )

    return fig, ax


if __name__ == "__main__":
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_samples

    X, _ = make_blobs(n_samples=500, centers=4, random_state=1)
    labels = KMeans(n_clusters=4, random_state=10).fit_predict(X)
    plot_silhouette(silhouette_samples(X, labels), labels)
    plt.show()
