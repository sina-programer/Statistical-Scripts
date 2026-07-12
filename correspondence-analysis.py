from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np


class FitResult:
    def __init__(self, inertia: np.ndarray, n: int, i: int, j: int):
        self.inertia = np.array(inertia)
        self.total_inertia = self.inertia.sum()
        self.explained_var = self.inertia / self.total_inertia  # explained variance in each axis

        self.n = int(n)  # number of observations (samples)
        self.i = int(i)
        self.j = int(j)
        self.dof = (self.i - 1) * (self.j - 1)
        self.min_dim = min(self.i, self.j)

        self.chi2 = self.n * self.total_inertia
        self.pvalue = stats.chi2.sf(self.chi2, self.dof)  # null hypothesis states independence
        self.cramer_v = np.sqrt(self.chi2 / (self.n * (self.min_dim - 1)))

    def update(self, **kwargs):
        return FitResult(
            kwargs.get('inertia', self.inertia),
            kwargs.get('n', self.n),
            kwargs.get('i', self.i),
            kwargs.get('j', self.j)
        )

    def summary(self):
        print('-' * 20)
        print('chi-squared:', self.chi2)
        print('degrees of freedom:', self.dof)
        print('number of samples:', self.n)
        print('p-value:', self.pvalue)
        print("Cramer's V:", self.cramer_v)
        print('explained var:', self.explained_var.round(4)*100)
        print('inertia:', self.inertia)
        print('-' * 20)


class CA:
    def __init__(self):
        self.table: np.ndarray = None
        self.row_coords: np.ndarray = None
        self.col_coords: np.ndarray = None
        self.result: FitResult = None

    def fit(self, table: np.ndarray):
        if not isinstance(table, np.ndarray):
            raise TypeError('table must be a numpy array')

        self.table = table
        n = np.sum(table)
        P = table / n
        row_masses = P.sum(axis=1)
        col_masses = P.sum(axis=0)

        if np.isclose(row_masses, 0).any() or np.isclose(col_masses, 0).any():
            raise ValueError('some margins sum up to zero')

        E = np.outer(row_masses, col_masses)  # expected proportions
        S = (P - E) / np.sqrt(E)  # standardized residual
        U, s, Vt = np.linalg.svd(S, full_matrices=False)
        self.row_coords = (U * s) / np.sqrt(row_masses[:, None])
        self.col_coords = (Vt.T * s) / np.sqrt(col_masses[:, None])

        ## alternative matrix-based method (identical)
        # D_r_sqrt_inv = np.diag(1 / np.sqrt(row_masses))
        # D_c_sqrt_inv = np.diag(1 / np.sqrt(col_masses))
        # S = D_r_sqrt_inv @ (P - E) @ D_c_sqrt_inv
        # U, s, Vt = np.linalg.svd(S, full_matrices=False)
        # self.row_coords = D_r_sqrt_inv @ U @ np.diag(s)
        # self.col_coords = D_c_sqrt_inv @ Vt.T @ np.diag(s)

        self.result = FitResult(s**2, n, *table.shape)
        return self

    def biplot(self, row_labels=None, col_labels=None, **kwargs):
        x_row = self.row_coords[:, 0]
        y_row = self.row_coords[:, 1]
        x_col = self.col_coords[:, 0]
        y_col = self.col_coords[:, 1]

        fig, ax = self.get_subplots(**kwargs)

        ax.scatter(x_row, y_row, alpha=0.6, c='red', label='Rows')
        ax.scatter(x_col, y_col, alpha=0.6, c='blue', label='Columns')

        if row_labels is not None:
            for i, label in enumerate(row_labels):
                ax.text(x_row[i], y_row[i], str(label), color='red',
                        horizontalalignment='center', verticalalignment='bottom')

        if col_labels is not None:
            for i, label in enumerate(col_labels):
                ax.text(x_col[i], y_col[i], str(label), color='blue',
                        horizontalalignment='center', verticalalignment='top')

        ax.set_title('Correspondence Analysis')
        ax.legend(framealpha=0.8)
        fig.tight_layout()

        return fig, ax

    def summary(self):
        return self.result.summary()

    def plot(self, *args, **kwargs):
        return self.biplot(*args, **kwargs)

    @property
    def is_fitted(self): return self.result is not None

    @classmethod
    def get_subplots(cls, centered=True, grid=True, **kwargs):
        fig, ax = plt.subplots(**kwargs)

        if centered:
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            # ax.set_xlabel('Dimension 1', loc='left')
            # ax.set_ylabel('Dimension 2', loc='bottom')

        if grid:
            ax.grid(True, alpha=0.35)

        return fig, ax


class MCA(CA):
    def __init__(self):
        self.vars = None
        self.labels = None
        self.colors = None
        self.legends = None
        super().__init__()

    def fit(self, table: pd.DataFrame, burt=False, correction=True):
        columns = table.columns.tolist()
        table = table.values
        n, K = table.shape

        self.labels = []
        self.vars = []
        stack = []
        for i in range(K):
            dummies = pd.get_dummies(table[:, i])
            self.labels += dummies.columns.tolist()
            self.vars.extend([i]*dummies.shape[1])
            stack.append(dummies.values.astype(int))
        table = np.column_stack(stack)

        colors = plt.cm.Set1(np.arange(K) / K)
        self.colors = [colors[x] for x in self.vars]
        self.legends = [columns[x] for x in self.vars]

        if burt:
            table = table.T @ table

        super().fit(table)

        updates = {}
        if correction:
            # Benzecri Correction
            threshold = 1 / K
            inertia = self.result.inertia
            inertia_corrected = np.square(K/(K-1) * (inertia - threshold))
            inertia_adjusted = np.where(inertia > threshold, inertia_corrected, 0)
            updates['inertia'] = inertia_adjusted
        self.result = self.result.update(**updates)
        return self

    def biplot(self, **kwargs):
        fig, ax = self.get_subplots(**kwargs)

        X = self.col_coords[:, 0]
        Y = self.col_coords[:, 1]
        last_title = ''

        for x, y, title, label, color in zip(X, Y, self.legends, self.labels, self.colors):
            legend = {}
            if title != last_title:
                legend['label'] = title
                last_title = title

            ax.scatter(x, y, color=color, alpha=0.6, **legend)
            ax.text(x, y, str(label), color=color, horizontalalignment='center', verticalalignment='bottom')

        ax.set_title('Multiple Correspondence Analysis')
        ax.legend(framealpha=0.8)
        fig.tight_layout()

        return fig, ax


if __name__ == '__main__':
    df = pd.DataFrame({
        "Gender": ["M","M","F","F","M","F","F","M"],
        "Education": ["PhD","MSc","BSc","MSc","PhD","BSc","MSc","BSc"],
        "Job": ["Tech","Tech","HR","HR","Tech","HR","HR","Tech"],
        "City": ["Urban","Urban","Rural","Rural","Urban","Rural","Urban","Urban"]
    })
    
    c1 = 'Education'
    c2 = 'City'

    cross = pd.crosstab(df[c1], df[c2])
    ca = CA().fit(cross.values)
    ca.summary()
    ca.biplot(row_labels=cross.index.values, col_labels=cross.columns.values)

    mca = MCA().fit(df[[c1, c2]], burt=False, correction=False)
    mca.summary()
    mca.biplot()

    plt.show()
