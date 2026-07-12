from dataclasses import dataclass
from scipy import stats
import numpy as np

@dataclass
class Chi2Test:
    statistic: float
    dof: int

    def __post_init__(self):
        self.pvalue = stats.chi2(self.dof).sf(self.statistic)

    def __str__(self):
        return "\n".join([
            'Chi-square test',
            'degrees of freedom: {}'.format(self.dof),
            'statistic: {:.4f}'.format(self.statistic),
            'p-value: {:.5f}'.format(self.pvalue)
        ])

@dataclass
class MutualInformation:
    mutual_information: float
    entropy_joint: float
    entropy_x: float
    entropy_y: float
    levels_x: int
    levels_y: int
    observations: int

    @classmethod
    def fit(cls, X, Y):
        h_x, nx = entropy(X, return_nuniques=True)
        h_y, ny = entropy(Y, return_nuniques=True)
        h_xy = joint_entropy(X, Y)
        mi = h_x + h_y - h_xy
        # mi = h_y - joint_entropy(X, Y, conditional=True)
        return cls(mi, h_xy, h_x, h_y, nx, ny, len(X))

    def normalized_mi(self, method='sqrt'):
        if method == 'sqrt':
            denom = np.sqrt(self.entropy_x * self.entropy_y)
        elif method == 'sum':
            denom = self.entropy_x + self.entropy_y
        elif method == 'max':
            denom = max(self.entropy_x, self.entropy_y)
        elif method == 'min':
            denom = min(self.entropy_x, self.entropy_y)
        else:
            denom = 1
        return self.mutual_information / denom

    def adjusted_mi(self):  # adjusted for chance
        expected = (self.levels_x-1) * (self.levels_y-1) / (2*self.observations*np.log(2))  # dof / g2 * MI
        maximum = min(np.log2(self.levels_x), np.log2(self.levels_y))
        ami = (self.mutual_information - expected) / (maximum - expected)
        return max(0, min(1, ami))

    def statistical_test(self):
        g2 = 2 * self.observations * np.log(2) * self.mutual_information
        dof = (self.levels_x-1) * (self.levels_y-1)
        return Chi2Test(g2, dof)

    def to_dict(self):
        return {
            'Adjusted MI': self.adjusted_mi(),
            'Normalized MI': self.normalized_mi(),
            **{key.replace('_', ' ').title(): getattr(self, key) for key in self.__annotations__.keys()},
            'Statistical Test': self.statistical_test()
        }

    def summary(self):
        return "\n".join([
            'Mutual Information Decomposition',
            *['{}: {}'.format(key, value) for key, value in self.to_dict().items()],
        ])


def logarithm(x, base=10):
    return np.log10(x) / np.log10(base)

def contingency_table(X, Y):
    U1, E1 = np.unique(X, return_inverse=True)
    U2, E2 = np.unique(Y, return_inverse=True)
    return np.histogram2d(E1, E2, bins=[len(U1), len(U2)])[0]

def entropy(X, base=2, return_nuniques=True):
    U, C = np.unique(X, return_counts=True)
    P = C / np.sum(C)
    h = -np.sum(P * logarithm(P, base))
    if return_nuniques:
        return h, len(U)
    return h

def joint_entropy(X, Y, base=2, conditional=False):
    C = contingency_table(X, Y)
    P = C / np.sum(C)
    if conditional:
        Q = P / P.sum(axis=1).reshape(-1, 1)  # P(Y|X)
    else:
        Q = P  # P(X, Y)
    return -np.sum(P * logarithm(Q, base))


if __name__ == '__main__':
    A = np.array(['X', 'X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X', 'Y'])
    B = np.array(['A', 'B', 'A', 'C', 'B', 'A', 'C', 'C', 'B', 'A'])
    mi = MutualInformation.fit(A, B)
    print(mi.summary())
