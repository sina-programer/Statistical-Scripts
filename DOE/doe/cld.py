from collections import UserDict
import itertools as it
import string

import pandas as pd


class Graph(UserDict):
    '''
    Graph structure implemented as a dictionary.
    Keys are considered to be nodes and each value
    attached to a node shall be a set of nodes.
    Formal annotation of format: dict[Node, set[Node]]
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._is_qualified():
            raise ValueError('Graph structure is not valid.')

    @property
    def nodes(self):
        return list(self.keys())

    @property
    def degrees(self):
        return dict(zip(self.nodes, map(self.get_degree, self.nodes)))

    def get_degree(self, node):
        adjacents = self.get(node)
        if adjacents is not None:
            return len(adjacents)
        return -1

    def get_maximal_cliques(self):
        if not self:
            return []
        return list(self._drop_self_references()._bron_kerbosch(set(), set(self.keys()), set()))

    def _is_qualified(self):
        '''checks for qualification of graph format'''
        for key, value in self.items():
            if isinstance(value, set):
                continue
            elif hasattr(value, '__iter__'):
                self[key] = set(value)
            else:
                return False
        return True

    def _drop_self_references(self):
        return Graph({
            u: self[u] - {u}
            for u in self.keys()
        })

    def _bron_kerbosch(self, R, P, X):
        '''Tomita's Bron-Kerbosch algorithm with pivoting'''
        if P or X:
            u = max(P | X, key=lambda v: len(self[v] & P))
            for v in P - self[u]:
                V = {v}
                Nv = self[v]
                yield from self._bron_kerbosch(R|V, P&Nv, X&Nv)
                P.remove(v)
                X.add(v)
        else:
            yield R

    def to_edges(self):
        for u in self.keys():
            for v in self.get(u):
                yield (u, v)

    @classmethod
    def from_edges(cls, edges: list[tuple], directed=False):
        nodes = {node for edge in edges for node in edge}
        graph = {node: set() for node in nodes}
        for u, v in edges:
            graph[u].add(v)
            if not directed:
                graph[v].add(u)
        return cls(graph)


def compact_letter_display(
    pvalues: pd.DataFrame,
    means: pd.Series,
    alpha: float = 0.05,
    ascending: bool = False,
    abbreviated: bool = True
) -> dict:
    """
    Generate a CLD for groups based on pairwise p-values.

    Parameters
    ----------
    pvalues : pd.DataFrame
        Square symmetry matrix of pairwise comparison p-values.
        Its columns must be the group labels being compared.
    means : pd.Series
        EMM (or mean) for each group, indexed by group labels.
        Order of groups is determined by their means.
    alpha : float, default=0.05
        Significance level used to distinguish statistically significant pairs.
    ascending : bool, default=False
        If True, the lowest mean gets 'a';
        if False, the highest mean gets 'a'.
    abbreviated : bool, default=False
        If True, group labels belonging to more than two letters
        are abbreviated into the first and last letters.

    Returns
    -------
    dict
        Compact letter display by group labels. Groups sharing at least one letter
        are not significantly different at the specified significance level.
    """

    labels = means.index

    graph = Graph({
        label: set(labels[pvalues[label] >= alpha])
        for label in labels
    })

    cliques = list(map(tuple, graph.get_maximal_cliques()))
    clique_means = {
        clique: means[list(clique)].mean()
        for clique in cliques
    }

    cliques.sort(key=clique_means.get, reverse=not ascending)

    n_cliques = len(cliques)
    n_batches = (n_cliques - 1) // 26
    if n_batches > 0:
        letters = list(it.islice(_letter_sequence_numbered(), n_cliques))
    else:
        letters = list(string.ascii_lowercase[:n_cliques])

    result = {
        label: []
        for label in labels
    }

    for clique, letter in zip(cliques, letters):
        for label in clique:
            result[label].append(letter)

    if abbreviated:
        for label, symbols in tuple(result.items()):
            if len(symbols) > 2:
                result[label] = [symbols[0], '-', symbols[-1]]

    return {
        label: "".join(symbols)
        for label, symbols in result.items()
    }


def _letter_sequence_numbered():
    """Generate numbered labels: a1, b1, ..., z1, a2, b2, ..."""

    alphabet = string.ascii_lowercase
    batch_size = len(alphabet)
    n = 1

    while True:
        d, m = divmod(n - 1, batch_size)
        label = alphabet[m] + str(d+1)
        yield label
        n += 1


def _letter_sequence_repeated():
    """Generate Excel-style labels: a, b, ..., z, aa, ab, ..."""

    alphabet = string.ascii_lowercase
    batch_size = len(alphabet)
    n = 1

    while True:
        value = n
        label = ""

        while value > 0:
            value, remainder = divmod(value - 1, batch_size)
            label = alphabet[remainder] + label

        yield label
        n += 1
