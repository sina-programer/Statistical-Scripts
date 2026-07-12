from copy import deepcopy
from scipy import stats
import itertools as it
import operator as op
import numpy as np
import string
import math

class DivergentError(Exception):
    '''raised when the expected frequencies can not be estimated'''


class EffectTerms:
    copy = deepcopy

    def __init__(self, *terms, hierarchal=True):
        if hierarchal:
            terms = set(terms)
            for term in terms.copy():
                for k in range(len(term)):
                    terms.update(it.combinations(term, k))

        self.terms = sorted(terms, key=len)

    def get_maximal_terms(self):
        terms = list(map(set, self.terms))
        for i, term in enumerate(terms):
            for j, other in enumerate(terms):
                if i!=j and term.issubset(other):
                    break
            else:
                yield tuple(term)

    def drop_term(self, term):
        model = self.copy()
        term = set(term)
        for t in self.terms:
            if term.issubset(t):
                model.terms.remove(t)
        return model

    def get_notation(self, source=string.ascii_uppercase):
        return tuple(''.join(map(source.__getitem__, term)) for term in self.get_maximal_terms())

    @classmethod
    def from_notation(cls, *notation: str):
        '''
        Create instances by common notation in mathematics.
        (AB, CB) -> EffectTerms.from_notation('AB', 'CB')
        '''
        uniques = set(''.join(notation))
        codes = {x: i for i, x in enumerate(sorted(uniques))}
        terms = [tuple(map(codes.get, term)) for term in notation]
        return cls(*terms, hierarchal=True)

    def __iter__(self):
        return iter(self.terms)

    def __len__(self):
        return len(self.terms)

    def __str__(self):
        return "({})".format(", ".join('x'.join(map(str, term)) if term else 'grand' for term in self.get_maximal_terms()))


class CrossTabulation:
    copy = deepcopy

    def __init__(self, observed: np.ndarray):
        observed = np.array(observed, dtype=int)
        assert np.all(observed >= 0)

        self.observed = observed
        self.nobs = observed.sum()
        self.cells = np.prod(observed.shape)
        self._marginal_cache = {}

    def get_marginal(self, axes: tuple, agg=np.sum, from_cache=True):
        if isinstance(axes, int):
            axes = (axes,)
        axes = tuple(axes)

        key = (agg, axes)
        if from_cache and key in self._marginal_cache.keys():
            return self._marginal_cache[key]

        other_axes = self.inverse_axes(axes)
        marginal = agg(self.observed, axis=other_axes)
        marginal = self.reshape_to_broadcast(marginal, axes)
        self._marginal_cache[key] = marginal
        return marginal

    def reshape_to_broadcast(self, array, axes):
        shape = np.ones(self.ndim, dtype=int)
        for i, ax in enumerate(axes):
            shape[ax] = array.shape[i]
        return array.reshape(shape)

    def inverse_axes(self, axes):
        return tuple(set(range(self.ndim)) - set(axes))

    @property
    def shape(self):
        return self.observed.shape

    @property
    def ndim(self):
        return self.observed.ndim

    @classmethod
    def from_raw_data(cls, data: np.ndarray):
        assert isinstance(data, np.ndarray)
        r, c = data.shape

        if c == 0:
            raise ValueError('input array must contain at least one column')

        elif c == 1:
            counts = np.unique(return_counts=True)[-1]

        else:
            items = [np.unique(data[:, i], return_inverse=True) for i in range(c)]
            uniques = list(map(op.itemgetter(0), items))
            indexes = tuple(map(op.itemgetter(1), items))
            counts = np.zeros(tuple(map(len, uniques)), dtype=int)
            np.add.at(counts, indexes, 1)

        return cls(counts)

    def __str__(self):
        return f"{type(self).__name__}({self.nobs}: {self.shape})"


class LogLinearModel:
    __counter__ = 0
    copy = deepcopy

    def __init__(self, table: CrossTabulation, terms: EffectTerms, name=None):
        assert isinstance(table, CrossTabulation)
        assert isinstance(terms, EffectTerms)
        LogLinearModel.__counter__ += 1

        self.table = table
        self.terms = terms
        self.name = name or "model-{}".format(self.__counter__)

        self.expected = self.iterative_proportional_fitting()
        self.residuals = self.table.observed - self.expected
        self.standardized_residuals = self.residuals / np.sqrt(self.expected)
        self.parameters = self.estimate_parameters()

        mask = table.observed > 0
        self.deviance = 2 * np.sum(table.observed[mask] * np.log(table.observed[mask] / self.expected[mask]))
        self.chi2 = np.sum(np.square(table.observed - self.expected) / self.expected)
        self.dof = table.cells - sum(math.prod(table.shape[t]-1 for t in term) for term in terms)
        self.significance = stats.chi2(self.dof).sf(self.deviance)  # goodness-of-fit test significance
        self.aic = self.deviance - 2 * self.dof
        self.bic = self.deviance - np.log(self.table.nobs) * self.dof

    def iterative_proportional_fitting(self, max_iter=100, tolerance=1e-8):
        maximal_terms = list(self.terms.get_maximal_terms())
        expected_table = CrossTabulation(np.ones(self.table.shape) * self.table.nobs/self.table.cells)

        for iteration in range(max_iter):
            max_change = 0.0

            for axes in maximal_terms:
                target_margin = self.table.get_marginal(axes)
                current_margin = expected_table.get_marginal(axes, from_cache=False)
                scale = np.where(current_margin>0, target_margin / current_margin, 1)
                new_expected = expected_table.observed * scale
                change = np.max(np.abs(new_expected - expected_table.observed))
                if change > max_change:
                    max_change = change
                expected_table.observed = new_expected

            if max_change < tolerance:
                return expected_table.observed

        raise DivergentError

    def estimate_parameters(self):
        log_table = CrossTabulation(np.log(self.expected))
        parameters = {}
        for term in self.terms:
            parameters[term] = log_table.get_marginal(term, agg=np.mean)
            for k in range(len(term)):
                for subterm in it.combinations(term, k):
                    parameters[term] -= parameters[subterm]
        return parameters

    def evaluate(self, other=None, method='mle'):
        method_options = ('mle', 'pearson')
        if method not in method_options:
            raise ValueError(f"method must be one of: {method_options}")

        if other is None:
            other = SaturatedModel(self.table)
        assert isinstance(other, LogLinearModel)

        if other.dof > self.dof:
            self, other = other, self

        delta_dof = self.dof - other.dof
        dist = stats.chi2(delta_dof)
        if method == 'mle':
            pvalue = dist.sf(self.deviance - other.deviance)
        elif method == 'pearson':
            pvalue = dist.sf(self.chi2 - other.chi2)

        return pvalue

    def shrink(self, criterion='bic', level=0.1):
        assert isinstance(level, float) and 0<level<1
        criterion_options = ('aic', 'bic', 'significance')
        if criterion not in criterion_options:
            raise ValueError('criterion must be in {}'.format(criterion_options))

        baseline = SaturatedModel(self.table)
        model = self.copy()
        if model.evaluate(baseline) <= level:
            return model

        while True:
            counter = 0
            new_models = []
            for term in model.terms.terms[1:]:
                new_model = LogLinearModel(self.table, model.terms.drop_term(term))
                if new_model.significance > level:
                    new_models.append(new_model)
                    counter += 1

            if counter == 0:
                break

            indicators = list(map(op.attrgetter(criterion), new_models))
            model = new_models[np.argmin(indicators)]

        model.name = '{} - shrank'.format(self.name)
        return model

    def extend(self, criterion='bic', level=0.1):
        assert isinstance(level, float) and 0<level<1
        criterion_options = ('aic', 'bic', 'significance')
        if criterion not in criterion_options:
            raise ValueError('criterion must be in {}'.format(criterion_options))

        baseline = SaturatedModel(self.table)
        model = self.copy()

        while model.evaluate(baseline) <= level:
            new_models = [
                LogLinearModel(self.table, EffectTerms(*(model.terms.terms+[term])))
                for term in set(baseline.terms) - set(model.terms)
            ]
            indicators = list(map(op.attrgetter(criterion), new_models))
            model = new_models[np.argmin(indicators)]

        model.name = "{} - extended".format(self.name, )
        return model

    def summary(self, points=5):
        print('-' * 30)
        print('Model:', self.name)
        print('Terms:', self.terms.get_notation())
        print('Number of Cells:', self.table.cells)
        print('Degrees of Freedom:', self.dof)
        print('Deviance (G2): {:.{}f}'.format(self.deviance, points))
        print('Chi-square: {:.{}f}'.format(self.chi2, points))
        print('GOF significance:', self.significance)
        print('AIC:', self.aic)
        print('BIC:', self.bic)
        print('MS standardized residuals:', np.sum(np.square(self.standardized_residuals))/self.dof)
        print('=' * 30)


class SaturatedModel(LogLinearModel):
    def __init__(self, table: CrossTabulation, **kwargs):
        # effects = [dim for k in range(table.ndim+1) for dim in it.combinations(range(table.ndim), k)]
        super().__init__(table, EffectTerms(tuple(range(table.ndim)), hierarchal=True), **kwargs)

class IndependenceModel(LogLinearModel):
    def __init__(self, table: CrossTabulation, **kwargs):
        effects = [()] + [(dim,) for dim in range(table.ndim)]
        super().__init__(table, EffectTerms(*effects), **kwargs)

class NullModel(LogLinearModel):
    def __init__(self, table: CrossTabulation, **kwargs):
        super().__init__(table, EffectTerms(()), **kwargs)


if __name__ == "__main__":
    np.random.seed(42)

    n = 200
    shape = (2, 2, 2, 3)

    p = np.ones(shape, dtype=float)
    for a in range(shape[0]):
        for b in range(shape[1]):
            for c in range(shape[2]):
                for d in range(shape[3]):
                    if a == b:
                        p[a,b,c,d] *= 2.0
                    if c == d:
                        p[a,b,c,d] *= 1.5

    p /= p.sum()
    observed = np.random.multinomial(n, p.flatten()).reshape(shape)
    cross = CrossTabulation(observed)

    terms = EffectTerms((0,), (1,), (2,), (3,), (0,1))
    model = LogLinearModel(cross, terms)
    model.summary()

    saturated = SaturatedModel(cross)
    saturated.shrink().summary()

    null = NullModel(cross)
    null.extend().summary()
