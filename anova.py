from typing import TypeAlias, Sequence, Iterable, Any
from scipy import stats
import itertools as it
import pandas as pd
import numpy as np
import string

Label: TypeAlias = str | tuple
TableLike: TypeAlias = dict[Label, dict[str, int|float]] | pd.DataFrame
ListLike: TypeAlias = Any | Sequence[Any]

class DegenerateDataError(Exception):
    '''raised in ANOVA when the degree of freedom is zero (not enough data)'''

class ConstantDataError(Exception):
    '''raised when all values in groups are identical'''

class NoFactorError(Exception):
    '''raised when ANOVA is running without any factor'''

def as_list(obj, cls=str) -> list:
    if isinstance(obj, cls):
        return [obj]
    elif hasattr(obj, '__iter__'):
        if cls is not None:
            obj = map(cls, obj)
        return list(obj)
    return []

def within_errors(series):
    '''returns squared deviations from sample mean'''
    return np.square(
        np.subtract(
            series,
            np.nanmean(series)
        )
    )

def sum_of_within_errors(series):
    '''
    compute SS_within or SS_error also called 'corrected total sum of squares'
    which is the sum of squared deviations from sample mean.
    this function can be used as a callable for `pd.api.typing.DataFrameGroupBy.apply`.
    also direct call for a sequence of numbers is supported
    which its division by `n=len(series)` or `n-1` results in variance.
    '''
    return np.sum(within_errors(series))

def sum_of_between_errors(series, mu: float=0.0):
    '''Usage: `SS_model = pd.api.typing.DataFrameGroupBy[column].apply(sum_of_between_errors, mu=mu).sum()`'''
    # return series.count() * np.square(series.mean() - mu)
    return np.count_nonzero(~np.isnan(series)) * np.square(np.nanmean(series) - mu)

def anova(data: pd.DataFrame, target: str, factors: ListLike, blocks: ListLike | None = None, return_table: bool = True, replace_nan: bool = True, join_labels: bool = False, source: Iterable | None = None) -> TableLike:
    '''
    Analysis Of Variance
    Tests if the population means of `target` are same across `factors` (null hypothesis).
    The output is ANOVA table as a table-like object (dict[dict] or pd.DataFrame).
    Model & Error will be partitioned; respectively factors & blocks if provided.

    Output is consist of following attributes:
    - dof: degree of freedom
    - ss: sum of squares
    - ms: mean squares
    - eta2: variability explained
    - f: f statistic
    - p: p-value (significance level)

    If `return_table` is set to `True`, the ANOVA table
    is returned as `pd.DataFrame` otherwise a nested dictionary.
    If `replace_nan` is set to `True`, the meaningless values are replaced with nan.
    meaningless values are 'p' for total and error and 'f' for total.
    The parameter `source` determines labels standing for each factor in output.
    If `source` is `None`, enumerative values (from one) will represent factors.
    If `join_labels` is set to `True`, the combined labels will be a single string.

    If target values are all identical, ConstantDataError will be raised.
    The size of groups may differ, but if degree of freedom turn out
    to be zero, then a DegenerateDataError will be raised.

    ANOVA assumptions
    - The observations are independent.
    - Each group is drawn from a normally distributed population.
    - The population variance of groups are all equal. (homoscedasticity)
    Note: non-normality effect is negligible due to approximation of randomization test.

    For homoscedasticity testing, Levene test is widely used.
    If null hypothesis of Levene's test was rejected, rank-based nonparametric tests may be useful.
    '''

    assert isinstance(data, pd.DataFrame), 'input data must be an instance of `pandas.DataFrame`'
    n = data.shape[0]

    target = str(target)
    assert data[target].dtype in (int, float), 'target type must be numeric'
    mu = data[target].mean()

    factors = as_list(factors, cls=str)
    blocks = as_list(blocks, cls=str)
    if len(factors) == 0:
        raise NoFactorError('you must pass at least one factor to distinguish groups')

    grouped_data = data.groupby(factors)[target]
    k = grouped_data.ngroups

    total = {
        'dof': n - 1,
        'ss': float(sum_of_within_errors(data[target]))
    }
    error = {
        'dof': n - k,
        'ss': float(grouped_data.transform(within_errors).sum())
    }
    model = {
        'dof': k - 1,
        'ss': float(np.square(grouped_data.transform('mean') - mu).sum()),
        # 'ss': float(grouped_data.apply(sum_of_between_errors, mu=mu).sum())
    }
    output = {
        'total': total,
        'error': error
    }

    if error['ss'] == 0:
        raise ConstantDataError('all values are equal across factors')

    for idx, block in enumerate(blocks, start=1):
        grouped = data.groupby(block)[target]
        info = dict(
            dof=grouped.ngroups-1,
            ss=float(grouped.apply(sum_of_between_errors, mu=mu).sum())
        )
        output["b{}".format(idx)] = info
        error['dof'] -= info['dof']
        error['ss'] -= info['ss']

    if error['dof'] <= 0:
        raise DegenerateDataError('error degrees of freedom is invalid: {}'.format(error['dof']))

    output['model'] = model

    fs = len(factors)
    if not hasattr(source, '__getitem__'):
        source = range(1, fs+1)
    candidates = tuple(source)[:fs]  # standing for each factor
    pairs = dict(zip(candidates, factors))
    last = model.copy()  # highest order interaction

    for r in range(1, fs):
        for comb in it.combinations(candidates, r):
            subfactors = list(map(pairs.get, comb))
            grouped = data.groupby(subfactors)[target]
            dof = grouped.ngroups - 1
            ss_subtotal = float(np.square(grouped.transform('mean') - mu).sum())
            ss = ss_subtotal

            current_key = set(comb)
            for key, info in output.items():
                if current_key.issuperset(key):
                    dof -= info['dof']
                    ss -= info['ss']

            last['dof'] -= dof
            last['ss'] -= ss

            output[comb] = dict(
                dof=dof,
                ss=ss
            )

    if fs > 1:
        output[candidates] = last

    mse = error['ss'] / error['dof']
    for key, info in tuple(output.items()):
        if info['dof'] <= 0:
            raise DegenerateDataError('degree of freedom of <{}> is not valid: {}'.format(key, info['dof']))

        info['ms'] = info['ss'] / info['dof']
        info['eta2'] = info['ss'] / total['ss']
        info['f'] = info['ms'] / mse
        info['p'] = float(stats.f.sf(info['f'], info['dof'], error['dof']))

        if join_labels and not isinstance(key, str):
            output[''.join(map(str, key))] = info
            output.pop(key)

    if replace_nan:
        total['f'] = np.nan
        total['p'] = np.nan
        error['p'] = np.nan

    if return_table:
        return pd.DataFrame(output).T.astype({'dof': int})
    return output


def levene_test(df: pd.DataFrame, target: str, factors: ListLike, agg='median'):
    factors = as_list(factors, cls=str)
    grouped = df.groupby(factors)[target]
    column = 'deviations'
    while column in factors:
        column = '_' + column
    new_df = df[factors].copy()
    new_df[column] = np.abs(df[target] - grouped.transform(agg))
    return anova(new_df, column, factors, return_table=False)['model']['p']


def get_sample_data(factors=2, replicates=3, error_scale=0.5, random_state=None):
    rng = np.random.default_rng(random_state)
    factor_names = list(string.ascii_uppercase[:factors])
    factor_indices = list(it.product([0, 1], repeat=factors)) * replicates
    data = pd.DataFrame(factor_indices, columns=factor_names).sort_values(factor_names).reset_index(drop=True)
    data['target'] = rng.normal(0, error_scale, data.shape[0])
    for i in range(factors):
        data['target'] += data.iloc[:, i] * (i+1)
    return data


if __name__ == "__main__":
    data = get_sample_data()
    factors = data.columns.difference(['target']).to_list()
    anv = anova(data, 'target', factors)
    print(anv)
    print("Levene Test of Homoscedasticity p-value:", levene_test(data, 'target', factors))
