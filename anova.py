from functools import partial
from scipy import stats
import itertools as it
import pandas as pd
import numpy as np


def sum_of_between_errors(series, delta=0):
    # return series.count() * np.square(series.mean() - delta)
    return np.count_nonzero(~np.isnan(series)) * np.square(np.nanmean(series) - delta)


def anova(df: pd.DataFrame, target, *factors):
    factors = list(factors)
    gdf = df.groupby(factors)[target]
    gdfs = {f: df.groupby(f)[target] for f in factors}

    mu = df[target].mean()
    mdfs = {f: g.mean() for f, g in gdfs.items()}
    se = partial(sum_of_between_errors, delta=mu)

    n = len(df)
    k = len(gdf)
    ks = {f: len(g) for f, g in gdfs.items()}
    output = {}

    total = {}
    total['df'] = n - 1
    total['ss'] = float(np.square(df[target] - mu).sum())
    total['ms'] = total['ss'] / total['df']
    total['f'] = np.nan
    total['p'] = np.nan
    output['total'] = total

    error = {}
    error['df'] = n - k
    error['ss'] = float(gdf.transform(lambda x: np.square(x - x.mean())).sum())
    error['ms'] = error['ss'] / error['df']
    error['f'] = np.nan
    error['p'] = np.nan
    output['error'] = error

    model = {}
    model['df'] = k - 1
    model['ss'] = float(gdf.apply(se).sum())
    model['ms'] = model['ss'] / model['df']
    model['f'] = model['ms'] / error['ms']
    model['p'] = float(stats.f.sf(model['f'], model['df'], error['df']))
    output['model'] = model

    for i, factor in enumerate(factors):
        info = {}
        info['df'] = ks[factor] - 1
        info['ss'] = float(gdfs[factor].apply(se).sum())
        info['ms'] = info['ss'] / info['df']
        info['f'] = info['ms'] / error['ms']
        info['p'] = float(stats.f.sf(info['f'], info['df'], error['df']))
        output[(i+1,)] = info

    rng = list(range(1, len(factors)+1))
    for r in rng[1:]:
        for comb in it.combinations(rng, r):
            fs = np.take(factors, np.subtract(comb, 1)).tolist()
            gf = df.groupby(fs)[target]
            mf = gf.mean()

            def interaction(sdf):
                return sdf.count() * np.square(
                    mf[sdf.name]
                    - sum([mdfs[fs[i]][sdf.name[i]] for i in range(len(fs))])
                    + mu * (len(fs)-1)
                ).sum()

            info = {}
            info['df'] = int(np.prod([output[(i,)]['df'] for i in comb]))
            info['ss'] = float(gf.apply(interaction).sum())
            info['ms'] = info['ss'] / info['df']
            info['f'] = info['ms'] / error['ms']
            info['p'] = float(stats.f.sf(info['f'], info['df'], error['df']))
            output[comb] = info

    return output
