from scipy import stats
import pandas as pd

def cronbach_alpha(X: pd.DataFrame, alpha=0.05):
    """
    Cronbach's alpha reliability measure.

    Internal consistency which is usually measured with Cronbach's alpha
    ranges between negative infinity and one.
    A negative value indicates that the within-subject variability
    is greater than between-subject variability.
    Confidence intervals are calculated using Feldt's method.
    """

    r, c = X.shape

    sum_var = X.var(axis=0).sum()
    var_sum = X.sum(axis=1).var()
    cronbach = c/(c-1) * (1 - sum_var/var_sum)

    df1 = r - 1
    df2 = df1 * (c - 1)
    lower = 1 - (1 - cronbach) * stats.f.isf(alpha/2, df1, df2)
    upper = 1 - (1 - cronbach) * stats.f.isf(1 - alpha/2, df1, df2)

    return cronbach, (lower, upper)
