from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np


class LogLogTransform:

    @staticmethod
    def transform(x):
        return np.log(-np.log(x))

    @staticmethod
    def inverse_transform(x):
        return np.exp(-np.exp(x))


class KaplanMeierEstimator:
    """Kaplan-Meier estimator for right-censored survival data"""

    def __init__(self):
        self.event_table = None

    def fit(self, durations, events):
        durations, events = self._validate_input(durations, events)
        self.event_table = self._get_event_table(durations, events)
        return self

    def predict(self, times):
        """
        Estimate S(t) for arbitrary time points.

        Kaplan-Meier is a right-continuous step function.
        """

        self._check_fitted()

        times = np.asarray(times, dtype=float)

        if np.any(~np.isfinite(times)):
            raise ValueError('times must not contain infinite values.')

        event_times = self.event_table['time'].values
        survival = self.event_table['survival'].values

        return np.where(times < event_times[0], 1.0, np.take(survival, np.searchsorted(event_times, times, side='right')-1))

    def plot(self, confidence_level=0.95, median=True, ax=None, **kwargs):
        """
        Plot Kaplan-Meier survival curve.

        set `confidence_level` to zero to ignore CI.
        set `median` to `False`to ignore median survival indicator.
        """

        self._check_fitted()

        if ax is None:
            _, ax = plt.subplots(**kwargs)

        times = self.event_table['time'].values
        survival = self.event_table['survival'].values

        ax.step(times, survival, where='post', label='Kaplan-Meier')

        if confidence_level > 0:
            lower, upper = self.confidence_interval(confidence_level)
            ax.fill_between(times, lower, upper, alpha=0.4)

        if median:
            ms = self.median_survival()
            if ms < float('inf'):
                ax.vlines(ms, times[0], times[-1], linestyles='dashed', label='median survival')

        ax.set_xlabel('Time')
        ax.set_ylabel('Survival Probability')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, None)
        ax.grid(alpha=0.25)

        if median:
            ax.legend()

        return ax

    def confidence_interval(self, level=0.95):
        '''
        Return Greenwood Confidence Intervals.

        Using log-log transformation we respect the 0 to 1 boundary.
        '''

        self._check_fitted()

        level = float(level)
        if not (0 < level < 1):
            raise ValueError('level must be within 0 and 1.')

        alpha = 1 - level
        q = 1 - alpha/2
        z = stats.norm.ppf(q)

        survival = self.event_table['survival'].values
        variance = self.event_table['variance'].values
        valid = (
            (survival > 0) &
            (survival < 1) &
            (variance > 0)
        )

        survival = survival[valid]
        variance = variance[valid]

        loglog = LogLogTransform.transform(survival)
        se = np.sqrt(variance) / (survival * np.abs(np.log(survival)))

        lower = np.full_like(valid, np.nan, dtype=float)
        lower[valid] = LogLogTransform.inverse_transform(loglog + z*se)

        upper = np.full_like(valid, np.nan, dtype=float)
        upper[valid] = LogLogTransform.inverse_transform(loglog - z*se)

        return lower, upper

    def median_survival(self):
        '''Return the first time at which S(t) <= 0.5'''
        return self.quantile_survival(0.5)

    def quantile_survival(self, q: float):
        '''Return the first time at which S(t) <= q'''

        self._check_fitted()

        q = float(q)
        if not (0 < q < 1):
            raise ValueError('quantile value must be within 0 and 1.')

        times = self.event_table['time'].values
        survival = self.event_table['survival'].values

        mask = survival <= q
        if not mask.any():
            return float('inf')

        return float(times[mask].min())

    @staticmethod
    def _get_event_table(durations: np.ndarray, events: np.ndarray) :
        records = [dict(time=0.0, at_risk=np.size(durations), events=0, survival=1.0, variance=0.0)]
        event_times = np.sort(np.unique(durations[events==1]))
        survival = records[-1]['survival']  # 1.0
        greenwood_sum = 0.0

        for t in event_times:
            t_mask = durations == t
            ni = int(np.sum(durations >= t))
            di = int(np.sum(t_mask & (events==1)))
            survival *= (1 - di/ni)
            if ni > di:
                greenwood_sum += di / (ni * (ni - di))

            variance = survival**2 * greenwood_sum
            records.append({
                'time': t,
                'at_risk': ni,
                'events': di,
                # 'censored': int(np.sum(t_mask)) - di,
                'survival': survival,
                'variance': variance
            })

        return pd.DataFrame(records)

    @staticmethod
    def _validate_input(durations, events):
        durations = np.asarray(durations, dtype=float)
        events = np.asarray(events, dtype=int)

        if durations.ndim != 1:
            raise ValueError("durations must be one-dimensional.")

        if events.ndim != 1:
            raise ValueError("events must be one-dimensional.")

        if durations.shape != events.shape:
            raise ValueError("durations and events must have the same length.")

        if durations.size < 1:
            raise ValueError("Input data cannot be empty.")

        if not np.all(np.isfinite(durations)):
            raise ValueError("durations must contain only finite values.")

        if np.any(durations < 0):
            raise ValueError("durations cannot be negative.")

        if not np.all(np.isin(events, [0, 1])):
            raise ValueError("events must contain only 0 (censored) and 1 (event).")

        return durations, events

    @property
    def is_fitted(self):
        return self.event_table is not None

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("KaplanMeierEstimator must be fitted for this operation.")

    def __repr__(self):
        props = dict(is_fitted=self.is_fitted)
        if self.is_fitted:
            props['n'] = self.event_table['at_risk'].iloc[0]
        text = ', '.join(map(lambda x: f"{x}={props[x]}", props.keys()))
        return "{}({})".format(type(self).__name__, text)


if __name__ == "__main__":
    n = 1000

    def survival(x):
        return np.exp(-x)

    T = -np.log(np.random.random(n))
    C = -np.log(np.random.random(n))
    events = (C < T).astype(int)
    durations = np.minimum(T, C)

    print(pd.Series(durations).groupby(events).describe())

    km = KaplanMeierEstimator().fit(durations, events)
    print(km)
    # print(km.event_table)

    a = km.event_table['time'].min()
    b = km.event_table['time'].max()
    t = np.linspace(a, b, 1000)
    p = survival(t)

    ax = km.plot()
    ax.plot(t, p, 'r-', label='true sf')
    ax.legend()

    plt.show()
