from contextlib import contextmanager
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np

Q = np.arange(1, 100) / 100

def fit_linear(X, Y):
    '''equals to `np.polyfit(X, Y, 1)`'''
    C = np.cov(X, Y)
    b = C[0, 1] / C[0, 0]
    a = np.mean(Y) - np.mean(X) * b
    return a, b

def fit_normal(array):
    mu = np.nanmean(array)
    sigma = np.nanstd(array)
    return stats.norm(mu, sigma)

def get_trend_coeffs(X, Y, n=1):
    '''returns coefficients of trend in increasing power order'''
    if n == 1:
        return fit_linear(X, Y)
    return np.polyfit(X, Y, n)

def coeffs_as_formula(coeffs, var='t', floating=3):
    '''convert coefficients in increasing power order into human-readable formula'''
    output = ''
    for i, c in enumerate(coeffs):
        symbol = ''
        if i > 0:
            symbol += var
        if i > 1:
            symbol += '^{}'.format(i)
        output += "{:+.{}f} {}".format(c, floating, symbol)
    return output.strip()

def seasonal_diff(series, lag=1):
    '''returns `Y[t] - Y[t-lag]` with nan values ignored'''
    assert isinstance(lag, int) and 0 <= lag < len(series)
    if lag == 0: return np.array(series)
    return np.subtract(series[lag:], series[:-lag])

def seasonal_goodness(array, max_lag=None):
    '''returns a dictionary from lag-rmse pairs'''
    if max_lag is None:
        max_lag = len(array) // 2

    return {
        lag: np.sqrt(
            np.nanmean(
                np.square(
                    seasonal_diff(
                        array,
                        lag
                    )
                )
            )
        )

        for lag in range(1, max_lag+1)
    }

def seasonal_period(array, k=3):
    '''returns the lag in which the least rmse is observed; lag=1 means no additive seasonal pattern'''
    statistics = seasonal_goodness(array)
    periods = sorted(statistics, key=statistics.get)
    period = periods[0]  # least observed rmse; can be returned as 
    if k > 1:
        period = min(p for p in periods[:k] if period%p == 0)
    return period

def deep_seasonal_periods(array, **kwargs):
    '''return a list of sequentially detected periods'''
    periods = []
    while (lag := seasonal_period(array, **kwargs)) > 1:
        periods.append(lag)
        array = seasonal_diff(array, lag)
    return periods

def percentiles(array):
    '''return all percentiles of `array` between 1 and 99'''
    return np.quantile(array, Q)

def percentiles_normal(array):
    '''return the percentiles of `array` assuming it was drawn from a normally distributed population'''
    return fit_normal(array).ppf(Q)

def qq_plot(x, y, line=True):
    '''plot percentiles of `x` against `y` as scatter'''
    qx = percentiles(x)
    qy = percentiles(y)
    plt.scatter(qx, qy)
    if line:
        plt.plot([np.min(qx), np.max(qx)], [np.min(qy), np.max(qy)])

def qq_plot_normal(array):
    '''plot percentiles of `array` against those if it were normally distributed'''
    qq_plot(array, percentiles_normal(array))

@contextmanager
def subplots(**kwargs):
    try:
        figure, axes = plt.subplots(squeeze=False, **kwargs)
        yield figure, axes

    finally:
        for sequence in axes:
            for ax in sequence:
                ax.grid()

                for line in ax.lines:
                    if not line._label.startswith('_'):
                        ax.legend()
                        break

        figure.tight_layout()

def decomposition(t, y, order=1, graph=True):
    '''
    Y = Trend + Seasonal + Residual

    only additive model is supported, since a multiplicative model
    can simply be converted to additive by log transform.

    output is consist of each component plus `lag` and `coefficients`.
    `coefficients` is the trend coefficients in increasing power order.

    if `graph` is associated with a truthy value, `plt.show()` must be called later.
    '''

    t, y = map(np.array, (t, y))
    obs = len(y)

    trend_coeffs = get_trend_coeffs(t, y, n=order)
    trend_func = np.poly1d(trend_coeffs[::-1])
    trend = trend_func(t)
    detrended = y - trend

    lag = seasonal_period(detrended)
    seasonal = np.array([np.mean(detrended[i::lag]) for i in range(lag)] * (obs//lag + 1))[:obs%lag-lag]
    seasonal_adj = y - seasonal

    estimate = trend + seasonal
    residual = y - estimate
    rmse = np.sqrt(np.mean(np.square(residual)))

    if graph:
        with subplots(nrows=2, ncols=2, figsize=(16, 9)) as (fig, ((ax11, ax12), (ax21, ax22))):
            ax11.plot(t, y, marker='.', color='blue', label='observed')
            ax11.plot(t, estimate, color='red', label='estimate', linestyle='--')
            ax11.set_title('Original Series')

            ax12.plot(t, detrended, marker='.', color='blue', label='detrended')
            ax12.plot(t, seasonal, color='red', label='seasonal', linestyle='--')
            ax12.set_title('Detrended ({})'.format(coeffs_as_formula(trend_coeffs)))

            ax21.plot(t, seasonal_adj, marker='.', color='blue', label='adjusted')
            ax21.plot(t, trend, color='red', label='trend', linestyle='--')
            ax21.set_title(f'Seasonal Adjusted (lag={lag})')

            ax22.plot(t, residual, marker='.', color='blue', label='residual')
            ax22.axhline(np.mean(residual), np.min(t), np.max(t), linestyle='--', color='red', label='mean')
            ax22.set_title('Residual (RMSE={:.4f})'.format(rmse))

        with subplots(nrows=2, ncols=2, figsize=(16, 9)) as (fig, ((ax11, ax12), (ax21, ax22))):
            qr = percentiles(residual)
            qz = percentiles_normal(residual)
            ax11.scatter(qr, qz, c='blue')
            ax11.plot([np.min(qr), np.max(qr)], [np.min(qz), np.max(qz)], 'r--')
            ax11.set_title('QQ plot for normality of residuals')
            ax11.set_xlabel('percentiles of residuals')
            ax11.set_ylabel('normal percentiles')

            intercept, slope = fit_linear(estimate, residual)
            span = np.linspace(estimate.min(), estimate.max(), 101)
            ax12.scatter(estimate, residual, c='blue')
            ax12.plot(span, span*slope+intercept, 'r--', label='tendency')
            ax12.set_title('Fitted vs Residual for relative error')
            ax12.set_xlabel('fitted (estimate)')
            ax12.set_ylabel('residual (error)')

            residual_norm = fit_normal(residual)
            span = np.linspace(residual_norm.ppf(0.001), residual_norm.ppf(0.999), 101)
            ax21.hist(residual, bins=min(50, obs), color='blue', density=True)
            ax21.plot(span, residual_norm.pdf(span), 'r--')
            ax21.set_title('Residual Histogram')

            series = pd.Series(seasonal_goodness(detrended))
            series.plot(color='blue', ax=ax22)
            minimals = series.sort_values()[:3]
            plt.hlines(minimals, 0, minimals.index, colors='red', linestyles='--')
            ax22.set_title('RMSE for each seasonal lag')
            ax22.set_xlabel('lag')
            ax22.set_ylabel('RMSE')

    return {
        'coefficients': trend_coeffs,
        'lag': lag,
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual,
    }


if __name__ == '__main__':
    n = 50
    t = np.arange(n)
    e = np.random.randn(n) * 2
    X = t/2 + 1 + e
    X[::3] -= np.mean(X)/2

    result = decomposition(t, X, graph=True)
    plt.show()
