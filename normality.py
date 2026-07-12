from scipy import stats
import numpy as np

alpha = 0.05
array = np.random.random(250)*2 - 1

print('Data has a normal distribution?')

shapiro = stats.shapiro(array)
print('Shapiro:', 'YES' if shapiro.pvalue > alpha else 'NO')

ntest = stats.normaltest(array)
print('Normal Test:', 'PROBABELY' if ntest.pvalue > alpha else 'NO')

ks = stats.kstest(array, stats.norm(array.mean(), array.std()).cdf)
print('Kolmogorov–Smirnov:', 'YES' if ks.pvalue > alpha else 'NO')

anderson = stats.anderson(array, 'norm')
print('Anderson (alpha=0.05):', 'YES' if anderson.statistic <= anderson.critical_values[2] else 'NO')

def normality_statistic(sample):
    return stats.shapiro(sample).statistic

monte_carlo = stats.monte_carlo_test(array, stats.norm.rvs, normality_statistic)
print('Monte-Carlo (shapiro based):', 'YES' if monte_carlo.pvalue > alpha else 'NO')
