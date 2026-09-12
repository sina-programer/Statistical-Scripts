from matplotlib.ticker import PercentFormatter
from matplotlib import pyplot as plt
import numpy as np

_percent_formatter = PercentFormatter(1.0)

def lorenz_curve(x):
    x = np.sort(np.asarray(x))
    cum = np.cumsum(x)
    cum = np.insert(cum, 0, 0) / cum[-1]
    pop = np.linspace(0, 1, len(cum))
    return pop, cum

def gini_coefficient(x):
    pop, cum = lorenz_curve(x)
    return 1 - 2 * np.trapezoid(x=pop, y=cum)

def plot_lorenz(x, ax=None, label=None, fill=True, **ax_kws):
    ax = ax or plt.gca()
    pop, cum = lorenz_curve(x)
    gini = gini_coefficient(x)

    ax.plot(pop, cum, marker='.', lw=2, label=label or 'Lorenz curve')
    ax.plot([0, 1], [0, 1], linestyle='--', color='red', lw=1, label='Equality')
    if fill:
        ax.fill_between(pop, cum, pop, alpha=0.15)

    ax.spines[['top', 'left']].set_visible(False)
    ax.legend(loc='upper left', frameon=False)

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')

    ax.xaxis.set_major_formatter(_percent_formatter)
    ax.yaxis.set_major_formatter(_percent_formatter)

    ax_kws.setdefault('xlim', (0, 1))
    ax_kws.setdefault('ylim', (0, 1))
    ax_kws.setdefault('xlabel', 'Cumulative Population Share')
    ax_kws.setdefault('ylabel', 'Cumulative Distribution Share')
    ax_kws.setdefault('title', 'Lorenz Curve (gini = {:.3f})'.format(gini))
    ax.set(**ax_kws)

    return ax


if __name__ == "__main__":
    n = 100
    sample = np.random.normal(size=n) ** 2

    plot_lorenz(sample)
    plt.tight_layout()
    plt.show()
