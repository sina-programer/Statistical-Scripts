import itertools as it

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from doe import ModelSpec, fit, emmeans, labeled_barplot


def get_demo_data(replicates=5) -> pd.DataFrame:
    factors = {
        'treatment': ['A', 'B', 'C'],
        'block': ['b1', 'b2'],
        'sliced': [0, 1]
    }

    grid = list(it.product(*factors.values()))
    data = pd.DataFrame(
        grid * replicates,
        columns=factors.keys()
    )

    data['response'] = np.random.randn(data.shape[0])
    data.loc[data['treatment'].eq('A'), 'response'] += 1.5
    data.loc[data['treatment'].eq('B') & data['block'].eq('b2'), 'response'] -= 1
    data['response'] += data['sliced']

    return data


def main():
    data = get_demo_data()
    spec = ModelSpec("response ~ treatment + block")

    fitted = fit(spec, data, slice_by=("sliced",))
    anv = fitted.anova()
    print(anv)

    emm = emmeans(fitted, "~ treatment").emm()
    print(emm)

    labeled_barplot(emm, x='treatment', y='estimate', label='cld', col='sliced')
    plt.show()


if __name__ == "__main__":
    main()
