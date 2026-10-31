import seaborn as sns


def labeled_barplot(data, x, y, label, hue=None, col=None, row=None, **kwargs):
    x_order = data[x].unique()
    hue_order = data[hue].unique() if hue is not None else [None]

    grid = sns.catplot(
        data=data, x=x, y=y, hue=hue, col=col, row=row,
        kind="bar", order=x_order, hue_order=hue_order if hue else None, **kwargs
    )

    keys = [k for k in (x, hue, row, col) if k is not None]
    lookup = dict(zip(data[keys].itertuples(index=False, name=None), data[label]))
    axes = grid.axes_dict if (row or col) else {(): grid.ax}

    for facet, ax in axes.items():
        facet = facet if isinstance(facet, tuple) else (facet,)
        for container, level in zip(ax.containers, hue_order):
            hue_key = () if hue is None else (level,)
            labels = [lookup.get((xi, *hue_key, *facet), "") for xi in x_order]
            ax.bar_label(container, labels=labels, padding=3, label_type="edge")

    return grid
