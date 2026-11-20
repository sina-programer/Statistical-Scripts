import seaborn as sns


def labeled_barplot(data, x, y, label, hue=None, col=None, row=None, order=None, hue_order=None, **kwargs):
    x_order = kwargs.pop('x_order') if 'x_order' in kwargs else (order or data[x].unique())
    if hue is not None and hue_order is None:
        hue_order = data[hue].unique()

    grid = sns.catplot(
        data=data, x=x, y=y, hue=hue, col=col, row=row,
        kind="bar", order=x_order, hue_order=hue_order, **kwargs
    )

    keys = [k for k in (x, hue, row, col) if k is not None]
    lookup = dict(zip(data[keys].itertuples(index=False, name=None), data[label]))
    axes = grid.axes_dict if (row or col) else {(): grid.ax}

    for facet, ax in axes.items():
        facet = facet if isinstance(facet, tuple) else (facet,)
        for container, level in zip(ax.containers, hue_order or [None]):
            hue_key = () if hue is None else (level,)
            labels = [lookup.get((xi, *hue_key, *facet), "") for xi in x_order]
            ax.bar_label(container, labels=labels, padding=3, label_type="edge")

    return grid
