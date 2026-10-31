import pandas as pd

from .core import ModelSpec, FittedAnalysis, EMMResult
from .cld import compact_letter_display
from .r import RBackend


def fit(spec: ModelSpec, data: pd.DataFrame, slice_by: tuple[str, ...] = ()) -> FittedAnalysis:
    slice_by = list(map(str, slice_by))
    if len(slice_by) == 0:
        groups = [((), data)]
    else:
        groups = data.groupby(slice_by, sort=False, dropna=False)

    models, anovas = {}, {}
    backend = RBackend(spec)
    for raw_key, frame in groups:
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        if slice_by:
            frame = frame.drop(columns=slice_by)

        models[key] = backend.fit_model(frame)
        anovas[key] = backend.anova(models[key], type=spec.anova_type).assign(**dict(zip(slice_by, key)))

    return FittedAnalysis(tuple(slice_by), models, anovas, spec)


def emmeans(
    fitted: FittedAnalysis,
    specification: str | None = None,
    alpha: float = 0.05,
    adjust: str = "tukey",
    response_type: str = "response",
    estimate_col: str = "estimate"
) -> EMMResult:
    if specification is None:
        specification = fitted.spec.formula.split('~', 1)[-1].strip()

    emms, comparisons = {}, {}
    for key, model in fitted.models.items():
        emm_groups, pwpm_groups = RBackend.estimate(model, specification, adjust, alpha, response_type)

        for group_index, (emm, pwpm) in enumerate(zip(emm_groups, pwpm_groups)):
            emm = emm.assign(**dict(zip(fitted.slice_by, key)))

            estimate_names = ("response", "prob", "rate", "estimate", "emmean", estimate_col)
            estimate_names_valid = list(filter(lambda x: x in emm.columns, estimate_names))
            if len(estimate_names_valid) == 0:
                raise ValueError(f"Could not identify the EMM estimate column in {list(emm.columns)}")

            estimate_name = estimate_names_valid[0]
            if estimate_name != estimate_col:
                emm.rename(columns={estimate_name: estimate_col}, inplace=True)

            emm_cols = emm.columns.tolist()
            emm = emm.set_index(emm_cols[:emm_cols.index(estimate_col)])
            pwpm.columns = emm.index
            emm['cld'] = pd.Series(compact_letter_display(pwpm, emm[estimate_col], alpha))

            group_key = (*key, group_index)
            emms[group_key] = emm.reset_index()
            comparisons[group_key] = pwpm

    return EMMResult(emms, comparisons)
