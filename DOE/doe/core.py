from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    formula: str
    engine: str = "lm"
    family: str = "gaussian"
    link: str = "identity"
    anova_type: Literal[2, 3] = 3
    packages: tuple[str, ...] = ()


@dataclass(frozen=True)
class FittedAnalysis:
    slice_by: tuple[str, ...]
    models: dict[tuple[Any, ...], Any]
    anovas: dict[tuple[Any, ...], pd.DataFrame]
    spec: ModelSpec

    def anova(self) -> pd.DataFrame:
        if not self.anovas:
            return pd.DataFrame()
        return pd.concat(self.anovas.values(), ignore_index=True)


@dataclass
class EMMResult:
    emms: dict[tuple[Any, ...], pd.DataFrame]
    pwpm: dict[tuple[Any, ...], pd.DataFrame]

    def emm(self) -> pd.DataFrame:
        if not self.emms:
            return pd.DataFrame()
        return pd.concat(self.emms.values(), ignore_index=True)
