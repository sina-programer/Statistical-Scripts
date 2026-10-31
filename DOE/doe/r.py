from abc import ABC, abstractmethod

from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from rpy2 import robjects as ro
import pandas as pd

from .core import ModelSpec


PACKAGE_ENGINES = {
    'lmer': 'lme4',
    'glmer': 'lme4',
    'glmmTMB': 'glmmTMB'
}


class RBackend:
    def __init__(self, spec: ModelSpec):
        extra_packages = {'car'}

        engine_package = PACKAGE_ENGINES.get(spec.engine)
        if engine_package is not None:
            extra_packages.add(engine_package)

        if "::" in spec.family:
            extra_packages.add(spec.family.split("::", 1)[0])

        self.spec = spec
        self.packages = list(set(spec.packages) | extra_packages)

    def fit_model(self, data: pd.DataFrame):
        checks = "\n".join(
            f'requireNamespace("{pkg}", quietly=TRUE) || stop("Install R package: {pkg}")'
            for pkg in self.packages
        )

        return ro.r(RScripts.FIT)(
            DataFrameConverter.to_r(data),
            self.spec.formula,
            self.spec.engine,
            self.spec.family,
            self.spec.link,
            checks
        )

    @classmethod
    def anova(cls, model, type=3):
        out = ro.r(RScripts.ANOVA)(model, type)
        return DataFrameConverter.to_py(out)

    @classmethod
    def estimate(cls, model, specification, adjust, alpha, response_type):
        out = ro.r(RScripts.EMM)(model, specification, adjust, alpha, response_type or "")
        emms = [DataFrameConverter.to_py(item) for item in out.rx2("emm")]
        pwpms = [DataFrameConverter.to_py(item).apply(RBackend._strip_compares) for item in out.rx2('pw')]
        return emms, pwpms

    @classmethod
    def _strip_compares(cls, series: pd.Series) -> pd.Series:
        return pd.to_numeric(series.astype(str).str.replace('<', '0'), errors='coerce')


class R2PyConverter(ABC):

    @classmethod
    @abstractmethod
    def to_py(cls, obj): raise NotImplemented

    @classmethod
    @abstractmethod
    def to_r(cls, obj): raise NotImplemented


class DataFrameConverter(R2PyConverter):

    @classmethod
    def to_py(cls, obj):
        with localconverter(ro.default_converter + pandas2ri.converter):
            return ro.conversion.get_conversion().rpy2py(obj)

    @classmethod
    def to_r(cls, obj: pd.DataFrame):
        with localconverter(ro.default_converter + pandas2ri.converter):
            return ro.conversion.get_conversion().py2rpy(obj)


class RScripts:
    FIT = """
function(data, formula_text, engine, family_name, link_name, checks) {
    eval(parse(text=checks))
    formula <- stats::as.formula(formula_text)

    if (engine == "lm") return(stats::lm(formula, data=data))
    if (engine == "lmer") return(lme4::lmer(formula, data=data))

    family_parts <- strsplit(family_name, "::", fixed=TRUE)[[1]]
    family_function <- getExportedValue(family_parts[[1]], family_parts[[2]])
    family <- do.call(family_function, list(link=link_name))

    if (engine == "glm") return(stats::glm(formula, data=data, family=family))
    if (engine == "glmer") return(lme4::glmer(formula, data=data, family=family))
    if (engine == "glmmTMB") return(glmmTMB::glmmTMB(formula, data=data, family=family))

    stop("Unsupported model engine: ", engine)
}
    """

    ANOVA = """
function(model, type){
    table <- as.data.frame(car::Anova(model, type=type))
    data.frame(
        term=rownames(table),
        table,
        check.names=FALSE,
        row.names=NULL
    )
}
    """

    EMM = """
symmetric_pwpm <- function(pwpm){
    p <- matrix(1, nrow=nrow(pwpm), ncol=ncol(pwpm))
    rownames(p) <- rownames(pwpm)
    colnames(p) <- colnames(pwpm)
    dimnames(p) <- dimnames(pwpm)
    p[upper.tri(p)] <- pwpm[upper.tri(pwpm)]
    p[lower.tri(p)] <- t(p)[lower.tri(p)]
    as.data.frame(p)
}

function(model, spec, adjustment, alpha, response_type) {
    emm <- emmeans::emmeans(model, stats::as.formula(spec), type=response_type)
    emm_table <- as.data.frame(base::summary(emm))
    by_vars <- emm@misc$by.vars
    emm_groups <- if (length(by_vars)) {
        split(emm_table, do.call(interaction, c(
            unname(emm_table[by_vars]), list(drop=TRUE, lex.order=TRUE, sep="|")
        )))
    } else {
        list(emm_table)
    }

    pw <- emmeans::pwpm(emm, adjust=adjustment, means=TRUE, diffs=FALSE)
    if (is.matrix(pw)) pw <- list(pw)

    list(
        emm = lapply(emm_groups, as.data.frame),
        pw = lapply(pw, symmetric_pwpm)
    )
}
    """
