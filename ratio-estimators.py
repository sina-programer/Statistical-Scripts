# streamlit run <filename>

import streamlit as st
import numpy as np


def vector_to_latex(vector):
    rows = r' \\ '.join(vector.astype(str))
    return r'\begin{pmatrix} ' + rows + r' \end{pmatrix}'


def form(value):
    return str(
        format(
            round(
                value,
                D
            ),
            ','
        )
    )


st.set_page_config(page_title='Sampling - Ratio Estimators', layout='wide')

placeholder = st.empty()

with st.sidebar:
    UNIT = st.number_input('Unit', 1e-12, value=1.)
    D = st.number_input('Decimals', 0, 6, value=2)
    N = st.number_input('Population Size', 0, placeholder='set zero for infinite population')
    if N == 0:
        N = float('inf')

x_raw = st.text_input('x').strip()
y_raw = st.text_input('y').strip()

if not x_raw and not y_raw:
    placeholder.info("first you must enter your numbers separated by space")
    st.stop()


cs = st.columns(6)

if x_raw:
    x = np.array(list(map(float, x_raw.split()))) / UNIT
    xbar = np.mean(x)
    sumx = np.sum(x)
    sumx2 = np.sum(np.square(x))
    varx = np.var(x, ddof=1)
    stdx = np.sqrt(varx)
    cvx = stdx / xbar

    cs[0].latex(r"\bar{x} = " + form(xbar))
    cs[1].latex(r"\sum{x_i} = " + form(sumx))
    cs[2].latex(r"\sum x^2_i = " + form(sumx2))
    cs[3].latex(r"s^2_x = " + form(varx))
    cs[4].latex(r"s_x = " + form(stdx))
    cs[5].latex(r"CV_x = " + form(cvx))

if y_raw:
    y = np.array(list(map(float, y_raw.split()))) / UNIT
    ybar = np.mean(y)
    sumy = np.sum(y)
    sumy2 = np.sum(np.square(y))
    vary = np.var(y, ddof=1)
    stdy = np.sqrt(vary)
    cvy = stdy / ybar

    cs[0].latex(r"\bar{y} = " + form(ybar))
    cs[1].latex(r"\sum{y_i} = " + form(sumy))
    cs[2].latex(r"\sum y^2_i = " + form(sumy2))
    cs[3].latex(r"s^2_y = " + form(vary))
    cs[4].latex(r"s_y = " + form(stdy))
    cs[5].latex(r"CV_y = " + form(cvy))



st.header('Estimation')

n = len(x)
coef = (1 - n / N)


if y_raw:
    cs = st.columns(2)

    cs[0].latex(r"\LARGE \hat{\bar{Y}}_{srs}")
    cs_ = cs[0].columns(3)
    char = r"y_{srs}"
    char_bar = r"\bar " + char
    ev_yb = coef * vary / n
    es_yb = np.sqrt(ev_yb)
    cs_[0].latex(r"\bar y_{srs} = " + form(ybar))
    cs_[1].latex(r"\hat {Var}(\bar y_{srs}) = " + form(ev_yb))
    cs_[2].latex(r"\hat {SE}(\bar y_{srs}) = " + form(es_yb))

    cs[1].latex(r"\LARGE \hat{Y}_{+srs}")
    cs_ = cs[1].columns(3)
    ev_yp = N**2 * ev_yb
    es_yp = np.sqrt(ev_yp)
    cs_[0].latex(r"N \bar y_{srs} = " + form(ybar * N))
    cs_[1].latex(r"\hat {Var}(N \bar y_{srs}) = " + form(ev_yp))
    cs_[2].latex(r"\hat {SE}(N \bar y_{srs}) = " + form(es_yp))


if x_raw:
    cs = st.columns(2)

    cs[0].latex(r"\LARGE \hat{\bar{X}}_{srs}")
    cs_ = cs[0].columns(3)
    char = r"x_{srs}"
    char_bar = r"\bar " + char
    ev_xb = coef * varx / n
    es_xb = np.sqrt(ev_xb)
    cs_[0].latex(r"\bar x_{srs} = " + form(xbar))
    cs_[1].latex(r"\hat {Var}(\bar x_{srs}) = " + form(ev_xb))
    cs_[2].latex(r"\hat {SE}(\bar x_{srs}) = " + form(es_xb))

    cs[1].latex(r"\LARGE \hat{X}_{+srs}")
    cs_ = cs[1].columns(3)
    ev_xp = N**2 * ev_xb
    es_xp = np.sqrt(ev_xp)
    cs_[0].latex(r"N \bar x_{srs} = " + form(xbar * N))
    cs_[1].latex(r"\hat {Var}(N \bar x_{srs}) = " + form(ev_xp))
    cs_[2].latex(r"\hat {SE}(N \bar x_{srs}) = " + form(es_xp))



if not (x_raw and y_raw):
    placeholder.error('you must enter both x & y')
    st.stop()

if np.shape(x) != np.shape(y):
    placeholder.warning('dimensions of X & Y must be equal for further estimations')
    st.stop()


st.latex(r'\LARGE \hat{R}')
cs = st.columns(6)

dx = x - xbar
dy = y - ybar
cov = np.sum(dx * dy) / (n-1)
rho = cov / (stdx * stdy)
rho_floor = 0.5 * cvx / cvy
r = sumy / sumx
z = y - r*x
zbar = np.mean(z)
sumz = np.sum(z)
# varz = np.var(z, ddof=1)
varz = vary + r**2*varx - (2*r*cov)

ev_r = (1 / (xbar ** 2)) * coef * varz / n
es_r = np.sqrt(ev_r)

cs[0].latex(r"r = " + form(r))
cs[1].latex(r"\hat {Var(r)} = " + form(ev_r))
cs[2].latex(r"\hat {SE}(r) = " + form(es_r))
# cs[1].latex(r"net = " + ('+' if rr>0 else '-' if rr<0 else '') + form(rr) + ' \%')
# cs[1].latex(r"R \in (" + form(rr - es_r*100) + ', ' + form(rr + es_r*100) + ')')
cs[3].latex(r"\hat S^2_z = " + form(varz))
cs[4].latex(r"s_{xy} = " + form(cov))
cs[5].latex(form(rho) + r" = \hat\rho > \frac{1}{2} \frac{CV_x}{CV_y} = " + form(rho_floor))




st.latex(r"\LARGE \hat Y_r")
cs = st.columns(5)

Xbar = cs[0].number_input('X-Bar', 0.0)
if Xbar > 0:
    ybar_r = r * Xbar
    var_ybar_r = coef * varz / n
    std_ybar_r = np.sqrt(var_ybar_r)
    rp_ybar_r = ev_yb / var_ybar_r
    cs[1].latex(r"\bar y_r = " + form(ybar_r))
    cs[2].latex(r"\hat {Var}(\bar y_r) = " + form(var_ybar_r))
    cs[3].latex(r"\hat {SE}(\bar y_r) = " + form(std_ybar_r))
    cs[4].latex(r"RP(\bar y_r | \bar y_{srs}) = " + form(rp_ybar_r))

cs = st.columns(5)
Xplus = cs[0].number_input('X-Plus', 0.0)
if Xplus > 0:
    sumy_r = r * Xplus
    var_yplus_r = N**2 * coef * varz / n
    std_yplus_r = np.sqrt(var_yplus_r)
    rp_yplus_r = ev_yp / var_yplus_r
    cs[1].latex(r"y_{+r} = " + form(sumy_r))
    cs[2].latex(r"\hat {Var}(y_{+r}) = " + form(var_yplus_r))
    cs[3].latex(r"\hat {SE}(y_{+r}) = " + form(std_yplus_r))
    cs[4].latex(r"RP(y_{+r} | y_{+srs}) = " + form(rp_yplus_r))


if n < 8:
    placeholder.warning('your sample size is too small')
