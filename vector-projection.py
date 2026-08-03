# run with: streamlit run <filename>

from matplotlib.patches import Circle
from matplotlib import pyplot as plt
from functools import partial
import streamlit as st
import operator as op
import numpy as np
import math

D = 3  # number of decimal places

is_negative = partial(op.ge, 0)
is_positive = partial(op.le, 0)

def dot(x, y):
    return np.sum(np.multiply(x, y))

def norm(vector):
    return math.sqrt(dot(vector, vector))

def unitize(vector):
    return np.array(vector) / norm(vector)

def project(x, y):
    a = dot(x, y) / dot(y, y)
    projection = a * y
    error = x - projection
    return dict(a=a, projection=projection, error=error)

def orthogonalize(x, y):
    u = unitize(x)
    p = project(y, u)
    v = unitize(p['error'])
    return u, v

def plot_vector(vector, v=None, **kwargs):
    plt.plot(*[(0 if v is None else v[i], vector[i]) for i in range(len(vector))], **kwargs)

def represent_vector(vector, d=D):
    return '(' + ', '.join(vector.round(d).astype(str)) + ')'

def vector_to_latex(vector, d=D):
    rows = r' \\ '.join(vector.round(d).astype(str))
    return r'\begin{pmatrix} ' + rows + r' \end{pmatrix}'

def is_vector_unit(vector):
    return dot(vector, vector) == 1

def cosine_similarity(x, y):
    return dot(x, y) / (norm(x) * norm(y))

def get_legend_loc(*vectors):
    x = list(map(op.itemgetter(0), vectors))
    y = list(map(op.itemgetter(1), vectors))
    x_pos = all(map(is_positive, x))
    x_neg = all(map(is_negative, x))
    y_pos = all(map(is_positive, y))
    y_neg = all(map(is_negative, y))
    if x_neg or y_neg:
        return 1  # upper right
    elif x_pos:
        return 2  # upper left
    elif y_pos:
        return 4  # lower right
    return 0  # best


if __name__ == "__main__":
    st.set_page_config(page_title='Vector Projection')

    placeholder = st.empty()

    c1, c2 = st.columns(2)
    v_raw = c1.text_input('vector V')
    u_raw = c2.text_input('vector U')

    if not v_raw and not u_raw:
        placeholder.info("you must enter vectors' items separated by space")
        st.stop()

    v = np.array(list(map(float, v_raw.split())))
    u = np.array(list(map(float, u_raw.split())))

    if np.shape(u) != np.shape(v):
        placeholder.error('dimensions of U & V must be equal!')
        st.stop()

    st.header('Projection of V on U')

    p = project(v, u)
    u_max = np.max(np.abs(u)).item()
    v_max = np.max(np.abs(v)).item()
    maximum = max(u_max, v_max)
    r = int(math.ceil(maximum) + 1)
    o = np.zeros_like(u)
    n = len(u)

    projection = p['projection']
    error = p['error']
    a = p['a']

    norm_v = norm(v)
    norm_u = norm(u)
    norm_projection = norm(projection)
    norm_error = norm(error)
    # norm_ceiling = max(norm_v, norm_u, norm_projection, norm_error)

    if norm_u == 0 or norm_v == 0:
        placeholder.error('projection process on a zero vector is not valid')
        st.stop()

    v_normed = v / norm_v
    u_normed = u / norm_u

    p_normed = project(v_normed, u_normed)
    projection_normed = p_normed['projection']
    error_normed = p_normed['error']
    a_normed = p_normed['a']

    norm_v_normed = norm(v_normed)
    norm_u_normed = norm(u_normed)
    norm_projection_normed = norm(projection_normed)
    norm_error_normed = norm(error_normed)

    is_perp = a == 0
    is_linear = norm_error == 0

    cosine = cosine_similarity(v, u)
    angle = math.acos(cosine) / math.pi * 180


    cs = st.columns(4)
    cs[0].metric('coeff', round(p['a'], D))
    cs[1].metric('<U, V>', round(dot(u, v), D))
    cs[2].metric('<U, U>', round(dot(u, u), D))
    cs[3].metric('<V, V>', round(dot(v, v), D))


    st.subheader('Original Vectors (with norms)')
    cs = st.columns(4)

    cs[0].latex(r"\textbf{V}")
    cs[0].latex(vector_to_latex(v))
    cs[0].latex(str(round(norm_v, D)))

    cs[1].latex(r"\textbf{U}")
    cs[1].latex(vector_to_latex(u))
    cs[1].latex(str(round(norm_u, D)))

    cs[2].latex(r"\textbf P^v_u")
    cs[2].latex(vector_to_latex(projection))
    cs[2].latex(str(round(norm_projection, D)))

    cs[3].latex(r"\textbf P^{v}_{u^{\perp}}")
    cs[3].latex(vector_to_latex(error))
    cs[3].latex(str(round(norm_error, D)))


    st.subheader('Unit Vectors')
    cs = st.columns(4)

    cs[0].latex(vector_to_latex(v_normed))
    cs[0].latex(str(round(norm_v_normed, D)))

    cs[1].latex(vector_to_latex(u_normed))
    cs[1].latex(str(round(norm_u_normed, D)))

    cs[2].latex(vector_to_latex(projection_normed))
    cs[2].latex(str(round(norm_projection_normed, D)))

    cs[3].latex(vector_to_latex(error_normed))
    cs[3].latex(str(round(norm_error_normed, D)))



    st.header('Orthogonal Vectors')


    st.subheader('V-Base')
    cs = st.columns(4)

    v1 = v_normed
    p_u_v1 = project(u, v1)
    v2 = unitize(p_u_v1['error'])

    cs[0].latex(r"\textbf V1")
    cs[0].latex(vector_to_latex(v1))

    cs[1].latex(r"\textbf V2")
    cs[1].latex(vector_to_latex(v2))

    cs[2].latex(r"\textbf P^u_{v_1}")
    cs[2].latex(vector_to_latex(p_u_v1['projection']))

    cs[3].latex(r"\textbf P^u_{v^{\perp}_1}")
    cs[3].latex(vector_to_latex(p_u_v1['error']))


    st.subheader('U-Base')
    cs = st.columns(4)

    u1 = u_normed
    p_v_u1 = project(v, u1)
    u2 = unitize(p_v_u1['error'])

    cs[0].latex(r"\textbf U1")
    cs[0].latex(vector_to_latex(u1))

    cs[1].latex(r"\textbf U2")
    cs[1].latex(vector_to_latex(u2))

    cs[2].latex(r"\textbf P^v_{u_1}")
    cs[2].latex(vector_to_latex(p_v_u1['projection']))

    cs[3].latex(r"\textbf P^v_{u^{\perp}_1}")
    cs[3].latex(vector_to_latex(p_v_u1['error']))


    if n in [2, 3]:
        st.header('Plot')
        figure = plt.figure()
        axis = figure.add_subplot(projection='3d' if n==3 else 'rectilinear')
        axis.set_box_aspect(1)

        if n==2:
            unit_circle = Circle(o, radius=1, fill=False, alpha=0.9, linewidth=0.5)
            axis.add_patch(unit_circle)

            axis.quiver(*o, *u, color='cyan', scale=1, scale_units='xy', units='xy', label='U')
            axis.quiver(*o, *v, color='dodgerblue', scale=1, scale_units='xy', units='xy', label='V')

            axis.text(*(u + v)/2, f"{angle:.2f}°")
            axis.plot(*list(zip(o, projection)), 'g--', linewidth=1, label='Project')
            axis.plot(*list(zip(projection, projection+error)), 'r--', linewidth=1, label='Error')

            if not (is_perp or is_linear):
                axis.plot(*list(zip(o, v1)), color='navy', label='V1', linewidth=1, linestyle=':')
                axis.plot(*list(zip(o, v2)), color='navy', label='V2', linewidth=1, linestyle='-.')

                axis.plot(*list(zip(o, u1)), color='c', label='U1', linewidth=1, linestyle=':')
                axis.plot(*list(zip(o, u2)), color='c', label='U2', linewidth=1, linestyle='-.')

            step = max(r // 5, 1)
            ticks = list(range(-r, r+step, step))
            plt.xticks(ticks)
            plt.yticks(ticks)
            plt.legend(loc=get_legend_loc(u, v))

        elif n==3:
            placeholder.info('3D graphs are not supported at the moment')

        figure.tight_layout()
        axis.grid(True)
        st.pyplot(figure)

    else:
        placeholder.warning(f'plotting is not supported for {n} dimensional vectors (only 2D available)')


    if is_perp:
        placeholder.info('V is perpendicular on U')

    if is_linear:
        placeholder.info('V and U are linear dependent')
