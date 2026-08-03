# running command: streamlit run <filename>

from matplotlib import pyplot as plt
from matplotlib import patches
from functools import partial
import itertools as it
import streamlit as st
import operator as op
import numpy as np
import math


def to_polar(point: tuple[float, float], origin=(0, 0)):
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    d = math.sqrt(dx**2 + dy**2)
    r = math.atan2(dy, dx)
    return r, d


def partialize(points):
    return [
        list(map(op.itemgetter(i), points))
        for i in range(len(points[0]))
    ]


def center_of_mass(points):
    n = len(points)
    return tuple(map(lambda x: sum(x)/n, partialize(points)))


def arange(points):
    return sorted(
        points,
        key=partial(
            to_polar,
            origin=center_of_mass(points)
        )
    )


def plot_shape(points):
    figure, axis = plt.subplots()
    x, y = partialize(arange(points))
    axis.scatter(x, y)
    axis.scatter(*center_of_mass(points))
    axis.plot(x + [x[0]], y + [y[0]])
    return figure, axis


def binarize(x):
    return bin(int(x))[2:]


def complete_square(n):
    # perfect cube
    return [
        tuple(map(int, row.rjust(n, '0')))
        for row in map(binarize, range(2**n))
    ]


def apply_matrix(matrix, shape):
    return np.array([matrix @ shape[i] for i in range(len(shape))])


def line_by_points(p1, p2, axis, **kwargs):
    return axis.plot(
        (p1[0], p2[0]),
        (p1[1], p2[1]),
        **kwargs
    )


def plot_shape(shape, color='black', ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots()

    for point in shape:
        axis.scatter(*point, c=color)

    for p1, p2 in it.pairwise(shape):
        line_by_points(p1, p2, ax, color=color)
    line_by_points(shape[-1], shape[0], ax, color=color, label=label)

    return ax


def evaluate(phrase):
    try:
        return eval(phrase)
    except Exception:
        return None


def evaluate_line(line):
    if line.strip():
        return list(map(evaluate, line.split()))


# polygon = patches.Rectangle((0, 0), 1, 1, edgecolor='red', facecolor='none')
# axis.add_patch(polygon)


if __name__ == '__main__':
    st.set_page_config(page_title='Determinant')

    placeholder = st.empty()

    st.header('Shape Transformation by Matrix')

    c1, c2 = st.columns(2)
    c11, c12 = c1.columns(2)
    a11 = c11.number_input('A11', value=1)
    a21 = c11.number_input('A21', value=0)
    a12 = c12.number_input('A12', value=0)
    a22 = c12.number_input('A22', value=1)
    shape = c2.text_area('Shape (square by default)', height=125, placeholder='x1 y1 \nx2 y2 \nx3 y3')

    matrix = np.array([[a11, a12], [a21, a22]])
    # matrix = list(map(evaluate_line, matrix.splitlines()))
    n = len(matrix)

    if n != 2:
        placeholder.warning('Matrix must be 2D')
        st.stop()
    elif any(len(matrix[i]) != n for i in range(n)):
        placeholder.warning('Matrix must be square')
        st.stop()

    if shape:
        shape = list(map(evaluate_line, shape.splitlines()))
    else:
        shape = complete_square(n)
    shape = np.array(arange(shape))

    matrix = np.array(matrix)
    new_shape = apply_matrix(matrix, shape)

    st.subheader('')
    figure, axis = plt.subplots()
    # axis.set_title('Matrixes')

    plot_shape(shape, ax=axis, color='blue', label='Original')
    plot_shape(new_shape, ax=axis, color='red', label='Transformed')

    axis.set_box_aspect(1)
    axis.legend()
    axis.grid(True)
    # figure.tight_layout()
    st.pyplot(figure)
