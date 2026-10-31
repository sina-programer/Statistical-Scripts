from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np


OUTPUT = "minkowski_contour.gif"
A = 2.0  # each axis will be considered in range [-A, A]
DPI = 120
FPS = 25
N_FRAMES = 300
RESOLUTION = 100   # grid points per axis
WIDTH, HEIGHT = 6.5, 6
CONTOURS = [0.5, 1.5, 2]
UNIFORMITY = 0.4  # 1 = visually uniform change, 0 = constant p step per frame
P_START = 0.5
P_END = 5.0

VMAX = 2 * A
SPAN = np.linspace(-A, A, RESOLUTION)
X, Y = np.meshgrid(SPAN, SPAN)


def minkowski(x, y, p):
    return (np.abs(x) ** p + np.abs(y) ** p) ** (1.0 / p)


def uniform_arc_length(n_frames, fine_steps=2000):
    """Pick p values so that the visible change between frames is constant.

    The change between two p values is the RMS difference of the displayed
    (clipped) distance fields. Accumulating it over a fine p grid gives a
    'visual arc length' s(p); frames are placed at equal steps of s.
    """

    p_fine = np.linspace(P_START, P_END, fine_steps)

    steps = np.empty(fine_steps)
    steps[0] = 0.0
    previous = np.minimum(minkowski(X, Y, p_fine[0]), VMAX)
    for i, p in enumerate(p_fine[1:], start=1):
        current = np.minimum(minkowski(X, Y, p), VMAX)
        steps[i] = np.sqrt(np.mean((current - previous) ** 2))
        previous = current

    arc = np.cumsum(steps ** UNIFORMITY)
    return np.interp(np.linspace(0, arc[-1], n_frames), arc, p_fine)


def main():
    print('Note that running this script may take high resources.')

    # P_STEP = 0.01
    # P = np.arange(P_START, P_END + P_STEP/2, P_STEP)
    P = uniform_arc_length(N_FRAMES)

    levels = np.linspace(0, VMAX, 41)
    norm = Normalize(0, VMAX)

    fig, ax = plt.subplots(
        figsize=(WIDTH, HEIGHT),
        dpi=DPI,
        layout='constrained'
    )

    fig.colorbar(
        cm.ScalarMappable(norm, "viridis"),
        label="Minkowski distance from (0, 0)",
        extend="max",
        ax=ax
    )

    def draw(p):
        ax.clear()
        Z = minkowski(X, Y, p)
        ax.contourf(X, Y, Z, levels=levels, cmap="viridis", norm=norm, extend="max")
        ax.contour(X, Y, Z, levels=CONTOURS, colors="white", linewidths=0.5, alpha=0.6)
        ax.contour(X, Y, Z, levels=[1.0], colors="red", linewidths=2)
        ax.set(
            xlim=(-A, A),
            ylim=(-A, A),
            xlabel="x",
            ylabel="y",
            aspect="equal",
            title=f"Minkowski Distance\nred curve: unit ball (d = 1)\np = {p:.2f}"
        )

    anim = FuncAnimation(fig, draw, frames=P, repeat=False)
    anim.save(OUTPUT, writer=PillowWriter(fps=FPS))


if __name__ == "__main__":
    main()
