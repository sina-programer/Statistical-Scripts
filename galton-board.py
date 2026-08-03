from matplotlib.patches import Circle
from matplotlib import pyplot as plt
from matplotlib import animation
import numpy as np
import random

def get_normal_pdf(mu, sigma):

    def pdf(x):
        s = (x - mu) / sigma
        denom = sigma * np.sqrt(2 * np.pi)
        return np.exp(-np.square(s) / 2) / denom

    return pdf

def bulky_simulate(n=1000, levels=10):
    U = np.random.random((n, levels))
    D = np.where(U<0.5, -1, 1)  # directions
    P = np.sum(D, axis=1)  # positions
    return P


class GaltonBoard:
    def __init__(self, levels=10, drop_interval=1.0):
        mod = levels % 2
        if mod == 1:
            levels += mod
            print('WARNING: value of level changed to <{}>'.format(levels))

        self.levels = int(levels)
        self.drop_interval = float(drop_interval)

        self.ball_drop_timer = 0
        self.active_balls = []
        self.bin_counts = np.zeros(levels+1, dtype=int)
        self.bin_centers = np.arange(levels + 1) - levels/2

        self.peg_positions = []
        for row in range(levels+1):
            pegs_in_row = row + 1
            for col in range(pegs_in_row):
                x = col - row/2
                y = levels - row + 0.5
                self.peg_positions.append((x, y))

        self.half_width = self.levels/2 + 0.55
        self.half_range = (self.levels+1) / 2

        fig_height = (self.levels + 3) * 0.75
        self.fig, (self.ax1, self.ax2) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(self.half_width*2, fig_height),
            gridspec_kw={
                'height_ratios': [3, 1]
            }
        )

        self.setup_plot()
        # self.fig.tight_layout()

    def setup_plot(self):
        self.ax1.set_xlim(-self.half_width, self.half_width)
        self.ax1.set_ylim(-1.5, self.levels + 1.5)
        self.ax1.set_aspect('equal')
        self.ax1.grid(False)
        self.ax1.axis('off')

        for (x, y) in self.peg_positions:
            circle = Circle((x, y), 0.2, color='gray', fill=True, alpha=0.6, zorder=1)
            self.ax1.add_patch(circle)

        self.ax1.vlines(np.arange(self.levels+2) - self.half_range, -1, -1.25, color='black', linewidth=1.5, zorder=1)
        self.ax1.hlines(-1, -self.half_range, self.half_range, 'k', linewidth=2, zorder=1)

        for bin_pos in self.bin_centers:
            self.ax1.text(bin_pos, -1.5, str(int(bin_pos)), va='center', ha='center')

        self.bar_rects = self.ax2.bar(self.bin_centers, self.bin_counts, width=0.9, edgecolor='black', alpha=0.7)
    
        self.ax2.set_xlim(-self.half_width, self.half_width)
        self.ax2.set_ylim(0, 5)
        self.ax2.set_xlabel('Bin Position')
        self.ax2.set_ylabel('Count')
        self.ax2.set_title('Distribution of Balls')
        self.ax2.grid(axis='y', alpha=0.3)
        self.ax2.set_xticks(self.bin_centers)

        self.normal_curve, = self.ax2.plot([], [], 'r-', linewidth=1.5)

        self.info_text = self.ax1.text(
            0.02,
            0.98,
            'Balls: 0\nMean: nan\nStd: nan',
            transform=self.ax1.transAxes,
            va='top',
            fontsize=10,
            bbox={
                'boxstyle': 'round',
                'facecolor': 'wheat',
                'alpha': 0.8
            }
        )

        plt.subplots_adjust(hspace=0.3, left=0.1, right=0.95, top=0.95, bottom=0.1)

    def drop_ball(self):
        start_x = 0.0
        start_y = self.levels + 0.8

        ball = Circle((start_x, start_y), 0.15, color='red', fill=True, alpha=1.0, zorder=3)
        self.ax1.add_patch(ball)
        self.active_balls.append([ball, start_x, start_y, 0])

    def update_ball(self, ball_data):
        ball, x, y, row = ball_data

        if row >= self.levels:
            if y > -0.8:
                y -= 0.15
                ball.center = (x, y)
                return [ball, x, y, row]
            else:
                d = (self.levels+1) / 2
                bin_index = int(np.floor(x + d))
                bin_index = max(0, min(self.levels, bin_index))
                self.bin_counts[bin_index] += 1
                ball.remove()
                return None

        target_y = self.levels - row - 0.5

        if y > target_y + 0.1:
            y -= 0.15
        else:
            if random.random() < 0.5:
                x += 0.5
            else:
                x -= 0.5
            row += 1

        ball.center = (x, y)
        return [ball, x, y, row]

    def update_normal_curve(self):
        total_balls = int(self.bin_counts.sum())
        if total_balls <= 1:
            if total_balls == 1:
                self.info_text.set_text(
                    (
                        "Balls: {}"
                        "\nMean: {:.3f}"
                        "\nStd: 0"
                    ).format(total_balls, self.bin_centers[self.bin_counts.argmax()])
                )
            return

        mean = np.average(self.bin_centers, weights=self.bin_counts)
        variance = np.average((self.bin_centers - mean)**2, weights=self.bin_counts)
        std = np.sqrt(variance)
        pdf = get_normal_pdf(mean, std)

        info = (
            "Balls: {}"
            "\nMean: {:.3f}"
            "\nStd: {:.3f}"
        ).format(total_balls, mean, std)
        self.info_text.set_text(info)

        X = np.linspace(-self.half_range, self.half_range, 200)
        Y = pdf(X)

        scaling_factor = total_balls * ((self.levels+1) / len(self.bin_centers))
        Y = Y * scaling_factor

        self.normal_curve.set_data(X, Y)

    def update_histogram_ylim(self):
        current_max = np.max(self.bin_counts)
        if current_max > 0:
            new_ylim = max(5, current_max * 1.2)
            self.ax2.set_ylim(0, new_ylim)

    def update(self, frame):
        self.ball_drop_timer += 1
        if self.ball_drop_timer >= self.drop_interval * 20:
            self.drop_ball()
            self.ball_drop_timer = 0

        updated_balls = []
        for ball_data in self.active_balls:
            if ball_data is not None:
                result = self.update_ball(ball_data)
                if result is not None:
                    updated_balls.append(result)
        self.active_balls = updated_balls

        for rect, h in zip(self.bar_rects, self.bin_counts):
            rect.set_height(h)

        self.update_histogram_ylim()
        self.update_normal_curve()

        self.fig.canvas.draw()
        return []
    
    def animate(self):
        anim = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=None,
            interval=10,
            blit=False,
            repeat=False,
            cache_frame_data=False
        )
        plt.show()
        return anim


if __name__ == "__main__":
    gb = GaltonBoard(levels=10, drop_interval=0.25)
    gb.animate()
