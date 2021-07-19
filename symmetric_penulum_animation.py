import os
import sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def symmetric_pendulum(t, y):  # , I, M, L):
    """The symmetric pendulum is two bars attached to each other at an end.
    They rotate counter to each other around their attached ends.
    q is the angle the right bar makes with the negative vertical axis,
    i.e. if q = 0, the bars are both vertically aligned and attached at the top.

    f(q) = sin(q)*cos(q) / (I/(M*(L/2)^2) + cos(q)^2)

    For a bar, the moment of inertia is I = 1/12 * M * L^2,
    so I/(M*(L/2)^2) = 1/3, independant of M or L
    """

    def f(y):
        return np.sin(y) * np.cos(y) / (1.0 / 3.0 + np.cos(y) ** 2)

    y0p = y[1]  # y' = w
    y1p = f(y[0]) * (y[1] ** 2)  # w' = f(y) w^2
    return [y0p, y1p]


# initial coordinates and velocities:
q1_i = 0
q1p_i = 1.0
y_init = [q1_i, q1p_i]

t_i, t_f, n_tm = 0.0, 5.0, 60
sol_sp = solve_ivp(symmetric_pendulum, [t_i, t_f], y_init, dense_output=True)

t = np.linspace(t_i, t_f, n_tm)
z = sol_sp.sol(t)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

fig.suptitle("Symmetric Pendulum")

qlines = ax1.plot(t, z.T)
ax1.legend(qlines, [r"$q$", r"$\dot{q}$"], loc="upper left")
vline = ax1.axvline(t[0], ls="--", color="k")

ax2.set_xlim([-1.1, 1.1])
ax2.set_ylim([-1.1, 1.1])

x = [-np.sin(q1_i), 0.0, np.sin(q1_i)]
y = [-0.5 * np.cos(q1_i), 0.5 * np.cos(q1_i), -0.5 * np.cos(q1_i)]

(line,) = plt.plot(x, y, color="b")

ax2.set_aspect("equal", adjustable="box")

ax2.margins(x=0)


def update(val):
    q1 = z.T[val, 0]

    x = [-np.sin(q1), 0.0, np.sin(q1)]
    y = [-0.5 * np.cos(q1), 0.5 * np.cos(q1), -0.5 * np.cos(q1)]

    line.set_xdata(x)
    line.set_ydata(y)
    # vline.set_xdata([t[tm_slider.val]] * 2)
    vline.set_xdata([t[val]] * 2)
    fig.canvas.draw_idle()


# Shave off last few frames to make cycle close.
a = animation.FuncAnimation(fig, update, frames=n_tm - 3, interval=50, repeat=True)

make_html = True
if make_html:
    # Make HTML output: -------------------
    from matplotlib.animation import HTMLWriter
    import matplotlib

    # Increase size limit for html file:

    matplotlib.rcParams["animation.embed_limit"] = 2 ** 32  # 128
    a.save("symmetric_pendulum.html", writer=HTMLWriter(embed_frames=True))

    # To open file in web browser:
    # > xdg-open symmetric_pendulum.html
    # --------------------------------------
else:
    plt.show()
