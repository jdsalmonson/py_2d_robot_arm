import numpy as np
from scipy.integrate import solve_ivp

from two_link_dyn_arm import TwoLinkDynArm, step_func
from robotarm2d import Link, RobotArm2DwDyn

init_q1 = 30.0
links = [
    Link(mass=1.0, length=0.3, angle=init_q1),
    Link(mass=1.0, length=0.2, angle=90.0),
    Link(mass=1.0, length=0.1, angle=-135.0),
]


def torque1(t: float) -> float:
    return step_func(t, 1.0, 2.0)


def torque2(t: float) -> float:
    return step_func(t, 2.5, 3.0, 0.1)


# arm = RobotArm2D(links=links)  # , x0=4.0, angle0=200.0)
arm = RobotArm2DwDyn(links=links[0:2], T1=torque1, T2=torque2, k1=0.0, k2=0.0)

# initial coordinates and velocities:
q1_i = np.radians(arm.link_coords[0].link.angle)
q1p_i = 0.0
q2_i = np.radians(arm.link_coords[1].link.angle)
q2p_i = 0.0
y_init = [q1_i, q1p_i, q2_i, q2p_i]

t_i, t_f, n_tm = 0.0, 5.0, 300
sol_tla = solve_ivp(arm.tlda.ODE_RHS, [t_i, t_f], y_init, dense_output=True)
t = np.linspace(t_i, t_f, n_tm)
z = sol_tla.sol(t)


import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
# (line,) = plt.plot(t, f(t, init_amplitude, init_frequency), lw=2)

qlines = ax1.plot(t, z.T)  # , label=["q1", "q1p", "q2", "q2p"])
qlines += [
    ax1.plot(t, torque1(t)),
    ax1.plot(t, torque2(t)),
]
vline = ax1.axvline(t[0], ls="--", color="k")
ax1.legend(qlines, ["q1", "q1p", "q2", "q2p", "torque1", "torque2"])

# fig = plt.figure()
# ax = fig.add_subplot()
ax2.set_xlim([-0.6, 0.6])
ax2.set_ylim([-0.6, 0.6])

arm.print_positions()
# arm.plot_positions(color="b")
x, y = arm.get_positions()
(line,) = plt.plot(x, y, color="b")

ax2.set_aspect("equal", adjustable="box")

axcolor = "lightgoldenrodyellow"
ax2.margins(x=0)

# adjust the main plot to make room for the sliders
plt.subplots_adjust(bottom=0.25)  # left=0.25, bottom=0.25)

# Make a horizontal slider to control the frequency.
axq1 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
tm_slider = Slider(
    ax=axq1,
    label="tm",
    valmin=0,
    valmax=n_tm - 1,
    valstep=1,
    valinit=0,
)

# The function to be called anytime a slider's value changes
def update(val):
    # arm.link_coords[0].link.angle = q1_slider.val
    # arm.link_coords[0].w_angle = np.degrees(z.T[tm_slider.val, 0])
    # arm.link_coords[1].w_angle = np.degrees(z.T[tm_slider.val, 2])
    arm.link_coords[0].link.angle = np.degrees(z.T[tm_slider.val, 0])
    arm.link_coords[1].link.angle = np.degrees(z.T[tm_slider.val, 2])
    arm.eval_positions()
    x, y = arm.get_positions()
    line.set_xdata(x)
    line.set_ydata(y)
    vline.set_xdata([t[tm_slider.val]] * 2)
    fig.canvas.draw_idle()


# register the update function with slider
tm_slider.on_changed(update)

plt.show()
