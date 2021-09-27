"""Make a Plot3D of a 3D arm and animate the joints.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from robotarm3d import Link, RobotArm3D

if __name__ == "__main__":

    init_q1 = 90.0
    init_q2 = 0.0
    links = [
        Link(mass=1.0, length=10.0, alt_th=init_q1, sw_phi=0.0),
        Link(mass=1.0, length=5.0, alt_th=25.0, sw_phi=init_q2),
        Link(mass=1.0, length=5.0, alt_th=-135.0, sw_phi=0.0),
    ]

    arm = RobotArm3D(links=links)  # , x0=4.0, angle0=200.0)

    arm.print_positions()

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim([-30, 30])
    ax.set_ylim([-30, 30])
    ax.set_zlim([-30, 30])

    x, y, z = arm.get_positions()
    (line,) = ax.plot(x, y, z, color="b")

    # Animation ------------
    axcolor = "lightgoldenrodyellow"
    # ax.margins(x=0)

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(bottom=0.25)  # left=0.25, bottom=0.25)

    # Make a horizontal slider to control the angles.
    axq1 = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    q1_slider = Slider(
        ax=axq1,
        label="q1",
        valmin=0,
        valmax=90,
        valinit=init_q1,
    )
    axq2 = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    q2_slider = Slider(
        ax=axq2,
        label="q2",
        valmin=0,
        valmax=90,
        valinit=init_q2,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        arm.links[0].alt_th = q1_slider.val
        arm.links[1].sw_phi = q2_slider.val
        arm.eval_positions()
        x, y, z = arm.get_positions()
        line.set_xdata(x)
        line.set_ydata(y)
        # weird hack for z-axis:
        line.set_3d_properties(zs=z)
        fig.canvas.draw_idle()

    # register the update function with slider
    q1_slider.on_changed(update)
    q2_slider.on_changed(update)

    plt.show()
