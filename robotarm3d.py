import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt


@dataclass
class Link:
    """A robot arm link

    Args:
      mass (float): arm link mass [gm]
      length (float): arm link length [cm]
      alt_th (float): altitude arm link angle wrt upstream joint[degrees]
      sw_phi (float): swivel angle around vector of upstream joint
      motor_mass (float): mass of upper-link motor [gm] (default: 0)
    """

    mass: float
    length: float
    alt_th: float  # [degrees]
    sw_phi: float  # [degrees]
    motor_mass: float = 0.0

    def __post_init__(self):
        """The moment of inertia of a rod spinning about its mid-length (center-of-mass)"""
        self.momI = self.mass * self.length ** 2 / 12.0

        self.set_local_position()

    def set_local_position(self):
        """set rotation matrix and local frame end effector position"""

        theta = np.radians(self.alt_th)
        R_z = R.from_rotvec(theta * np.array([0, 0, 1]))
        phi = np.radians(self.sw_phi)
        R_y = R.from_rotvec(phi * np.array([0, 1, 0]))
        self.loc_rot = R.from_matrix(R_y.as_matrix() @ R_z.as_matrix())

        pos0 = np.array([0, self.length, 0])  # link pointing upward along y-axis
        self.loc_pos = self.loc_rot.apply(pos0)

    def get_position(
        self, pos0=np.array([0, 0, 0]), rot0=R.from_matrix(np.identity(3))
    ):
        """apply upstream (toward base) position, pos0, and rotation, rot0,
        to get new position and rotation"""

        self.set_local_position()

        self.rot = R.from_matrix(rot0.as_matrix() @ self.loc_rot.as_matrix())
        self.pos = pos0 + rot0.apply(self.loc_pos)

        return self.pos, self.rot


class RobotArm3D(object):
    """The position vector of the end effector, pos_n, of the nth link is:

    pos_1 = (R_y @ R_z)_1 @ pos_1'
    pos_2 = (R_y @ R_z)_1 @ [pos_1' + (R_y @ R_z)_2 @ pos_2']
    pos_3 = (R_y @ R_z)_1 @ [pos_1' + (R_y @ R_z)_2 @ [pos_2' + (R_y @ R_z)_3 @ pos_3']]
    ...

    where pos_n' is length_n*[0,1,0] ; the length of that link pointed in y.

    R_z rotates around z-axis first, to adjust alt_theta angle, then R_y to rotate around
    the link's vertical, y-axis.

    Note that we technically start this sum from 0 with the position and orientation of
    the base of the arm, pos0, rot0, but those are trivial by default:
    pos_1 = rot0 @ [pos0 + (R_y @ R_z)_1 @ pos_1']
    ...
    """

    def __init__(
        self, links=[], pos0=np.array([0, 0, 0]), rot0=R.from_matrix(np.identity(3))
    ):
        """
        Args:
          links (list): list of links that make up the arm
          pos0 (array): position of arm anchor point (default: [0,0,0])
          rot0 (rotation): initial rotation of arm from y-axis (default: identity)
        """

        self.links = links
        self.pos0 = pos0
        self.rot0 = rot0

        self.eval_positions()

    def eval_positions(self):
        """Evaluate the arm position coordinates"""
        pos, rot = self.pos0, self.rot0
        for link in self.links:
            pos, rot = link.get_position(pos, rot)

    def print_positions(self):
        for iseg, lnkcoord in enumerate(self.links):
            print(iseg, lnkcoord.pos)

    def get_positions(self):
        x, y, z = [self.pos0[0]], [self.pos0[1]], [self.pos0[2]]
        for link in self.links:
            x.append(link.pos[0])
            y.append(link.pos[1])
            z.append(link.pos[2])
        return x, y, z


if __name__ == "__main__":

    init_q1 = 90.0
    init_q2 = 0.0
    links = [
        Link(mass=1.0, length=10.0, alt_th=init_q1, sw_phi=0.0),
        Link(mass=1.0, length=5.0, alt_th=25.0, sw_phi=init_q2),
        Link(mass=1.0, length=5.0, alt_th=-135.0, sw_phi=0.0),
    ]

    pos = np.array([0, 0, 0])
    rot = R.from_matrix(np.identity(3))
    for link in links:
        pos, rot = link.get_position(pos, rot)

    arm = RobotArm3D(links=links)  # , x0=4.0, angle0=200.0)

    arm.print_positions()

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    # fig, ax = plt.subplots()
    # (line,) = plt.plot(t, f(t, init_amplitude, init_frequency), lw=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))
    ax1.set_xlim([-30, 30])
    ax1.set_ylim([-30, 30])
    ax2.set_xlim([-30, 30])
    ax2.set_ylim([-30, 30])

    # arm.plot_positions(color="b")
    x, y, z = arm.get_positions()
    (line1,) = ax1.plot(x, y, color="b")
    (line2,) = ax2.plot(z, y, color="b")

    ax1.set_aspect("equal", adjustable="box")
    ax2.set_aspect("equal", adjustable="box")

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax2.set_xlabel("z")
    ax2.set_ylabel("y")

    axcolor = "lightgoldenrodyellow"
    ax1.margins(x=0)

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
        line1.set_xdata(x)
        line1.set_ydata(y)
        line2.set_xdata(z)
        line2.set_ydata(y)
        fig.canvas.draw_idle()

    # register the update function with slider
    q1_slider.on_changed(update)
    q2_slider.on_changed(update)

    plt.show()
