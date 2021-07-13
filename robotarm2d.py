import numpy as np
from dataclasses import dataclass

import pylab as plt


@dataclass
class Link:
    """A robot arm link

    Args:
      mass (float): arm link mass [gm]
      length (float): arm link length [cm]
      angle (float): local frame arm angle [degrees]
      motor_mass (float): mass of upper-link motor [gm] (default: 0)
    """

    mass: float
    length: float
    angle: float  # [degrees]
    motor_mass: float = 0.0

    def __post_init__(self):
        """The moment of inertia of a rod spinning about its mid-length (center-of-mass)"""
        self.momI = self.mass * self.length ** 2 / 12.0


@dataclass
class LinkCoords:
    """World coordinates of an arm link.  Positive angles rotate CCW from due South."""

    link: Link  # arm link
    w_angle: float = 0.0  # [degrees] world angle
    x: float = 0.0  # x-position of end effector of link
    y: float = 0.0  # y-position of end effector of link

    def eval_positions(self, angle0, x0, y0):
        """Given the angle and position (angle0, x0, y0) of the upper end of the
        arm link, evaluate the angle of this link and the (x, y) position of lower end.

        Args:
          angle0 (float): angle of parent link w.r.t. initial arm angle
          x0 (float): x-coordinate of upper joint of arm link
          y0 (float): y-coordinate of upper joint of arm link

        Returns:
          angle (float): angle of current link w.r.t. initial arm angle
          x (float): x-coordinate of lower joint of arm link
          y (float): y-coordinate of lower joint of arm link

        """
        global_angle = self.link.angle + angle0
        theta = np.radians(global_angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        # default link points along negative y-axis:
        local_lnk_coords = np.array([0, -self.link.length])
        global_lnk_coords = np.array([x0, y0]) + np.dot(R, local_lnk_coords)

        self.w_angle = global_angle
        self.x, self.y = global_lnk_coords

        return global_angle, *global_lnk_coords


class RobotArm2D(object):
    def __init__(self, links=[], angle0=0.0, x0=0.0, y0=0.0):
        """
        Args:
          links (list): list of links that make up the arm
          angle0 (float): reference angle of first arm link (default: 0.0, horizontal)
          x0 (float): x-coordinate of arm anchor point (default: 0.0)
          y0 (float): y-coordinate of arm anchor point (default: 0.0)
        """

        self.angle0 = angle0
        self.x0 = x0
        self.y0 = y0

        angle0, x0, y0 = self.angle0, self.x0, self.y0
        self.link_coords = []
        for link in links:
            lnkcoord = LinkCoords(link=link)
            angle0, x0, y0 = lnkcoord.eval_positions(angle0, x0, y0)
            self.link_coords.append(lnkcoord)

    def eval_positions(self):
        """Evaluate the arm position coordinates"""
        angle0, x0, y0 = self.angle0, self.x0, self.y0
        for lnkcoord in self.link_coords:
            angle0, x0, y0 = lnkcoord.eval_positions(angle0, x0, y0)

    def print_positions(self):
        for iseg, lnkcoord in enumerate(self.link_coords):
            print(iseg, lnkcoord.w_angle, lnkcoord.x, lnkcoord.y)

    def plot_positions(self, color="k"):
        x, y = [self.x0], [self.y0]
        for lnkcoord in self.link_coords:
            x.append(lnkcoord.x)
            y.append(lnkcoord.y)
        plt.plot(x, y, c=color)
        # plt.show()


if __name__ == "__main__":

    links = [
        Link(mass=1.0, length=10.0, angle=10.0),
        Link(mass=1.0, length=5.0, angle=15.0),
        Link(mass=1.0, length=5.0, angle=-135.0),
    ]

    arm = RobotArm2D(links=links)  # , x0=4.0, angle0=200.0)

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.xlim([-30, 30])
    plt.ylim([-30, 30])

    arm.print_positions()
    arm.plot_positions(color="b")

    arm.link_coords[0].link.angle -= 90
    arm.link_coords[1].link.angle = 90  # += 53
    arm.eval_positions()

    arm.print_positions()
    arm.plot_positions(color="r")

    ax.set_aspect("equal", adjustable="box")
    plt.show()