import numpy as np


class TwoLink2DDynArm(object):
    """Parameters and dynamical equations of a two link arm comprised of two joints:
    a shoulder, q1, and an elbow, q2.  Link #1 is the upper arm and #2 is the lower
    arm.  If the moments of inertia are not set, then their respective links are
    calculated as bars: I = (M*L^2)/12.  If gravitational constant is not set, it
    is set to g = 9.8 [m/s^2].

    Args:
      T1 (callable): torque function of time of joint 1 (default: None)
      T2 (callable): torque function of time of joint 2 (default: None)
      M1 (float): upper link mass [kg]
      L1 (float): upper link length [m]
      k1 (float): coefficient of damping torque of shoulder joint, q1, (T = -k1*q1')
      M2 (float): lower link mass [kg]
      L2 (float): lower link length [m]
      k2 (float): coefficient of damping torque of elbow joint, q2, (T = -k2*q2')
      I1 (float): upper link moment of inertia (default: None)
      I2 (float): lower link moment of intertia (default: None)
      g (float): gravitational constant
    """

    def __init__(
        self,
        T1=None,
        T2=None,
        M1=1.0,
        L1=1.0,
        k1=0.1,
        M2=1.0,
        L2=1.0,
        k2=0.1,
        I1=None,
        I2=None,
        g=None,
    ):

        self.T1 = (lambda x: 2 * 3.0) if T1 is None else T1
        self.T2 = (lambda x: 2 * 0.75) if T2 is None else T2

        self.M1 = M1
        self.L1 = L1
        self.k1 = k1
        self.M2 = M2
        self.L2 = L2
        self.k2 = k2
        # Make links simple bars:
        self.I1 = I1 if I1 is not None else self.momI_bar(M1, L1)
        self.I2 = I2 if I2 is not None else self.momI_bar(M2, L2)
        self.g = g if g is not None else 9.8  # [9.8 m/s^2]

    @staticmethod
    def momI_bar(M, L):
        """Moment of Inertia of a bar spinning about its mid-length (center of mass: CM).

        Args:
          M (float): mass of bar
          L (float): length of bar
        """
        return (M * L ** 2) / 12.0

    def A1(self, t, q1):
        M1, M2, L1, g = self.M1, self.M2, self.L1, self.g
        return self.T1(t) - self.T2(t) - (M1 / 2.0 + M2) * g * L1 * np.sin(q1)

    def B1(self):
        return -self.k1

    def C1(self):
        return self.k1

    def E1(self, q1, q2):
        M2, L1, L2 = self.M2, self.L1, self.L2
        return -M2 * L1 * L2 / 2.0 * np.sin(q1 - q2)

    def F1(self):
        I1, L1, M1, M2 = self.I1, self.L1, self.M1, self.M2
        return -(I1 + (L1) ** 2 * (M1 / 4.0 + M2))

    def G1(self, q1, q2):
        M2, L1, L2 = self.M2, self.L1, self.L2
        return -M2 * L1 * L2 / 2.0 * np.cos(q1 - q2)

    def A2(self, t, q2):
        M2, L2, g = self.M2, self.L2, self.g
        return self.T2(t) - M2 * g * (L2 / 2.0) * np.sin(q2)

    def C2(self):
        return -self.k2

    def D2(self, q1, q2):
        M2, L1, L2 = self.M2, self.L1, self.L2
        return M2 * L1 * L2 / 2.0 * np.sin(q1 - q2)

    def G2(self):
        I2, M2, L2 = self.I2, self.M2, self.L2
        return -(I2 + M2 * (L2 / 2.0) ** 2)

    def F2(self, q1, q2):
        return self.G1(q1, q2)

    def K(self, q1, q2):
        return self.G2() * self.F1() - self.F2(q1, q2) * self.G1(q1, q2)

    def ODE1(self, t, q1p, q2p, q1, q2):
        """returns q1'', the 2nd time derivative of q1 (q1'' == q1pp)"""
        return (
            -1.0
            / self.K(q1, q2)
            * (
                (self.G2() * self.A1(t, q1) - self.A2(t, q2) * self.G1(q1, q2))
                + (self.G2() * self.B1() - 0.0) * q1p
                + (self.G2() * self.C1() - self.C2() * self.G1(q1, q2)) * q2p  # B2 = 0
                + (0.0 - self.D2(q1, q2) * self.G1(q1, q2)) * (q1p) ** 2  # D1 = 0
                + (self.G2() * self.E1(q1, q2) - 0.0) * (q2p) ** 2  # E2 = 0
            )
        )

    def ODE2(self, t, q1p, q2p, q1, q2):
        """returns q2'', the 2nd time derivative of q2 (q2'' == q2pp)"""
        return (
            1.0
            / self.K(q1, q2)
            * (
                (self.F2(q1, q2) * self.A1(t, q1) - self.A2(t, q2) * self.F1())
                + (self.F2(q1, q2) * self.B1() - 0.0) * q1p
                + (self.F2(q1, q2) * self.C1() - self.C2() * self.F1()) * q2p  # B2 = 0
                + (0.0 - self.D2(q1, q2) * self.F1()) * (q1p) ** 2  # D1 = 0
                + (self.F2(q1, q2) * self.E1(q1, q2) - 0.0) * (q2p) ** 2  # E2 = 0
            )
        )

    def ODE_RHS(self, t, y):
        """The right-hand-sides of the four ODEs required to solve the two link arm
        dynamical equations.

        The y array is defined as the array of variables, q, and their derivatives:
        y[0] = q1, y[1] = q1p, y[2] = q2, y[3] = q2p
        where p = prime denotes derivative in tim (i.e. q1p = q1')

        so the four 1st order ODE's to be solved are:
          y[0]' = q1p
          y[1]' = q1pp
          y[2]' = q1p
          y[3]' = q2pp
        """
        q1pp = self.ODE1(t, y[1], y[3], y[0], y[2])
        q2pp = self.ODE2(t, y[1], y[3], y[0], y[2])
        return [y[1], q1pp, y[3], q2pp]


def step_func(t: float, t0: float, t1: float, amp: float = 1.0) -> float:
    """Step function. Returns 0 if t < t0 or t > t1, otherwise returns 'amp'

    Args:
      t (float): time at which to evaluate step function
      t0 (float): start time of step
      t1 (float): end time of step
      amp (float): amplitude of step (default: 1.)
    """
    return amp * np.heaviside(t - t0, 1.0) * np.heaviside(t1 - t, 1.0)


if __name__ == "__main__":

    tla = TwoLink2DDynArm(k1=10.0, k2=10.0)

    from scipy.integrate import solve_ivp

    # initial coordinates and velocities:
    q1_i = 0.0
    q1p_i = 0.0
    q2_i = 0.0
    q2p_i = 0.0
    y_init = [q1_i, q1p_i, q2_i, q2p_i]

    ti, tf = 0.0, 10.0
    sol_tla = solve_ivp(tla.ODE_RHS, [ti, tf], y_init, dense_output=True)
    t = np.linspace(ti, tf, 300)
    z = sol_tla.sol(t)

    import matplotlib.pyplot as plt

    plt.plot(t, z.T, label=["q1", "q1p", "q2", "q2p"])
    plt.legend()
    plt.show()
