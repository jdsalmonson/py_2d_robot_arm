import numpy as np
from two_link_arm import TwoLink2DDynArm, step_func

from numpy import heaviside
from scipy.integrate import solve_ivp


def torque1(t: float) -> float:
    return step_func(t, 1.0, 8.0)


def torque2(t: float) -> float:
    return step_func(t, 2.5, 3.0, -1.0)


tla = TwoLink2DDynArm(T1=torque1, T2=torque2, k1=10.0, k2=10.0)

# initial coordinates and velocities:
q1_i = 0.0
q1p_i = 0.0
q2_i = 0.0
q2p_i = 0.0
y_init = [q1_i, q1p_i, q2_i, q2p_i]

n_stp, dt = 10, 0.01  # 1/240.
t_i, t_f = 1.0, 1.0 + n_stp * dt

"""
# dense_output example
sol_tla = solve_ivp(tla.ODE_RHS, [ti, tf], y_init, dense_output=True)
t = np.linspace(ti, tf, 300)
z = sol_tla.sol(t)

import matplotlib.pyplot as plt
plt.plot(t, z.T, label=["q1","q1p","q2","q2p"])
plt.legend()
plt.show()
"""

sol_tla = solve_ivp(tla.ODE_RHS, [t_i, t_f], y_init)

import matplotlib.pyplot as plt

# Single integration:
plt.plot(sol_tla.t, sol_tla.y.T, label=["q1", "q1p", "q2", "q2p"])

# Sequential integrations:
y_i = y_init
for istp in range(n_stp):
    sol_tla = solve_ivp(tla.ODE_RHS, [t_i, t_i + dt], y_i)
    plt.plot(sol_tla.t, sol_tla.y.T, "--")
    t_i += dt
    y_i = sol_tla.y.T[-1, :]

plt.legend()
plt.show()

# solve_ivp(tla.ODE_RHS, [tf, tf+1./240.], sol_tla.y[:,-1]).y[:,-1]
