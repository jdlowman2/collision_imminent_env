import numpy as np
from numpy import sin, cos, arctan
from collections import namedtuple
import time
import matplotlib.pyplot as plt

import IPython

def add_states(s1, s2):
    new = [s1[ind] + s2[ind] for ind in range(len(s1))]
    return State(*new)

State = namedtuple("State", \
                    ["x", "y", "psi", \
                    "u", "v", "omega", \
                    "delta_f", "delta_r"])

Input = namedtuple("Input", ["delta_f_dot", "delta_r_dot"])

class Vehicle:
    def __init__(self, state):
        # Vehicle Parameters
        self.mass   =  2041.0        # mass (kg)
        self.I_zz = 4964.0 # Yaw moment of inertia
        self.L_f = 1.56 # front wheel to CG distance
        self.L_r = 1.64 # rear wheel to CG distance

        # Tire Properties
        self.mu = 0.8 # friction
        self.B=13.0 # empirical tire property
        self.C=1.285 # empirical tire property

        self.state = state
        self.delta_t = 0.1

        self.F_zf = 1016
        self.F_zr = 966

    def step(self, action):
        dx = self.dx(action)

        # self.state = self.state + dx
        self.state = add_states(self.state, dx)


    def dx(self, action_input):
        def F_y(delta, L, Fz):
            s_y = sigma_y(delta, L)
            return self.mu * Fz * s_y

        def sigma_y(delta, L):
            Vx = V_x(delta, L)
            Vy = V_y(delta, L)

            return -Vy / Vx * sin(self.C * arctan(self.B * Vy / Vx))

        def V_x(delta, L):
            return self.state.u*cos(delta) + \
                        (self.state.v + self.state.omega*L) \
                        * sin(delta)

        def V_y(delta, L):
            return -self.state.u*sin(delta) + \
                        (self.state.v + self.state.omega*L) \
                        * cos(delta)

        action = Input(action_input[0], action_input[1])

        dx = self.state.u*cos(self.state.psi) -\
                self.state.v*sin(self.state.psi)

        dy = self.state.u*sin(self.state.psi) +\
                self.state.v*cos(self.state.psi)

        F_yf = F_y(self.state.delta_f, self.L_f, self.F_zf)
        F_yr = F_y(self.state.delta_r, self.L_r, self.F_zr)

        dv = -1 * self.state.u*self.state.omega + \
                    1.0/self.mass * ( F_yr*cos(self.state.delta_r) + \
                    F_yf * cos(self.state.delta_f) )

        domega = 1.0/self.I_zz * (-1 * self.L_r * F_yr * \
                    cos(self.state.delta_r) + \
                    self.L_f*F_yf * cos(self.state.delta_f))

        return State(dx, dy, self.state.omega, 0.0, dv, domega,
                            action.delta_f_dot, action.delta_r_dot)
