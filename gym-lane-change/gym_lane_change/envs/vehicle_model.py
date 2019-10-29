import numpy as np
from numpy import sin, cos, arctan
from collections import namedtuple
import time
import matplotlib.pyplot as plt

import IPython

MIN_STEERING = np.radians(-35.0)
MAX_STEERING = np.radians(35.0)

MIN_F_STEERING = np.radians(-35.0)
MAX_F_STEERING = np.radians(35.0)

MIN_R_STEERING = np.radians(-10.0)
MAX_R_STEERING = np.radians(10.0)

## These parameters should change together
VEHICLE_TIMESTEP = 0.1
INTEGRATION_STEPS_PER_UPDATE = 1
##

def saturate_steering(delta):
    delta_f, delta_r = delta

    return max(MIN_F_STEERING, min(MAX_F_STEERING, delta_f)),\
            max(MIN_R_STEERING, min(MAX_R_STEERING, delta_r))

State = namedtuple("State", \
                    ["x", "y", "psi", \
                    "u", "v", "omega", \
                    "delta_f", "delta_r"])

Input = namedtuple("Input", ["delta_f_dot", "delta_r_dot"])

class Vehicle:
    def __init__(self, state):
        # Vehicle Parameters
        self.mass   =  2041.0        # mass (kg)
        self.width = 1.8 # width [m]
        self.length = 4.8 # length [m] TODO: check this
        self.I_zz = 4964.0 # Yaw moment of inertia
        self.L_f = 1.56 # front wheel to CG distance
        self.L_r = 1.64 # rear wheel to CG distance

        # Tire Properties
        self.mu = 0.8 # friction
        self.B  = 13.0 # empirical tire property
        self.C  = 1.285 # empirical tire property

        self.state = state
        self.d_state = State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.delta_t = VEHICLE_TIMESTEP

        self.F_zf = 0.514 * self.mass
        self.F_zr = 0.486 * self.mass

        self.integration_steps = INTEGRATION_STEPS_PER_UPDATE

        assert(abs((self.F_zf + self.F_zr) - self.mass) < 1E-5)

    def step(self, action):
        for i in range(self.integration_steps):
            self.d_state = self.compute_dx(action)

            s1 = []
            for i in range(len(self.state)):
                s1.append(self.state[i] + self.delta_t * self.d_state[i])

            s1[-2:] = saturate_steering(s1[-2:])

            self.state = State(*s1)

    def compute_dx(self, action_input):
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

        assert(action.delta_f_dot == action_input[0])
        assert(action.delta_r_dot == action_input[1])

        dx = self.state.u*cos(self.state.psi) -\
                self.state.v*sin(self.state.psi)

        dy = self.state.u*sin(self.state.psi) +\
                self.state.v*cos(self.state.psi)

        dpsi = self.state.omega
        du = 0.0

        F_yf = F_y(self.state.delta_f, self.L_f, self.F_zf)
        F_yr = F_y(self.state.delta_r, self.L_r, self.F_zr)

        dv = -1.0 * self.state.u*self.state.omega + \
                    1.0/self.mass * ( F_yr*cos(self.state.delta_r) + \
                    F_yf * cos(self.state.delta_f) )

        domega = 1.0/self.I_zz * (-1.0 * self.L_r * F_yr * \
                    cos(self.state.delta_r) + \
                    self.L_f*F_yf * cos(self.state.delta_f))

        return State(dx, dy, dpsi, du, dv, domega,
                            action.delta_f_dot, action.delta_r_dot)


    def __repr__(self):
        s = State(*[round(i, 3) for i in [*self.state]])
        ds = State(*[round(i, 3) for i in [*self.d_state]])
        return "Vehicle State: " + str(s) + \
                        "\n" + "D_State: " + str(ds)
