import numpy as np
from numpy import sin, cos, arctan
from collections import namedtuple
import time
import matplotlib.pyplot as plt
from rectangle import *

import IPython

MAX_F_STEERING = np.radians(35.0)
MIN_F_STEERING = -MAX_F_STEERING

MAX_R_STEERING = np.radians(10.0)
MIN_R_STEERING = -MAX_R_STEERING

## These parameters should change together
# each call to vehicle.step updates by this timestep
VEHICLE_TIMESTEP = 0.05
# number of euler integrations per call to vehicle.step
INTEGRATION_STEPS_PER_UPDATE = 2
##

MAX_F_STEERING_RATE = np.radians(70.0) #*VEHICLE_TIMESTEP # np.radians(70)*VEHICLE_TIMESTEP Paper says 70 degrees per second
MAX_R_STEERING_RATE = np.radians(35.0) # Paper says 35 degrees per second

MIN_F_STEERING_RATE = -MAX_F_STEERING_RATE
MIN_R_STEERING_RATE = -MAX_R_STEERING_RATE

def saturate_steering(delta, min_f, max_f, min_r, max_r):
    delta_f, delta_r = delta

    return max(min_f, min(max_f, delta_f)),\
            max(min_r, min(max_r, delta_r))

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

        self.F_zf = 0.514 * self.mass # mass distribution
        self.F_zr = 0.486 * self.mass # mass distribution
        assert(abs((self.F_zf + self.F_zr) - self.mass) < 1E-5)

        # Tire Properties
        self.mu = 0.8 # friction
        self.B  = 13.0 # empirical tire property
        self.C  = 1.285 # empirical tire property

        self.state = state
        self.d_state = State(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.delta_t = VEHICLE_TIMESTEP

        self.integration_steps = INTEGRATION_STEPS_PER_UPDATE

        self.max_f = MAX_F_STEERING
        self.max_r = MAX_R_STEERING
        self.min_f = MIN_F_STEERING
        self.min_r = MIN_R_STEERING

        self.max_f_rate = MAX_F_STEERING_RATE
        self.max_r_rate = MAX_R_STEERING_RATE

        self.min_f_rate = MIN_F_STEERING_RATE
        self.min_r_rate = MIN_R_STEERING_RATE

    def get_rectangle(self):
        return Rectangle(self.length, self.width, self.state.x, self.state.y)

    def step(self, action):
        for int_step in range(self.integration_steps):
            self.d_state = self.compute_dx(action)

            s1 = []
            for ind in range(len(self.state)):
                s1.append(self.state[ind] + \
                    self.delta_t / self.integration_steps * self.d_state[ind])

            s1[-2:] = saturate_steering(s1[-2:],
                                            self.min_f, self.max_f,
                                            self.min_r, self.max_r)

            self.state = State(*s1)

    def compute_dx(self, action_input):
        def F_y(delta, L, Fz):
            s_y = sigma_y(delta, L)
            return self.mu * Fz * s_y

        def sigma_y(delta, L):
            Vx = V_x(delta, L)
            Vy = V_y(delta, L)

            return sin(self.C * arctan(self.B * Vy / Vx))

        def V_x(delta, L):
            return self.state.u*cos(delta) + \
                        (self.state.v + self.state.omega*L) \
                        * sin(delta)

        def V_y(delta, L):
            return -self.state.u*sin(delta) + \
                        (self.state.v + self.state.omega*L) \
                        * cos(delta)

        action = Input(action_input[0], action_input[1])
        self.action = action
        assert(action.delta_f_dot == action_input[0])
        assert(action.delta_r_dot == action_input[1])

        self.vxf = V_x(self.state.delta_f, self.L_f)
        self.vyf = V_y(self.state.delta_f, self.L_f)
        self.sigma_yf = sigma_y(self.state.delta_f, self.L_f)
        self.F_yf = F_y(self.state.delta_f, self.L_f, self.F_zf)
        F_yf = self.F_yf

        self.vxr = V_x(self.state.delta_r, self.L_r)
        self.vyr = V_y(self.state.delta_r, self.L_r)
        self.sigma_yr = sigma_y(self.state.delta_r, self.L_r)
        self.F_yr = F_y(self.state.delta_r, self.L_r, self.F_zr)
        F_yr = self.F_yr

        dx = self.state.u*cos(self.state.psi) -\
                self.state.v*sin(self.state.psi)

        dy = self.state.u*sin(self.state.psi) +\
                self.state.v*cos(self.state.psi)

        dpsi = self.state.omega
        du = 0.0

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
