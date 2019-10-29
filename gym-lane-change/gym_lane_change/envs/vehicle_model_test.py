from vehicle_model import *
import numpy as np
import unittest

import matplotlib.pyplot as plt

import IPython


START_STATE = State(0.0, 0.0, 0.0, 35.0, 0.0, 0.0, 0.0, 0.0)

def make_dummy_vehicle(x, y, v):
    state = State(x, y, 0.0, v, 0.0, 0.0, 0.0, 0.0)
    return Vehicle(state)

def compute_dx(si, vehicle, dt, test_input):
    vx_f = si.u * np.cos(si.delta_f) + \
            (si.v + si.omega * vehicle.L_f)*np.sin(si.delta_f)
    vy_f = - si.u * np.sin(si.delta_f) + \
                (si.v + si.omega * vehicle.L_f)*np.cos(si.delta_f)

    sigma_f = - vy_f / vx_f * np.sin(vehicle.C * \
                                np.arctan(vy_f / vx_f * vehicle.B))
    F_yf = vehicle.mu * vehicle.F_zf * sigma_f

    vx_r = si.u * np.cos(si.delta_r) + \
            (si.v + si.omega * vehicle.L_r)*np.sin(si.delta_r)
    vy_r = - si.u * np.sin(si.delta_r) + \
                (si.v + si.omega * vehicle.L_r)*np.cos(si.delta_r)
    sigma_r = - vy_r / vx_r * np.sin(vehicle.C * \
                                np.arctan(vy_r / vx_r * vehicle.B))
    F_yr = vehicle.mu * vehicle.F_zr * sigma_r


    dx = si.u * np.cos(si.psi) - si.v * np.sin(si.psi)
    dy = si.u * np.sin(si.psi) + si.v * np.cos(si.psi)
    dpsi = si.omega
    du = 0.0
    dv = -1.0*si.u * si.omega + (1/vehicle.mass) * \
                            (F_yf * np.cos(si.delta_f) + \
                                F_yr * np.cos(si.delta_r))

    domega = 1.0 / vehicle.I_zz *(F_yf * np.cos(si.delta_f) * vehicle.L_f - \
                F_yr * np.cos(si.delta_r) * vehicle.L_r)
    ddeltaf = test_input[0]
    ddeltar = test_input[1]

    return State(dx, dy, dpsi, du, dv, domega, ddeltaf, ddeltar)


class TestVehicle(unittest.TestCase):

    def test_no_input(self):
        vehicle = make_dummy_vehicle(0.0, 0.0, 0.0)
        for i in range(2):
            vehicle.step([0.0, 0.0])
            print("Vehicle state: ")
            print(vehicle.state)
            self.assertTrue(vehicle.state, State(0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0,
                                                 0.0, 0.0))

    def test_saturate_steering(self):
        f, r = saturate_steering([MAX_F_STEERING, MAX_R_STEERING])
        assert(f == MAX_F_STEERING and r == MAX_R_STEERING)

        f, r = saturate_steering([0.5*MAX_F_STEERING, 0.5*MAX_R_STEERING])
        assert(f == 0.5*MAX_F_STEERING and r == 0.5*MAX_R_STEERING)

        f, r = saturate_steering([-1.1*MAX_F_STEERING, -1.1*MAX_R_STEERING])
        assert(f == MIN_F_STEERING and r == MIN_R_STEERING)

    def test_delta_x(self):
        vehicle = Vehicle(START_STATE)
        vehicle.integration_steps = 1
        dt = vehicle.delta_t
        test_input = np.array([1.0, 2.0])

        delta_s = compute_dx(START_STATE, vehicle, dt, test_input)

        vehicle.step(test_input)

        d_diff = []
        for state_var_ind in range(len(delta_s)):
            d_diff.append(abs(vehicle.d_state[state_var_ind] - \
                                    delta_s[state_var_ind]))
            self.assertTrue(d_diff[-1] <= 1E-4)

    def test_single_step(self):
        vehicle = Vehicle(START_STATE)
        vehicle.integration_steps = 1

        dt = vehicle.delta_t
        test_input = np.array([1.0, 2.0])

        vehicle.step(test_input)

        diff = []
        for state_var_ind in range(len(START_STATE)):
            new_val = START_STATE[state_var_ind] + \
                            dt * vehicle.d_state[state_var_ind]
            diff.append(abs(vehicle.state[state_var_ind] - new_val))
            self.assertTrue(diff[-1] <= 1E-4)

if __name__ == '__main__':
    unittest.main()
