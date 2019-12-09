import sys
import unittest

sys.path.append("C:/Users/jdlow/Box/Fall 2019/ROB 590 RL for Driving/collision_environment/gym-lane-change/gym_lane_change/envs/")
sys.path.append("C:/Users/jdlow/Box/Fall 2019/ROB 590 RL for Driving/collision_environment/gym-lane-change/gym_lane_change/tests/")

from vehicle_model import *
import numpy as np
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
        vehicle = make_dummy_vehicle(0.0, 0.0, 35.0)
        for i in range(10):
            vehicle.step([0.0, 0.0])
            # print(vehicle.state)
            self.assertTrue(vehicle.state.psi == 0.0)
            self.assertTrue(vehicle.state.u == 35.0)
            self.assertTrue(vehicle.state.delta_f == 0.0)
            self.assertTrue(vehicle.state.delta_r == 0.0)

    def test_saturate_steering(self):
        f, r = saturate_steering([MAX_F_STEERING, MAX_R_STEERING],
                                            MIN_F_STEERING, MAX_F_STEERING,
                                            MIN_R_STEERING, MAX_R_STEERING)
        assert(f == MAX_F_STEERING and r == MAX_R_STEERING)

        f, r = saturate_steering([0.5*MAX_F_STEERING, 0.5*MAX_R_STEERING],
                                            MIN_F_STEERING, MAX_F_STEERING,
                                            MIN_R_STEERING, MAX_R_STEERING)
        assert(f == 0.5*MAX_F_STEERING and r == 0.5*MAX_R_STEERING)

        f, r = saturate_steering([-1.1*MAX_F_STEERING, -1.1*MAX_R_STEERING],
                                            MIN_F_STEERING, MAX_F_STEERING,
                                            MIN_R_STEERING, MAX_R_STEERING)
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

            if state_var_ind == 6:
                new_val = max(MIN_F_STEERING, min(MAX_F_STEERING, new_val))
            if state_var_ind == 7:
                new_val = max(MIN_R_STEERING, min(MAX_R_STEERING, new_val))

            diff.append(abs(vehicle.state[state_var_ind] - new_val))
            self.assertTrue(diff[-1] <= 1E-4)


    def test_vehicle_rectangle(self):
        for test_num in range(10):
            vehicle = make_dummy_vehicle(*np.random.random(3))
            self.assertTrue(vehicle.get_rectangle().x == vehicle.state.x)
            self.assertTrue(vehicle.get_rectangle().y == vehicle.state.y)


def plot_vehicle(step):
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    colors = ["red", "blue"]

    for ind, sign in enumerate([-1.0, 1.0]):
        x,y = [], []
        vehicle = Vehicle(START_STATE)
        for i in range(10):
            vehicle.step(sign*step)
            x.append(vehicle.state.x)
            y.append(vehicle.state.y)
        ax.scatter(x, y, c=colors[ind], label="input:" + str(sign*step))
    ax.set_title("Vehicle XY Position")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.tight_layout()
    # plt.show()

def plot_vehicle_psi(step):
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    for ind, sign in enumerate([-1.0, 1.0]):
        y = []
        vehicle = Vehicle(START_STATE)
        for i in range(10):
            vehicle.step(sign*step)
            y.append(vehicle.state.psi)
        ax.plot(y, label="input:" + str(sign*step))

    ax.set_title("Vehicle psi")
    ax.set_xlabel("time")
    ax.set_ylabel("psi")
    ax.legend()
    plt.tight_layout()
    # plt.show()

def plot_vehicle_omega(step):
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    for ind, sign in enumerate([-1.0, 1.0]):
        y = []
        vehicle = Vehicle(START_STATE)
        for i in range(10):
            vehicle.step(sign*step)
            y.append(vehicle.state.omega)
        ax.plot(y, label="input:" + str(sign*step))

    ax.set_title("Vehicle Omega")
    ax.set_xlabel("time")
    ax.set_ylabel("omega")
    ax.legend()
    plt.tight_layout()
    # plt.show()

def plot_action(step):
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

    for ind, sign in enumerate([-1.0, 1.0]):
        x, x2 = [], []
        vehicle = Vehicle(START_STATE)
        for i in range(10):
            vehicle.step(sign*step)
            x.append(vehicle.action.delta_f_dot)
            x2.append(vehicle.action.delta_r_dot)
        ax[ind].plot(x, label="delta_f_dot")
        ax[ind].plot(x2, label="delta_r_dot")
        ax[ind].legend()
        ax[ind].set_title("Vehicle Inputs")
        ax[ind].set_xlabel("Time")
        ax[ind].set_ylabel("delta_dot")
    plt.tight_layout()
    # plt.show()

def plot_steering(step):
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

    for ind, sign in enumerate([-1.0, 1.0]):
        f, r = [], []
        vehicle = Vehicle(START_STATE)
        for i in range(10):
            vehicle.step(sign*step)
            f.append(vehicle.state.delta_f)
            r.append(vehicle.state.delta_r)
        ax[ind].plot(f, label="delta_f")
        ax[ind].plot(r, label="delta_r")
        ax[ind].legend()
        ax[ind].set_title("Vehicle Steering")
        ax[ind].set_xlabel("Time")
        ax[ind].set_ylabel("delta")
    plt.tight_layout()

def plot_lateral_forces(step):
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

    for ind, sign in enumerate([-1.0, 1.0]):
        x,y = [], []
        vehicle = Vehicle(START_STATE)
        for i in range(10):
            vehicle.step(sign*step)
            x.append(vehicle.F_yf)
            y.append(vehicle.F_yr)
        ax[ind].plot(x, label="F_yf")
        ax[ind].plot(y, label="F_yr")
        ax[ind].legend()
        ax[ind].set_title("Vehicle Fy")
        ax[ind].set_xlabel("Time")
        ax[ind].set_ylabel("Fy")
    plt.tight_layout()
    # plt.show()

def plot_sigma(step):
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

    for ind, sign in enumerate([-1.0, 1.0]):
        x,y = [], []
        vehicle = Vehicle(START_STATE)
        for i in range(10):
            vehicle.step(sign*step)
            x.append(vehicle.sigma_yf)
            y.append(vehicle.sigma_yr)
        ax[ind].plot(x, label="sigma_yf")
        ax[ind].plot(y, label="sigma_yr")
        ax[ind].legend()
        ax[ind].set_title("Vehicle sigma_y")
        ax[ind].set_xlabel("Time")
        ax[ind].set_ylabel("sigma_y")
    plt.tight_layout()
    # plt.show()

def plot_tests():
    step = np.array([np.radians(30).round(4), 0.0])
    plot_sigma(step)
    plot_action(step)
    plot_steering(step)
    plot_lateral_forces(step)
    plot_vehicle(step)
    plt.show()

if __name__ == '__main__':
    unittest.main()
