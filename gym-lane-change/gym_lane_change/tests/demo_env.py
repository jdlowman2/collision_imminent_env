import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import IPython
import time

import sys
import unittest

sys.path.append("C:/Users/jdlow/Box/Fall 2019/ROB 590 RL for Driving/collision_environment/gym-lane-change/gym_lane_change/envs/")
sys.path.append("C:/Users/jdlow/Box/Fall 2019/ROB 590 RL for Driving/collision_environment/gym-lane-change/gym_lane_change/tests/")

from lane_change_env import *

def show_setup():
    env = LaneChangeEnv()
    env.reset()

    for i in range(20):
        env.render()
        env.reset()
        time.sleep(0.9)
    env.close()

def demonstrate():
    env = LaneChangeEnv()
    s = env.reset()
    done = False

    for i in range(20):
        env.render()
        s, r, done, info = env.step(env.action_space.sample()) # random action
        print("Observation: \n", s)

    # IPython.embed()

    env.close()

def demonstrate_hard_coded_action(action):
    env = LaneChangeEnv()
    s = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        env.render()
        s, r, done, info = env.step(action) # random action
        total_reward += r
        print("Total reward: ", total_reward)
    env.close()

    return total_reward

def test_vehicle_rotation():
    plt.ion()
    fig, ax = plt.subplots(1, 1, num='LaneChangeEnv')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])

    for heading in [0, 45, 90]:#, 30, 45, 60, 90]:
        heading = np.radians(heading)
        x,y = 3.0+.01*np.degrees(heading), 5.0+.01*np.degrees(heading)
        w, l = 1.0, 4.0

        td2dis = ax.transData
        coords = td2dis.transform([x, y])
        tr = mpl.transforms.Affine2D().rotate_around(coords[0], coords[1], heading)
        t = td2dis + tr

        d = np.sqrt(.25*w**2 + .25*l**2)
        alpha = np.arctan(w/l)
        dx = d*cos(heading + alpha)
        dy = d*sin(heading + alpha)

        dx = .5*l
        dy = .5*w

        r = patches.Rectangle(
                            (x-dx, y-dy),
                            height=w,
                            width=l,
                            color="lightgreen",
                            transform=t)

        arrow = patches.Arrow(
                x=x,y=y,
                dx = 2*np.cos(heading),
                dy = 2*np.sin(heading),
                )

        ax.add_patch(r)
        ax.add_patch(arrow)

        plt.plot([x, x+5*np.cos(heading)], [y,y+5*np.sin(heading)], "red")

        plt.pause(0.1)
    input("Press <enter> to quit")

def test_vehicle_rectangle():
    plt.ion()
    fig, ax = plt.subplots(1, 1, num='LaneChangeEnv')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])

    state1 = State(3.0, 5.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0)
    state2 = State(3.0, 7.0, np.radians(30), 5.0, 0.0, 0.0, 0.0, 0.0)
    w = 1.0
    l = 4.0

    d = np.sqrt(.25*w**2 + .25*l**2)
    angle_d = np.arctan(w/l)

    td2dis = ax.transData
    coords = td2dis.transform([state1.x, state1.y])
    tr = mpl.transforms.Affine2D().rotate_around(coords[0], coords[1], state1.psi)
    t1 = td2dis + tr

    coords = td2dis.transform([state2.x, state2.y])
    tr = mpl.transforms.Affine2D().rotate_around(coords[0], coords[1], state2.psi)
    t2 = td2dis + tr

    r1 = patches.Rectangle(
                        (state1.x-.5*l, state1.y-.5*w),
                        height=w,
                        width=l,
                        angle = 0.0,
                        color="lightgreen",
                        transform=t1)
    r2 = patches.Rectangle(
                    (state2.x-.5*l, state2.y-.5*w),
                    height=w,
                    width=l,
                    angle = 0.0,
                    color="blue",
                    transform=t2)

    arrow1 = patches.Arrow(
            x=state1.x,
            y=state1.y,
            dx = 2*np.cos(state1.psi),
            dy = 2*np.sin(state1.psi),
            )
    arrow2 = patches.Arrow(
            x=state2.x,
            y=state2.y,
            dx = 2*np.cos(state2.psi),
            dy = 2*np.sin(state2.psi),
            )

    ax.add_patch(r1)
    ax.add_patch(r2)
    ax.add_patch(arrow1)
    ax.add_patch(arrow2)

    plt.plot([3, 3+5*np.cos(state2.psi)], [7,7+5*np.sin(state2.psi)], "red")

    plt.pause(0.1)

    ax.scatter(state1.x, state1.y, color="red", s=10)
    ax.scatter(state2.x, state2.y, color="yellow", s=10)

    plt.pause(0.1)

    input("Press <enter> to quit")
    plt.close()

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    # show_setup()
    test_vehicle_rotation()
    # demonstrate()

    env = LaneChangeEnv()
    print("Demonstrating zero turning...")
    demonstrate_hard_coded_action(0*env.action_space.low)

    print("Demonstrating low action...")
    demonstrate_hard_coded_action(env.action_space.low)

    print("Demonstrating high action...")
    demonstrate_hard_coded_action(env.action_space.high)
