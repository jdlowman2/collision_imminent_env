import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

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


def demonstrate_action(test_action="random", sparse=False):
    env = LaneChangeEnv(sparse)
    s = env.reset()

    if not args.random:
        env.road.obstacle.x = 60
        env.road.obstacle.y = 0.0
        env.road.obstacle.width = 0.5*LANE_WIDTH
        env.road.obstacle.length = 2.0*LANE_WIDTH

        env.road.goal.x = 100.0
        env.road.goal.y = 0.0
        env.road.goal.width = 0.5*LANE_WIDTH
        env.road.goal.length = 5.0*LANE_WIDTH

        env.road.vehicle.state = State(0.0, 0.0, 0, 15.0, 0, 0, 0, 0)

    done = False
    total_reward = 0.0
    tsteps = 0
    # IPython.embed()

    while not done:
        tsteps += 1
        env.render()
        if type(test_action)==str and test_action.lower() == "random":
            a = env.action_space.sample()
        else:
            a = test_action.copy()

        s, r, done, info = env.step(a) # random action
        total_reward += r
        print("Total reward: ", total_reward)
        print("Test Action: ", test_action)
    print("Total timesteps: ", tsteps)
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    np.set_printoptions(precision=2, suppress=True)
    # show_setup()
    # test_vehicle_rotation()

    sparse = args.sparse
    env = LaneChangeEnv()
    low, high = env.action_space.low, env.action_space.high

    print(f"Demonstrating with sparse? {sparse}")

    print("Demonstrating random action...")
    time.sleep(0.9)
    demonstrate_action("random", sparse)

    print("Demonstrating zero turning...")
    time.sleep(0.9)
    demonstrate_action(0*low, sparse)

    print("Demonstrating low action...", env.action_space.low)
    time.sleep(0.9)
    demonstrate_action(low, sparse)

    print("Demonstrating high action...")
    time.sleep(0.9)
    demonstrate_action(high, sparse=sparse)

    print("Demonstrating front wheel only action...")
    time.sleep(0.9)
    demonstrate_action(np.array([high[0], 0.0]), sparse=sparse)

    print("Demonstrating front wheel only action...")
    time.sleep(0.9)
    demonstrate_action(np.array([low[0], 0.0]), sparse=sparse)
