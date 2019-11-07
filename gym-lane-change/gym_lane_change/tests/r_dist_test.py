import numpy as np
import matplotlib.pyplot as plt

import IPython

import sys
import unittest

sys.path.append("C:/Users/jdlow/Box/Fall 2019/ROB 590 RL for Driving/collision_environment/gym-lane-change/gym_lane_change/envs/")
sys.path.append("C:/Users/jdlow/Box/Fall 2019/ROB 590 RL for Driving/collision_environment/gym-lane-change/gym_lane_change/tests/")

from lane_change_env import *


def r_dist_fun(current_dist, max_dist):
    return max(0.0, 0.05 / (0.1 + 0.1*current_dist/max_dist))

def test_rewards():
    env = LaneChangeEnv()
    done = False
    obs = env.reset()

    rewards = []
    x = []
    y = []
    while not done:
        # env.render()
        obs, r, done, _ = env.step(np.array([np.radians(-30.0),
                                              np.radians(0.0)]))
        rewards.append(r)
        x.append(env.road.vehicle.state.x)
        y.append(env.road.vehicle.state.y)

    plt.ioff()
    plt.plot(rewards)
    plt.show()
    plt.close()

    plt.scatter(x, y)
    plt.scatter(env.road.goal.x, env.road.goal.y, color="green")
    plt.show()

def plot_rewards_map():
    env = LaneChangeEnv()
    data = []
    for y in range(-5, 10):
        for x in range(0, 120):
            env.road.vehicle.state = State(x, y, 0, 0, 0, 0, 0, 0)
            data.append([x, y, env.get_reward()])
            env.reset()

    plot_x = [d[0] for d in data]
    plot_y = [d[1] for d in data]
    plot_r = [d[2] for d in data]

    plt.ioff()
    im_data = plt.scatter(plot_x, plot_y, s=8, c=plot_r,
                cmap="viridis", vmin=-1.0, vmax=1.0)
    plt.colorbar(im_data)
    plt.title("Rewards values at XY locations")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    plt.close()


def test_r_dist_plot():
    max_dist = 100.0

    dist = np.flip(np.linspace(0, 200, 1000))
    r_dist = []
    for current_dist in dist:
        r_dist.append(r_dist_fun(current_dist, max_dist))

    plt.plot(dist, r_dist)
    plt.xlabel("Dist Vehicle - Goal")
    plt.ylabel("Reward")
    plt.show()


if __name__ == "__main__":
    # test_rewards()
    plot_rewards_map()

