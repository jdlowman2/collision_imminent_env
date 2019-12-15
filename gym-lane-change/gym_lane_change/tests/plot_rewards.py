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

def test_rewards(sparse):
    env = LaneChangeEnv(sparse)
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


def make_obstacles_deterministic(obstacles):
    new_obs = []
    for i, obstacle in enumerate(obstacles):
        obstacle.x = 20*(i+1)
        obstacle.y = 0.0
        obstacle.width = 0.5*LANE_WIDTH
        obstacle.length = 2.0*LANE_WIDTH

        new_obs.append(obstacle)

    return new_obs

def plot_rewards_map(sparse):
    env = LaneChangeEnv(sparse, num_obstacles=4)
    data = []

    # min_x, max_x = -10.0, 120.0
    # min_y, max_y = -25.0, 40.0

    min_x, max_x = -10.0, 110.0
    min_y, max_y = -10.0, 20.0

    for y in np.linspace(min_y, max_y, 100):
        for x in np.linspace(min_x, max_x, 100):
            # env.road.obstacles = make_obstacles_deterministic(env.road.obstacles)

            env.road.goal.x = 100.0
            env.road.goal.y = 0.0
            env.road.goal.width = 0.5*LANE_WIDTH
            env.road.goal.length = 5.0*LANE_WIDTH

            env.road.vehicle.state = State(x, y, 0, 0, 0, 0, 0, 0)
            data.append([x, y, env.get_reward()])
            env.reset()

    plot_x = [d[0] for d in data]
    plot_y = [d[1] for d in data]
    plot_r = [d[2] for d in data]

    plt.ioff()
    im_data = plt.scatter(plot_x, plot_y, s=2, c=plot_r,
                cmap="RdYlGn")#, vmin=-1.6, vmax=1.6)
    plt.colorbar(im_data)

    plt.scatter([0.0], [0.0], c="blue", s=150, marker="s", alpha=0.5)

    plt.title("Rewards values at XY locations")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    # plt.show()
    plt.savefig(sparse*"sparse" + "reward_map"+ ".png", dpi=200)
    plt.close()

# def plot_rewards_contour():
    # env = LaneChangeEnv()

    # x = np.arange(-5, 10)
    # y = np.arange(0, 120)

    # Z = np.zeros((y.shape[0], x.shape[0]))


    # def f(x):
    #     env.road.vehicle.state = State(x_val, y_val, 0, 0, 0, 0, 0, 0)
    #         Z[y_ind, x_ind] = env.get_reward()
    #         env.reset()

    #     return (x[:,0]**2 + x[:,1]**2)

    # xx, yy = np.meshgrid(x, y)
    # X_grid = np.c_[ np.ravel(xx), np.ravel(yy) ]
    # z = f(X_grid)

    # z = z.reshape(xx.shape)

    # plt.contour(xx, yy, z)



    # for y_ind, y_val in enumerate(y):
    #     for x_ind, x_val in enumerate(x):
    #         env.road.vehicle.state = State(x_val, y_val, 0, 0, 0, 0, 0, 0)
    #         Z[y_ind, x_ind] = env.get_reward()
    #         env.reset()

    # IPython.embed()
    # plt.ioff()
    # im_data = plt.contourf(x, y, Z, cmap="RdYlGn")
    # plt.colorbar(im_data)
    # plt.title("Rewards values at XY locations")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()
    # plt.savefig("reward_contour" + ".png", dpi=200)
    # plt.close()


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
    plot_rewards_map(True)
    plot_rewards_map(False)
    # plot_rewards_contour()

