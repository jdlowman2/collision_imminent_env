import sys
import unittest

sys.path.append("C:/Users/jdlow/Box/Fall 2019/ROB 590 RL for Driving/collision_environment/gym-lane-change/gym_lane_change/envs/")
sys.path.append("C:/Users/jdlow/Box/Fall 2019/ROB 590 RL for Driving/collision_environment/gym-lane-change/gym_lane_change/tests/")

from lane_change_env import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import time
import IPython

env = LaneChangeEnv(num_obstacles=4)
env.reset()
env.render()

for i in range(50):
    env.render()
    env.step(np.array([np.radians(-3.0),
                          np.radians(1.0)]))


# env = LaneChangeEnv()

# env.reset()
# env.render()
# plt.pause(0.1)
# env.render()
# plt.pause(0.1)

# obs = env.reset()
# done = False

# it = 0
# while not done:
#     env.render()
#     obs, r, done, _ = env.step(np.array([np.radians(3.0),
#                                           np.radians(1.0)]))
#     print("\n")
#     print(env.road.vehicle)
#     # IPython.embed()
#     it +=1
#     # if it > 10:
#     #     break

# print("Finished!")

# # IPython.embed()
# time.sleep(3)
