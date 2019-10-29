from lane_change_env import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import time
import IPython

road = Road()
viewer = Viewer(road)

for i in range(50):
    road.vehicle.step(np.array([np.radians(3.0),
                                  np.radians(1.0)]))
    viewer.update_data(road)
    viewer.show()


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
