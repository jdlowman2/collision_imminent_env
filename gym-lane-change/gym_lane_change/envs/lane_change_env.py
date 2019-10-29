import os, time
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import numpy as np

from road_model import *
from vehicle_model import *
from viewer import *

# After you have installed your package with pip install -e gym-foo,
# you can create an instance of the environment with
# gym.make('gym_foo:foo-v0')


"""
Initial environment setup
autonomous car
    initial_state = [x=0, y=lane_width/2.0, v=35m/s]
current_lane
    - left_boundary
    - right_boundary
opposing_lane
    - left_boundary
    - right_boundary
Center line is equivalent to
    - current_lane.left_boundary
    - opposing_lane.right_boundary
Obstacle
    - state=[x=55m, y=lane_witdth/2.0]
    - rectangle=[w=lane_witdth, l=2*lane_width]

Diagram:

x=0
|_______________________________________


---------------------------------------
car-->                         [obstacle]
_______________________________________
y=0
"""

MAX_ENV_STEPS = 100

class LaneChangeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Vehicle has 8 states. 8 road and obstacle states
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)

        # Action space omits the Tackle/Catch actions, which are useful on defense
        self.action_space = spaces.Box(MIN_STEERING, MAX_STEERING, (2,), dtype=np.float32)
        self.max_steps = MAX_ENV_STEPS

        self.road = Road()
        self.viewer = Viewer(self.road)

        self.steps_taken = 0

    def step(self, action):
        self.steps_taken += 1
        self.road.vehicle.step(action)

        return [self.get_observation(),
                self.get_reward(),
                self.is_done(),
                None]

    def reset(self):
        self.steps_taken = 0
        self.road.reset()

        return self.get_observation()

    def render(self, mode='human'):
        self.viewer.update_data(self.road)
        self.viewer.show()

    def close(self):
        pass

    def get_observation(self):
        obs = np.array([*self.road.vehicle.state, # unpack 8 state parameters
                self.road.obstacle.get_left_boundary(),
                self.road.obstacle.get_right_boundary(),
                self.road.obstacle.get_start(),
                self.road.obstacle.get_end(),
                self.road.current_lane.get_left_boundary(),
                self.road.current_lane.get_right_boundary(),
                self.road.opposing_lane.get_left_boundary(),
                self.road.opposing_lane.get_right_boundary(),
                ])

        return obs


    def get_reward(self):

        r_vehicle_on_road = -1 * self.road.is_vehicle_in_road()
        r_vehicle_in_collision = -1 * self.road.is_vehicle_in_collision()
        r_vehicle_at_goal = +1 * self.road.is_vehicle_in_goal()

        return r_vehicle_on_road + r_vehicle_in_collision + r_vehicle_at_goal

    def is_done(self):
        if self.steps_taken >= self.max_steps or\
            self.road.is_vehicle_in_collision() or\
                self.road.is_vehicle_in_goal():

            return True

        return False
