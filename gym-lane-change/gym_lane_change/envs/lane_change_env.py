import os, time
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import numpy as np

from road_model import *
from vehicle_model import *
from viewer import *
import IPython

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

MAX_ENV_STEPS = 200

class LaneChangeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, sparse_reward=False, num_obstacles=1):
        self._max_episode_steps = MAX_ENV_STEPS
        self.spec = gym.envs.registration.EnvSpec("LaneChangeEnv-v0")
        self.max_steps = MAX_ENV_STEPS
        self.steps_taken = 0

        self.road = Road(num_obstacles)
        self.viewer = None

        self.observation_space = spaces.Box(-np.inf, np.inf,
                                shape=self.road.get_observation().shape,
                                dtype=np.float32)

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.valid_region = {"min_x": -10.0, "max_x": 110.0,
                                "min_y": -10.0, "max_y": 20.0}

        self.sparse_reward=sparse_reward
        self.num_obstacles=num_obstacles
        self.penalty = 0.1

    def scale_action_to_model(self, action):
        action[0] = (action[0] + 1)/2.0 * \
                    (self.road.vehicle.max_f_rate - self.road.vehicle.min_f_rate) +\
                        self.road.vehicle.min_f_rate
        action[1] = (action[1] + 1)/2.0 * \
                    (self.road.vehicle.max_r_rate - self.road.vehicle.min_r_rate) +\
                        self.road.vehicle.min_r_rate

        return action

    def step(self, action):
        self.steps_taken += 1

        # scale actions
        scaled_action = self.scale_action_to_model(action)

        self.road.vehicle.step(scaled_action)

        if self.viewer is not None:
            self.viewer.last_reward = self.get_reward()

        return [self.get_observation(),
                self.get_reward(),
                self.is_done(),
                None]

    def reset(self):
        self.steps_taken = 0
        self.road.reset()
        if self.viewer is not None:
            self.viewer.reset()
        while self.is_vehicle_outside_valid_region() or self.road.is_vehicle_in_collision():
            self.road.reset()

        return self.get_observation()

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = Viewer(self.road)
        self.viewer.update_data(self.road)
        self.viewer.show()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def get_observation(self):
        return self.road.get_observation()

    def get_reward(self):
        if self.sparse_reward:
            return (10.0 + self.penalty) * self.road.is_vehicle_in_goal() - \
                        self.penalty - 10.0*self.road.is_vehicle_in_collision()

        r_dist = self.get_r_dist()**2
        r_vehicle_on_road = -1.0 * (not self.road.is_vehicle_in_road())

        r_vehicle_in_collision = -1.5 * self.road.is_vehicle_in_collision()
        r_vehicle_at_goal = +2.0 * self.road.is_vehicle_in_goal()

        return -self.penalty + r_vehicle_on_road + \
                r_vehicle_in_collision + \
                r_vehicle_at_goal +\
                r_dist +\
                -0.1*self.is_vehicle_outside_valid_region()

    def get_r_dist(self):
        vehicle_start = np.array([VEHICLE_START_STATE.x/10.0, VEHICLE_START_STATE.y])
        goal_pos = np.array([self.road.goal.x/10.0, self.road.goal.y])
        vehicle_curr = np.array([self.road.vehicle.state.x/10.0, self.road.vehicle.state.y])

        max_dist = np.linalg.norm(vehicle_start - goal_pos)
        current_dist = np.linalg.norm(vehicle_curr - goal_pos)

        return max(0.0, 0.05 / (0.1 + 1.0*current_dist/(max_dist)))


    def is_vehicle_outside_valid_region(self):
        in_valid_region = (self.valid_region["min_x"] <= self.road.vehicle.state.x <= self.valid_region["max_x"] and \
                        self.valid_region["min_y"] <= self.road.vehicle.state.y <= self.valid_region["max_y"])

        return not in_valid_region

    def is_done(self):
        is_done = False

        if self.steps_taken >= self.max_steps or\
            self.road.is_vehicle_in_collision():
            is_done = True

        if self.is_vehicle_outside_valid_region():
            # print("Vehicle outside of viewing area")
            is_done = True

        if self.viewer is not None:
            self.viewer.is_done = is_done

        return is_done
