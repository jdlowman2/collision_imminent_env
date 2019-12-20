import numpy as np
from vehicle_model import Vehicle, State
from rectangle import *
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

LANE_WIDTH = 3.7 #m
VEHICLE_START_STATE = State(0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0)


config = {
            "vehicle_x": [0.0, 15.0],
            "vehicle_y": [0.0, 5.0],
            "vehicle_speed": [20.0, 30.0],
            "vehicle_h": [-np.radians(5.0), np.radians(5.0)],
            "vehicle_f_wheel": [-np.radians(5.0), np.radians(5.0)],
            "vehicle_r_wheel": [-np.radians(1.0), np.radians(1.0)],
            "obs_x": [20.0, 120.0],
            "obs_y": [-20.0, 20.0],
            "obs_w": 1.85,
            "obs_l": 7.4,
            }


class Lane(Rectangle):
    pass

class Obstacle(Rectangle):
    pass


class Road:
    def __init__(self, num_obstacles=1):
        self.obs_num = num_obstacles
        assert(num_obstacles <= 4)

        self.reset()

    def reset(self):
        start_x = np.random.uniform(low=config["vehicle_x"][0], high=config["vehicle_x"][1])
        start_y = np.random.uniform(low=config["vehicle_y"][0], high=config["vehicle_y"][1])

        speed = np.random.uniform(low=config["vehicle_speed"][0], high=config["vehicle_speed"][1])
        start_h = np.random.uniform(low=config["vehicle_h"][0], high=config["vehicle_h"][1])

        front_wheel_angle = np.random.uniform(low=config["vehicle_f_wheel"][0], high=config["vehicle_f_wheel"][1])
        rear_wheel_angle = np.random.uniform(low=config["vehicle_r_wheel"][0], high=config["vehicle_r_wheel"][1])

        start_state = State(start_x, start_y, start_h, speed, start_h, start_h, front_wheel_angle, rear_wheel_angle)
        self.vehicle = Vehicle(start_state)

        self.current_lane = Lane(length=1000, width=LANE_WIDTH,
                                    x=-100, y=0)
        self.opposing_lane = Lane(length=1000, width=LANE_WIDTH,
                                    x=-100, y=LANE_WIDTH)
        self.obstacles = []

        obs_w =  config["obs_w"]
        obs_l =  config["obs_l"]

        for obs_num in range(self.obs_num):
            obs_x = np.random.uniform(low=config["obs_x"][0], high=config["obs_x"][1])
            obs_y = np.random.uniform(low=config["obs_y"][0], high=config["obs_y"][1])

            self.obstacles.append(Obstacle(length=obs_l, width=obs_w,
                                        x=obs_x, y=obs_y))

        self.goal = Rectangle(length=5*LANE_WIDTH, width=0.5*LANE_WIDTH,
                                    x=100.0, y=0.0)


    def is_vehicle_in_road(self):
        # allows part of the vehicle outside lane as long as center of vehicle
        # is inside
        return self.current_lane.is_inside(self.vehicle.state.x, self.vehicle.state.y) or \
            self.opposing_lane.is_inside(self.vehicle.state.x, self.vehicle.state.y)

    def is_vehicle_in_collision(self):
        for obstacle in self.obstacles:
            if obstacle.intersects(self.vehicle.get_rectangle()):
                return True

        return False

    def is_vehicle_in_goal(self):
        return self.goal.intersects(self.vehicle.get_rectangle())

    def get_observation(self):
        obs_coords = [] # make obstacle states as [obs1.x, obs1.y, obs2.x, obs2.y, ...]
        for obj in self.obstacles:
            obs_coords.append(obj.x)
            obs_coords.append(obj.y)

        obs = np.array([
                    *self.vehicle.state, # unpack 8 state parameters
                    *obs_coords,
                ])
        return obs
