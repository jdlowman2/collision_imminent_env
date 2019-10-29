from vehicle_model import Vehicle, State

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

class Rectangle:
    def __init__(self, length, width, x, y):
        self.length = length
        self.width = width
        self.x = x
        self.y = y

    def is_inside(self, x, y):
        lower_x = self.x - self.length/2.0
        upper_x = self.x + self.length/2.0
        lower_y = self.y - self.width/2.0
        upper_y = self.y + self.width/2.0

        if x <= upper_x and x >= lower_x:
            if y <= upper_y and y >= lower_y:
                return True

        return False

    def get_left_boundary(self):
        return self.y + self.width/2.0

    def get_right_boundary(self):
        return self.y - self.width/2.0

    def get_start(self):
        return self.x - self.width/2.0

    def get_end(self):
        return self.x + self.width/2.0


class Lane(Rectangle):
    pass

class Obstacle(Rectangle):
    pass


class Road:
    def __init__(self):
        self.reset()

    def reset(self):
        self.vehicle = Vehicle(VEHICLE_START_STATE)
        self.current_lane = Lane(length=1000, width=LANE_WIDTH,
                                    x=0, y=0)
        self.opposing_lane = Lane(length=1000, width=LANE_WIDTH,
                                    x=0, y=LANE_WIDTH)
        self.obstacle = Obstacle(length=2.0*LANE_WIDTH, width=LANE_WIDTH,
                                    x=55.0, y=0.0)

        self.goal = Rectangle(length=5*LANE_WIDTH, width=LANE_WIDTH,
                                    x=100.0, y=0.0)

        # print("Road reset!")

    def is_vehicle_in_road(self):
        if self.current_lane.is_inside(self.vehicle.state.x, self.vehicle.state.y) or \
            self.opposing_lane.is_inside(self.vehicle.state.x, self.vehicle.state.y):

            return True

        return False

    def is_vehicle_in_collision(self):
        return self.obstacle.is_inside(self.vehicle.state.x, self.vehicle.state.y)

    def is_vehicle_in_goal(self):
        return self.goal.is_inside(self.vehicle.state.x, self.vehicle.state.y)

