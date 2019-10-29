import matplotlib.patches as patches
import matplotlib.pyplot as plt

from vehicle_model import *
from road_model import *

class Viewer:
    def __init__(self, road):
        self.road = road
        plt.ion() ## interactive to prevent blocking
        self.fig, self.ax = plt.subplots(1, 1, num='LaneChangeEnv')

    def update_data(self, road):
        self.road = road

    def show(self):
        self.clear_screen()
        self.plot_lanes()

        self.plot_rectangle(self.road.obstacle, "red")
        self.plot_rectangle(self.road.goal, "green")

        self.plot_vehicle()
        self.plot_steering_wheel()

        self.ax.set_xlim([-5, 100])
        self.ax.set_ylim([-10, 20])

        plt.show()
        plt.pause(0.1)

    def clear_screen(self):
        self.plot_rectangle(Rectangle(length=1000.0,
                                        width=3000.0,
                                        x=-5.0,
                                        y=-10.0),
                                        color="white")

    def plot_lanes(self):
        self.plot_rectangle(self.road.current_lane, (0.86, 0.86, 0.86))
        self.plot_rectangle(self.road.opposing_lane, (0.86, 0.86, 0.86))

        for lane in [self.road.current_lane, self.road.opposing_lane]:
            self.ax.axhline(y=lane.get_left_boundary(),
                                linestyle='-', color='black')
            self.ax.axhline(y=lane.get_right_boundary(),
                        linestyle='-', color='black')

        center_line = 0.5*(self.road.current_lane.get_left_boundary() + \
                        self.road.opposing_lane.get_right_boundary())
        self.ax.axhline(y=center_line,
                            linestyle='--', color='yellow')

    def plot_vehicle(self):
        heading_rad = self.road.vehicle.state.psi
        l = self.road.vehicle.length
        w = self.road.vehicle.width
        self.plot_rectangle(Rectangle(length=l,
                                        width=w,
                                        x=self.road.vehicle.state.x,
                                        y=self.road.vehicle.state.y),
                                        angle=np.degrees(heading_rad),
                                        color="blue")

    def plot_steering_wheel(self):
        wheel_f = patches.Arrow(
                x=80.0,
                y=17.0,
                dx = 5*np.cos(self.road.vehicle.state.delta_f),
                dy = 5*np.sin(self.road.vehicle.state.delta_f),
                )
        wheel_b = patches.Arrow(
                x=70.0,
                y=17.0,
                dx = 5*np.cos(self.road.vehicle.state.delta_r),
                dy = 5*np.sin(self.road.vehicle.state.delta_r),
                )
        self.ax.add_patch(wheel_f)
        self.ax.add_patch(wheel_b)

    def plot_rectangle(self, rect, color, angle=0.0):
        rectangle = patches.Rectangle(
                        (rect.x-0.5*rect.length, rect.y-0.5*rect.width),
                        height=rect.width,
                        width=rect.length,
                        angle = angle,
                        facecolor=color)
        self.ax.add_patch(rectangle)
