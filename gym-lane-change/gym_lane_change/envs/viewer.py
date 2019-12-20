import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os, sys
from pathlib import Path
import IPython

from vehicle_model import *
from road_model import *

class Viewer:
    def __init__(self, road):
        plt.ion() ## interactive to prevent blocking
        self.road = road
        self.fig, self.ax = plt.subplots(1, 1, num='LaneChangeEnv')
        self.reset()
        plt.pause(0.5)
        # self.car_img = plt.imread("car.png")

    def reset(self):
        self.is_done = False
        self.last_reward = 0.0
        self.show()
        plt.pause(0.1)

    def update_data(self, road):
        self.road = road

    def show(self):
        self.clear_screen()
        self.plot_lanes()

        self.plot_rectangle(self.road.obstacle, "red")
        self.plot_rectangle(self.road.goal, "green")

        self.plot_vehicle()
        self.plot_steering()
        self.plot_status()

        self.ax.set_xlim([-5, 100])
        self.ax.set_ylim([-10, 20])

        plt.show()
        plt.pause(0.001)

    def clear_screen(self):
        for txt in self.ax.texts:
            txt.remove()

        for patch in self.ax.patches:
            patch.remove()

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
        td2dis = self.ax.transData
        coords = td2dis.transform([self.road.vehicle.state.x,
                                    self.road.vehicle.state.y])
        tr = mpl.transforms.Affine2D().rotate_around(coords[0], coords[1],
                                                self.road.vehicle.state.psi)
        t = td2dis + tr

        vehicle = patches.Rectangle(
                        (self.road.vehicle.state.x - 0.5*self.road.vehicle.length,
                            self.road.vehicle.state.y - 0.5*self.road.vehicle.width),
                        height=self.road.vehicle.width,
                        width=self.road.vehicle.length,
                        facecolor="blue",
                        transform=t,
                        zorder=100)


        # heading = patches.Arrow(
        #         x=self.road.vehicle.state.x,
        #         y=self.road.vehicle.state.y,
        #         dx = self.road.vehicle.length * np.cos(self.road.vehicle.state.psi),
        #         dy = 30/110.0*self.road.vehicle.length * np.sin(self.road.vehicle.state.psi),
        #         color="white",
        #         zorder=101)

        self.ax.add_patch(vehicle)
        # self.ax.add_patch(heading)

    def plot_steering(self):
        y_scale = 30.0/105.0
        arrow_scale = 5.0

        delta_f_norm = np.linalg.norm(self.road.vehicle.state.delta_f)
        delta_r_norm = np.linalg.norm(self.road.vehicle.state.delta_r)

        wheel_f = patches.Arrow(\
                x=80.0,
                y=14.0,
                dx = arrow_scale * np.cos(-self.road.vehicle.state.delta_f),
                dy = arrow_scale * y_scale * np.sin(-self.road.vehicle.state.delta_f),
                )
        wheel_r = patches.Arrow(\
                x=70.0,
                y=14.0,
                dx = arrow_scale * np.cos(-self.road.vehicle.state.delta_r),
                dy = arrow_scale * y_scale * np.sin(-self.road.vehicle.state.delta_r),
                )

        self.ax.add_patch(wheel_f)
        self.ax.add_patch(wheel_r)

    def plot_status(self):
        if self.is_done:
            txt = "DONE"
            c = "green"
        else:
            txt = "RUNNING"
            c = "yellow"

        self.ax.text(x=0.0, y=15.0, s="Status:\n"+txt, color=c,
            bbox={'facecolor': 'black', 'pad': max(0, 7-len(txt))})

        self.ax.text(x=40.0, y=-8.0,
            s="Reward:\n"+str(round(self.last_reward, 2)),
            color="black")

        self.ax.text(x=30.0, y=17.0,
            s=f"Speed: {round(self.road.vehicle.state.u, 2)} m/s",
            color="black")
        self.ax.text(x=30.0, y=14.0,
            s=f"Heading: {round(np.degrees(self.road.vehicle.state.psi), 2)} deg",
            color="black")


        self.ax.text(x=80.0, y=16.0, s="Front\nSteer", color="black")
        self.ax.text(x=70.0, y=16.0, s="Rear\nSteer", color="black")


    def plot_rectangle(self, rect, color, angle=0.0):
        rect = patches.Rectangle(
                        (rect.x-0.5*rect.length, rect.y-0.5*rect.width),
                        height=rect.width,
                        width=rect.length,
                        angle = angle,
                        facecolor=color)
        self.ax.add_patch(rect)

    def close(self):
        plt.close('all')
