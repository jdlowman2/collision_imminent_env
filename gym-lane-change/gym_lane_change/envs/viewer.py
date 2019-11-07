import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from vehicle_model import *
from road_model import *

class Viewer:
    def __init__(self, road):
        plt.ion() ## interactive to prevent blocking
        self.road = road
        self.fig, self.ax = plt.subplots(1, 1, num='LaneChangeEnv')
        self.is_done = False
        self.last_reward = 0.0
        self.text = {"status": None, "reward": None}

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
        plt.pause(0.1)

    def clear_screen(self):
        for text_key in list(self.text.keys()):
            try:
                self.text[text_key].set_visible(False)
            except:
                pass

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
        td2dis = self.ax.transData
        coords = td2dis.transform([self.road.vehicle.state.x,
                                    self.road.vehicle.state.y])
        tr = mpl.transforms.Affine2D().rotate_around(coords[0], coords[1],
                                                self.road.vehicle.state.psi)
        t = td2dis + tr

        r = patches.Rectangle(
                        (self.road.vehicle.state.x - 0.5*self.road.vehicle.length,
                            self.road.vehicle.state.y - 0.5*self.road.vehicle.width),
                        height=self.road.vehicle.width,
                        width=self.road.vehicle.length,
                        facecolor="blue",
                        transform=t)
        self.ax.add_patch(r)

    def plot_steering(self):
        delta_f_norm = np.linalg.norm(self.road.vehicle.state.delta_f)
        delta_r_norm = np.linalg.norm(self.road.vehicle.state.delta_r)
        wheel_f = patches.Arrow(
                x=80.0,
                y=14.0,
                dx = 5*np.cos(-self.road.vehicle.state.delta_f),
                dy = 5*np.sin(-self.road.vehicle.state.delta_f),
                )
        wheel_b = patches.Arrow(
                x=70.0,
                y=14.0,
                dx = 5*np.cos(-self.road.vehicle.state.delta_r),
                dy = 5*np.sin(-self.road.vehicle.state.delta_r),
                )
        self.ax.text(x=80.0, y=16.0, s="Front\nSteer", color="black")
        self.ax.text(x=70.0, y=16.0, s="Rear\nSteer", color="black")
        self.ax.add_patch(wheel_f)
        self.ax.add_patch(wheel_b)

    def plot_status(self):
        if self.is_done:
            txt = "DONE"
            c = "green"
        else:
            txt = "RUNNING"
            c = "yellow"

        self.text["status"] = self.ax.text(x=0.0, y=15.0, s="Status:\n"+txt, color=c,
            bbox={'facecolor': 'black', 'pad': max(0, 7-len(txt))})

        self.text["reward"] = self.ax.text(x=40.0, y=15.0,
            s="Reward:\n"+str(round(self.last_reward, 2)),
            color="black")

    def plot_rectangle(self, rect, color, angle=0.0):
        rectangle = patches.Rectangle(
                        (rect.x-0.5*rect.length, rect.y-0.5*rect.width),
                        height=rect.width,
                        width=rect.length,
                        angle = angle,
                        facecolor=color)
        self.ax.add_patch(rectangle)
