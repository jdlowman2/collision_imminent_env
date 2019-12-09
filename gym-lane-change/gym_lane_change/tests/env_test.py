import sys
import unittest

sys.path.append("C:/Users/jdlow/Box/Fall 2019/ROB 590 RL for Driving/collision_environment/gym-lane-change/gym_lane_change/envs/")
sys.path.append("C:/Users/jdlow/Box/Fall 2019/ROB 590 RL for Driving/collision_environment/gym-lane-change/gym_lane_change/tests/")

from lane_change_env import *
import IPython


def make_dummy_vehicle(x, y):
    state = State(x, y, 0, 35, 0, 0, 0, 0)
    return Vehicle(state)

class TestRectangle(unittest.TestCase):
    def validate_origin_rect_inside(self, r):
        self.assertTrue(r.is_inside(0, 0))

        self.assertTrue(r.is_inside(2.5, 0))
        self.assertTrue(r.is_inside(-2.5, 0))

        self.assertTrue(r.is_inside(2.5, 5))
        self.assertTrue(r.is_inside(2.5, -5))

        self.assertFalse(r.is_inside(2.6, 0))
        self.assertFalse(r.is_inside(2.5, 5.1))
        self.assertFalse(r.is_inside(-2.6, -5.1))

    def test_shapes(self):
        r = Rectangle(length=5, width=10, x=0, y=0)
        self.validate_origin_rect_inside(r)

    def test_lane_inheritance(self):
        r = Lane(length=5, width=10, x=0, y=0)
        self.validate_origin_rect_inside(r)

    def test_obstacle_inheritance(self):
        r = Obstacle(length=5, width=10, x=0, y=0)
        self.validate_origin_rect_inside(r)

    def test_corners(self):
        r1 = Rectangle(1.0, 1.0, 0.5, 0.5)
        corners = r1.get_corners()
        self.assertTrue([1.0, 1.0] in corners)
        self.assertTrue([0.0, 1.0] in corners)
        self.assertTrue([1.0, 0.0] in corners)
        self.assertTrue([0.0, 0.0] in corners)

        r1 = Rectangle(1.0, 1.0, 0.0, 0.0)
        corners = r1.get_corners()
        self.assertTrue([0.5, 0.5] in corners)
        self.assertTrue([0.5, -0.5] in corners)
        self.assertTrue([-0.5, 0.5] in corners)
        self.assertTrue([-0.5, -0.5] in corners)

        r1 = Rectangle(1.0, 1.0, -.75, -.75)
        corners = r1.get_corners()
        self.assertTrue([-0.25, -0.25] in corners)
        self.assertTrue([-0.25, -1.25] in corners)
        self.assertTrue([-1.25, -0.25] in corners)
        self.assertTrue([-1.25, -1.25] in corners)

    def test_touching_rectangle(self):
        r1 = Rectangle(1.0, 1.0, 0.5, 0.5)
        r2 = Rectangle(1.0, 1.0, 0.75, 0.5)
        self.assertTrue(r1.intersects(r2))

        r1 = Rectangle(1.0, 1.0, 0.5, 0.5)
        r2 = Rectangle(1.0, 1.0, 0.75, 0.75)
        self.assertTrue(r1.intersects(r2))

        r1 = Rectangle(1.0, 1.0, 0.5, 0.5)
        r2 = Rectangle(1.0, 1.0, 0.5, 0.75)
        self.assertTrue(r1.intersects(r2))

        r1 = Rectangle(1.0, 1.0, 0.5, 0.5)
        r2 = Rectangle(1.0, 1.0, 0.0, 0.0)
        self.assertTrue(r1.intersects(r2))

        r1 = Rectangle(1.0, 1.0, 0.5, 0.5)
        r2 = Rectangle(1.0, 1.0, -0.25, -0.25)
        self.assertTrue(r1.intersects(r2))

        r1 = Rectangle(1.0, 1.0, 0.5, 0.5)
        r2 = Rectangle(1.0, 1.0, -0.499, -0.5)
        self.assertTrue(r1.intersects(r2))

    def test_inside_rectangle(self):
        r = Rectangle(10.0, 1.0, 5.0, 0.5)
        for i in np.linspace(0.0, 10.0, 200):
            for j in np.linspace(0.0, 1.0, 100):
                self.assertTrue(r.is_inside(i, j))


class TestRoad(unittest.TestCase):

    def test_road_lanes(self):
        road = Road()

        self.assertAlmostEqual(road.current_lane.get_left_boundary(),
                road.opposing_lane.get_right_boundary(), 5)

        total_road_width = road.opposing_lane.get_left_boundary() - \
                              road.current_lane.get_right_boundary()

        self.assertAlmostEqual(total_road_width, 2*LANE_WIDTH, 5)


class TestEnv(unittest.TestCase):

    def reset(self):
        plt.close('all')

    def test_vehicle_in_road(self):
        road = Road()
        self.assertTrue(road.is_vehicle_in_road())

        road.vehicle = make_dummy_vehicle(0.0, - 1.1 * LANE_WIDTH)
        self.assertFalse(road.is_vehicle_in_road())

        road.vehicle = make_dummy_vehicle(road.goal.x, road.goal.y)
        self.assertTrue(road.is_vehicle_in_road())

    def test_vehicle_in_collision(self):
        road = Road()
        self.assertFalse(road.is_vehicle_in_collision())

        road.vehicle = make_dummy_vehicle(road.obstacle.x, road.obstacle.y)
        self.assertTrue(road.is_vehicle_in_collision())

        road.vehicle = make_dummy_vehicle(road.obstacle.x - road.obstacle.length/2.0,
                                            road.obstacle.y)
        self.assertTrue(road.is_vehicle_in_collision())

        road.vehicle = make_dummy_vehicle(\
                        road.obstacle.x - road.obstacle.length/2.0- road.vehicle.length/2.0 - 1.0,
                        road.obstacle.y)
        self.assertFalse(road.is_vehicle_in_collision())


    def test_vehicle_in_goal(self):
        road = Road()
        self.assertFalse(road.is_vehicle_in_goal())

        road.vehicle = make_dummy_vehicle(road.goal.x, road.goal.y)
        self.assertTrue(road.is_vehicle_in_goal())

        road.vehicle = make_dummy_vehicle(road.goal.x - road.goal.length/2.0, road.goal.y)
        self.assertTrue(road.is_vehicle_in_goal())


        road.vehicle = make_dummy_vehicle(road.goal.x - \
                                road.goal.length/2.0 - road.vehicle.length/2.0 - 1.0, road.goal.y)
        self.assertFalse(road.is_vehicle_in_goal())


    def test_rewards(self):
        env = LaneChangeEnv(False)
        env.road.vehicle = make_dummy_vehicle(env.road.goal.x, env.road.goal.y)
        self.assertTrue(env.get_reward() > 0.99)

        env.road.vehicle = make_dummy_vehicle(0.0, 0.0)
        self.assertTrue(abs(env.get_reward()) < 1.0)

        env.road.vehicle = make_dummy_vehicle(50.0, 15.0)
        r_outside_lane = env.get_reward()
        self.assertTrue(r_outside_lane < 0.0)

        env.road.vehicle = make_dummy_vehicle(env.road.obstacle.x, env.road.obstacle.y)
        r_collision = env.get_reward()
        self.assertTrue(r_collision < r_outside_lane, f"collision reward {r_collision}, outside lane reward {r_outside_lane}")

    def test_sparse_rewards(self):
        env = LaneChangeEnv(True)
        env.road.vehicle = make_dummy_vehicle(env.road.goal.x, env.road.goal.y)
        self.assertTrue(env.get_reward() > 0.99)


    def test_reset(self):
        env = LaneChangeEnv()
        s1 = env.reset()
        s2 = env.reset()

        self.assertFalse((s1==s2).all())

    def test_car_corner_in_goal(self):
        self.reset()
        env = LaneChangeEnv()
        env.road.vehicle = make_dummy_vehicle(env.road.goal.x - 0.49*env.road.goal.get_length() - 0.4*env.road.vehicle.length,
                                                env.road.goal.y)
        # print("vehicle: ", env.road.vehicle.get_rectangle())
        # print("goal: ", env.road.goal)
        self.assertTrue(env.road.is_vehicle_in_goal(), "The environment should evaluate the vehicle as within the goal")

        self.reset()
        env = LaneChangeEnv()
        env.road.vehicle = make_dummy_vehicle(env.road.goal.x - env.road.vehicle.length/2.0-1.0,
                                        env.road.goal.y - env.road.vehicle.width/2.0 - 0.1)
        # env.render()
        # IPython.embed()

        self.assertTrue(env.road.is_vehicle_in_goal(), "The environment should evaluate the vehicle as within the goal")

if __name__ == '__main__':
    unittest.main()
