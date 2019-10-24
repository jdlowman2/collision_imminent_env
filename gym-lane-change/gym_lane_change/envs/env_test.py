from lane_change_env import *
import unittest
import IPython


def make_dummy_vehicle(x, y):
    state = State(x, y, 0, 35, 0, 0, 0, 0)
    return Vehicle(state)

class TestRectangle(unittest.TestCase):
    def validate_inside(self, r):
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
        self.validate_inside(r)

    def test_lane_inheritance(self):
        r = Lane(length=5, width=10, x=0, y=0)
        self.validate_inside(r)

    def test_obstacle_inheritance(self):
        r = Obstacle(length=5, width=10, x=0, y=0)
        self.validate_inside(r)

class TestRoad(unittest.TestCase):

    def test_road_lanes(self):
        road = Road()

        self.assertAlmostEqual(road.current_lane.get_left_boundary(),
                road.opposing_lane.get_right_boundary(), 5)

        total_road_width = road.opposing_lane.get_left_boundary() - \
                              road.current_lane.get_right_boundary()

        self.assertAlmostEqual(total_road_width, 2*LANE_WIDTH, 5)


class TestEnv(unittest.TestCase):

    def test_vehicle_in_road(self):
        road = Road()
        self.assertTrue(road.is_vehicle_in_road())

        road.vehicle = make_dummy_vehicle(0.0, - 1.1 * LANE_WIDTH)
        self.assertFalse(road.is_vehicle_in_road())

        road.vehicle = make_dummy_vehicle(500.0, 0.0)
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
                        road.obstacle.x - road.obstacle.length/2.0 - 1.0,
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
                                road.goal.length/2.0 -1.0, road.goal.y)
        self.assertFalse(road.is_vehicle_in_goal())


    def test_rewards(self):
        road = Road()
        ## TODO

if __name__ == '__main__':
    unittest.main()
