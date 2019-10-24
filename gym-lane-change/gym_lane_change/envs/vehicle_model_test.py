from vehicle_model import *
import unittest
import IPython


START_STATE = State(0, 0, 0, 35, 0, 0, 0, 0)

def make_dummy_vehicle(x, y, v):
    state = State(x, y, 0, v, 0, 0, 0, 0)
    return Vehicle(state)



class TestVehicle(unittest.TestCase):

    def test_no_input(self):
        vehicle = make_dummy_vehicle(0.0, 0.0, 0.0)
        for i in range(2):
            vehicle.step([0.0, 0.0])
            print("Vehicle state: ")
            print(vehicle.state)
            self.assertTrue(vehicle.state, State(0.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0,
                                                 0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
