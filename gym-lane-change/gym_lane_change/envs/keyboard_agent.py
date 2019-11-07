#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import sys, gym, time
from pynput import keyboard
import time

sys.path.append("C:/Users/jdlow/Box/Fall 2019/ROB 590 RL for Driving/collision_environment/gym-lane-change/gym_lane_change/envs/")
from lane_change_env import LaneChangeEnv

def get_action():
    action = input("")
    if "q" in action:
        return None # throws an error to quit
    action = np.array([0.0, 0.0])
    return action


def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

    if key.char == "a":
        global action
        action = np.array([1.0, 0.0])

def on_release(key):
    print('{0} released'.format(
        key))

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

env = LaneChangeEnv()

state = env.reset()
done = False

while not done:
    env.render()

    if listener.has_input():
        action = listener.key()
    else:
        action = get_action()
    state, reward, done, info = env.step(action)
    time.sleep(0.01)

print("Finished! ")

env.close()
