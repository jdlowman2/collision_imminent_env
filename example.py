import gym
import numpy as np
import torch
import IPython

import sys
# a bit of a hack, need to register environment with gym
sys.path.append("gym-lane-change/gym_lane_change/envs/")

from lane_change_env import LaneChangeEnv


env = LaneChangeEnv(sparse_reward=True)

state = env.reset()
done = False
episode_reward = 0.0

while not done:
    env.render()
    state, reward, done, _ = env.step(env.action_space.sample()) # random action

    episode_reward += reward

print("Total episode reward: ", round(episode_reward, 2))
