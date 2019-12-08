import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='LaneChange-v0',
    entry_point='gym_lane_change.envs:LaneChangeEnv',
    timestep_limit=500,
    reward_threshold=1.0,
    nondeterministic = True,
)
