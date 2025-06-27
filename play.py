import gym
from gym_wrappers import SIWrapper

from gym.utils import play

env = gym.make("SpaceInvaders-ramNoFrameskip-v4", render_mode="rgb_array")

play.play(env, zoom=2)
