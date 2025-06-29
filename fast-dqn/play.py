import gym
from gym.utils import play
from si_wrappers import make_env


def callback(*args, **kwargs):
    print(args[0])


env = make_env("condensed_ram")

play.play(env, zoom=2, fps=10, callback=callback)
