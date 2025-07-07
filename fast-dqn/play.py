import os
import sys
from time import sleep
import torch
import gym
from model import QNet
from gym.utils import play
from si_wrappers import SIWrapper, DiscreteSI
import cv2
import numpy as np

if __name__ == "__main__":
    # checkpoint = torch.load(sys.argv[1], map_location="cuda")
    obs_mode = "frame"

    if obs_mode == "frame":
        env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="human")
        env = gym.wrappers.atari_preprocessing.AtariPreprocessing(env)
        env = gym.wrappers.frame_stack.FrameStack(env, 4)
    else:
        env = gym.make("SpaceInvaders-ramNoFrameskip-v4", render_mode="rgb_array")
        env = SIWrapper(
            env,
            frame_skip=4,
            random_starts=False,
            normalize_reward=False,
        )

    num_inputs = env.observation_space.shape
    num_outputs = env.action_space.n

    print(num_inputs)
    (state, _) = env.reset()

    done = False
    while not done:
        state, reward, term, trunc, _ = env.step(1)

        done = term or trunc
