import gym
from gym.spaces import *
from gym_wrappers import *

import warnings
import random

import numpy as np
import os
import datetime
import math
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import threading as th
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dqn import DQN
from prm import Transition,PrioritizedReplayMemory
import time


N_THREADS = 8
T_max = 1#32_000
# env = SIWrapper(env,skip_pauses=True,frame_skip=4)

def async_actor(i,steps):
    start = time.time()
    global T
    env = SIWrapper(gym.make("SpaceInvaders-ramNoFrameskip-v4"),frame_skip=4,skip_pauses=True)

    #optimizer
    
    env.reset()
    t = 0
    while t <= steps:
        obs,reward,term,trunc,info = env.step(env.action_space.sample())
        to = torch.tensor(obs,requires_grad=True).cuda()
        vi = torch.pow(to,torch.tensor([2.]*len(obs)).cuda()) + torch.randn([len(obs)]).cuda()
        vi.max().backward()
        print(vi.grad)
        done = term or trunc
        if done:
            env.reset()
        t+=1
    end = time.time()
    # print(f"{i} done {steps} steps in {end-start}")

if __name__ == '__main__':
    # workers = [ mp.Process(target=async_actor,args=(i,T_max // N_THREADS,),daemon=True) for i in range(N_THREADS)]
    # s_t = time.time()
    # for worker in workers:
    #     worker.start()
    
    # for worker in workers:
    #     worker.join()
    # e_t = time.time()


    s_t2 = time.time()
    async_actor(-1,T_max)
    e_t2 = time.time()

    # print(N_THREADS,"done in",e_t - s_t)
    # print("seq done in",e_t2 - s_t2)
