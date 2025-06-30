
import os
import sys
import gym
import random
import gym.wrappers
import gym.wrappers.step_api_compatibility
import numpy as np

from datetime import datetime

import torch
import torch.optim as optim
import torch.nn.functional as F
from model import *
from worker import Worker
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

from si_wrappers import *

from shared_adam import SharedAdam
from shared_rmsprop import SharedRMSprop
from collections import deque
from config import lr, device,batch_size,async_update_step,memory_size

def play_game(env,net,num_games=10):
    avg_score = 0.
    for i in range(num_games):
      state,_ = env.reset()
      # frame_stack = deque([state]*4,maxlen=4)

      state = torch.Tensor(state).to(device).unsqueeze(0)
      done = False
      while not done:
        action = env.action_space.sample() if np.random.rand() < .05 else net(state).max(1).indices.view(1, 1).item()
        next_state,reward,term,trunc,_ = env.step(action)
        # frame_stack.append(next_state)
        state = torch.Tensor(next_state).to(device).unsqueeze(0)
        avg_score += reward
        done = term or trunc
    return avg_score/num_games


def main():
    env = gym.make("SpaceInvaders-ramNoFrameskip-v4")
    env = SIWrapper(env,normalize_reward=False,random_starts=False)
    torch.manual_seed(500)
    torch.multiprocessing.set_start_method("spawn")
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)
    print(f'using {device.type}')
    online_net = QNet(num_inputs,num_actions)
    target_net = QNet(num_inputs,num_actions)

    target_net.load_state_dict(online_net.state_dict())
    online_net.share_memory()
    target_net.share_memory()

    # optimizer = SharedAdam(online_net.parameters(), lr=lr,eps=0.01,betas=(.95,.95))
    optimizer = SharedRMSprop(online_net.parameters(), lr=lr)
    global_ep, global_ep_r,global_step, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Value('i',0), mp.Queue()

    writer = SummaryWriter(f"logs/{datetime.now().isoformat()}")

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()

    N =  mp.cpu_count()
    workers = [Worker(online_net, target_net, optimizer, global_ep, global_ep_r,global_step, memory_size, res_queue, i) for i in range(N)]
    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
            [w_name,ep, avg_train_score, epsilon, loss] = r
            score = play_game(env,online_net,)
            writer.add_scalar('log/avg_train_score',avg_train_score, ep)
            writer.add_scalar('log/score',score, ep)
            writer.add_scalar('log/loss', float(loss), ep)
            writer.flush()
            torch.save({'model':online_net.state_dict(),'step':ep},f'checkpoint{ep}')
        else:
            break
    [w.join() for w in workers]
    [w.close() for w in workers]


if __name__=="__main__":
    main()
