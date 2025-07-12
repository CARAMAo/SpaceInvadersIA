import os
import sys
import gym
import random
import gym.wrappers
import gym.wrappers
import gym.wrappers.atari_preprocessing
import gym.wrappers.frame_stack
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
from config import (
    lr,
    device,
    batch_size,
    async_update_step,
    memory_size,
    update_target,
    double_dqn,
    prioritized_memory,
    obs_mode,
)

import numpy as np

def play_game(env, net, num_games=30):
    scores = []
    q_values = []

    for i in range(num_games):
        state, _ = env.reset()
        total_reward = 0.
        steps = 0
        state = torch.Tensor(np.array(state)).to(device).unsqueeze(0)
        done = False
        while not done:
            if np.random.rand() <= 0.001:
                action = env.action_space.sample()
            else:
                q_value,indices = net(state).max(1)
                q_values.append(q_value.item())
                action = indices.item()
            
            next_state, reward, term, trunc, _ = env.step(action)
            # frame_stack.append(next_state)
            state = torch.Tensor(np.array(next_state)).to(device).unsqueeze(0)
            total_reward += reward
            steps += 1
            done = term or trunc 
        scores.append(total_reward)
    return (np.mean(scores),np.max(scores),np.mean(q_values))


def main():
    if obs_mode == "frame":
        env = gym.make("SpaceInvadersNoFrameskip-v4")
        env = gym.wrappers.atari_preprocessing.AtariPreprocessing(env)
        env = gym.wrappers.frame_stack.FrameStack(env, 4)
    else:
        env = gym.make("SpaceInvaders-ramNoFrameskip-v4")
        env = SIWrapper(
            env, normalize_reward=False, random_starts=False
        )
    # torch.manual_seed(500)
    #torch.multiprocessing.set_start_method("spawn")
    num_inputs = env.observation_space.shape
    num_actions = env.action_space.n
    print("state size:", num_inputs)
    print("action size:", num_actions)
    print(f"using {device.type}")
    if obs_mode == 'frame':
        online_net = CNNQNet()
        target_net = CNNQNet()
        eval_net = CNNQNet()
    else:
        online_net = QNet(num_inputs[0],num_actions)
        target_net = QNet(num_inputs[0],num_actions)
        eval_net = QNet(num_inputs[0],num_actions)
    eval_net.to(device)
    target_net.load_state_dict(online_net.state_dict())
    eval_net.load_state_dict(online_net.state_dict())
    online_net.share_memory()
    target_net.share_memory()

    N = mp.cpu_count()
    optimizer = SharedRMSprop(online_net.parameters(), lr=lr,alpha=0.95)
    global_ep, global_ep_r, global_step, res_queue, init_barrier = (
        mp.Value("i", 0),
        mp.Value("d", 0.0),
        mp.Value("i", 0),
        mp.Queue(),
        mp.Barrier(parties=N),
    )

    init_seed = 42 # random.getrandbits(32)

    online_net.to(device)
    target_net.to(device)
    online_net.train()
    target_net.train()
    eval_net.eval()

    run_name = f"{lr}_{async_update_step}_{batch_size}_{memory_size}_{obs_mode}_{datetime.now().strftime('%d-%m-%y-%H-%M-%S')}_{update_target}_{'DDQN' if double_dqn else 'DQN'}{'_prioritized' if prioritized_memory else ''}"
    writer = SummaryWriter(f"D:/logs/{run_name}")
    workers = [
        Worker(
            online_net,
            target_net,
            optimizer,
            global_ep,
            global_ep_r,
            global_step,
            memory_size,
            res_queue,
            init_barrier,
            init_seed,
            i,
        )
        for i in range(N)
    ]

    [w.start() for w in workers]
    score,max_score,avg_q = play_game(env, online_net)
    writer.add_scalar("log/score", score, 0)
    writer.add_scalar("log/max_score", max_score, 0)
    writer.add_scalar("log/avg_q", avg_q, 0)
    print("Training start, score:", score, "max",max_score,"Q:",avg_q)
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
            [w_name, step, log_type, epsilon, loss, lr_log] = r
            if log_type == "epoch_end":
                eval_net.load_state_dict(online_net.state_dict())
                score,max_score,avg_q = play_game(
                    env,
                    eval_net,
                )
                writer.add_scalar("log/score", score, step)
                writer.add_scalar("log/max_score", max_score, step)
                writer.add_scalar("log/avg_q", avg_q, step)
                print(f"Epoch {step}: score {score} loss {loss}, avg. Q {avg_q}")
                if not os.path.exists("D:/checkpoints/" + run_name):
                    os.makedirs("D:/checkpoints/" + run_name, exist_ok=True)
                torch.save(
                    {"model": eval_net.state_dict(), "step": step},
                    f"D:/checkpoints/{run_name}/checkpoint{step}",
                )
            else:
                writer.add_scalar("log/loss/" + w_name, float(loss), step)
                writer.add_scalar("log/epsilon/" + w_name, float(epsilon), step)
                writer.add_scalar("log/lr/" + w_name, float(lr_log), step)
            writer.flush()

        else:
            break
    [w.join() for w in workers]
    [w.close() for w in workers]


if __name__ == "__main__":
    main()
