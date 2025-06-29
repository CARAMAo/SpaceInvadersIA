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
from config import (
    lr,
    device,
    async_update_step,
    memory_size,
    obs_mode,
    update_target,
    max_evaluation_steps,
    frame_skip,
    max_steps,
    layer_size,
)

first_evaluation = True
q_evaluation_states = []


def play_game(env, net, num_games=10):
    global first_evaluation
    global q_evaluation_states
    avg_score = 0.0
    for i in range(num_games):
        state, _ = env.reset()

        state = torch.Tensor(state).to(device).unsqueeze(0)
        done = False
        steps = 0
        while not done:
            action = (
                env.action_space.sample()
                if np.random.rand() < 0.05
                else net(state).max(1).indices.view(1, 1).item()
            )
            next_state, reward, term, trunc, _ = env.step(action)
            # frame_stack.append(next_state)
            state = torch.Tensor(next_state).to(device).unsqueeze(0)
            if first_evaluation and len(q_evaluation_states) < 1024:
                q_evaluation_states.append(state)
            avg_score += reward
            done = term or trunc or steps >= max_evaluation_steps
    first_evaluation = False
    avg_q = net(torch.cat(q_evaluation_states).to(device)).mean().item()
    return (avg_score / num_games, avg_q)


def main():
    print(first_evaluation)
    env = make_env(
        obs_mode, normalize_reward=False, random_starts=True, frame_skip=frame_skip
    )
    torch.manual_seed(500)
    torch.multiprocessing.set_start_method("spawn")
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print("state size:", num_inputs)
    print("action size:", num_actions)
    print(f"using {device.type}")
    if obs_mode == "image":
        online_net = CNNQNet(4, num_actions)
        target_net = CNNQNet(4, num_actions)
        eval_net = CNNQNet(4, num_actions)
    else:
        online_net = QNet(num_inputs, num_actions)
        target_net = QNet(num_inputs, num_actions)
        eval_net = QNet(num_inputs, num_actions)
    eval_net.load_state_dict(online_net.state_dict())
    target_net.load_state_dict(online_net.state_dict())
    online_net.share_memory()
    target_net.share_memory()

    N = mp.cpu_count()

    # optimizer = SharedAdam(online_net.parameters(), lr=lr,eps=0.01,betas=(.95,.95))

    optimizer = SharedRMSprop(online_net.parameters(), lr=lr)
    optimizer.share_memory()

    global_epoch, global_update_step, global_step, res_queue, init_barrier = (
        mp.Value("i", 0),
        mp.Value("i", 0),
        mp.Value("i", 0),
        mp.Queue(),
        mp.Barrier(N),
    )
    run_name = f"{obs_mode}/{lr}_{memory_size}_{update_target}_{max_steps}_{async_update_step}_{layer_size}"
    writer = SummaryWriter(f"logs/{run_name}")
    epsilon_tags = [f"log/epsilon/w{i}" for i in range(N)]
    layout = {"epsilon": {run_name: ["multiline", epsilon_tags]}}
    writer.add_custom_scalars(layout)
    eval_net.to(device)
    online_net.to(device)
    target_net.to(device)
    eval_net.eval()
    online_net.train()
    target_net.train()

    print(f"Using {N} workers")
    workers = [
        Worker(
            online_net,
            target_net,
            optimizer,
            global_epoch,
            global_update_step,
            global_step,
            memory_size,
            res_queue,
            init_barrier,
            i,
        )
        for i in range(N)
    ]
    [w.start() for w in workers]
    res = []
    max_score, avg_q = play_game(env, eval_net)
    writer.add_scalar("log/score", max_score, 0)
    writer.add_scalar("log/avg_q", avg_q, 0)
    print(f"Epoch: 0, Score: {max_score}, Avg Q {avg_q}")
    step = 0
    while True:

        r = res_queue.get()
        if r is not None:
            res.append(r)
            [w_name, epoch, evaluate, epsilon] = r
            if evaluate:
                eval_net.load_state_dict(online_net.state_dict())
                score, avg_q = play_game(
                    env,
                    eval_net,
                )
                if score > max_score:
                    max_score = score
                    if not os.path.exists(f"checkpoints/{run_name}/"):
                        os.makedirs(f"checkpoints/{run_name}/")
                    torch.save(
                        {"model": eval_net.state_dict(), "step": epoch},
                        f"checkpoints/{run_name}/checkpoint{epoch}",
                    )
                # writer.add_scalar("log/avg_train_score", avg_train_score, epoch)
                writer.add_scalar("log/score", score, epoch)
                writer.add_scalar("log/avg_q", avg_q, epoch)
                print(f"Epoch: {epoch}, Score: {score}, Avg Q: {avg_q}")
            else:
                writer.add_scalar("log/epsilon/" + w_name, epsilon, epoch)
            writer.flush()
        else:
            break

    [w.join() for w in workers]
    [w.close() for w in workers]


if __name__ == "__main__":
    main()
