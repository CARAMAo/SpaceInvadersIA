import gym
import gym.wrappers
import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Lock
import numpy as np
from model import QNet
from memory import ReplayMemory, Transition
from si_wrappers import *
from copy import deepcopy
from config import *
from collections import deque


class Worker(mp.Process):
    def __init__(
        self,
        online_net,
        target_net,
        optimizer,
        global_epoch,
        global_update_steps,
        global_step,
        res_queue,
        init_barrier,
        name,
    ):
        super(Worker, self).__init__(daemon=True)

        self.env = make_env(
            obs_mode, frame_skip=4, random_starts=True, episodic_life=True
        )

        self.name = "w%i" % name
        (
            self.global_epoch,
            self.global_update_steps,
            self.global_step,
            self.res_queue,
            self.init_barrier,
        ) = (global_epoch, global_update_steps, global_step, res_queue, init_barrier)
        self.online_net, self.target_net, self.optimizer = (
            online_net,
            target_net,
            optimizer,
        )
        self.memory = ReplayMemory(async_update_step)

    def record(self, epoch, evaluate, epsilon=None):
        self.res_queue.put([self.name, epoch, evaluate, epsilon])

    def update_target_model(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def optimize_model(self, s, a, r, s1, done):

        s = torch.cat(s).to(device=device, dtype=torch.float32)
        a = torch.cat(a).to(device=device, dtype=torch.long)
        r = torch.cat(r).to(device=device, dtype=torch.float32)
        s1 = torch.cat(s1).to(device=device, dtype=torch.float32)
        done = torch.cat(done).to(device=device, dtype=torch.float32)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.online_net(s).gather(1, a)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(async_update_step, device=device)
        with torch.no_grad():
            next_state_values = self.target_net(s1).max(1).values
            # Compute the expected Q values
            expected_state_action_values = (done * next_state_values * gamma) + r

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()

        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return torch.tensor(
                [[self.env.action_space.sample()]], device=device, dtype=torch.long
            )
        else:
            with torch.no_grad():
                return self.online_net(state.to(device)).max(1).indices.view(1, 1)

    def run(self):

        global_update_step = 0
        steps = 0

        epsilon_update_target = 50_000

        epsilon = np.random.uniform(epsilon_end, epsilon_start)

        self.record(0, False, epsilon)

        while self.global_epoch.value < training_epochs:

            done = False

            score = 0
            (state, _) = self.env.reset()

            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            while not done:

                action = self.get_action(state, epsilon)
                next_state, reward, term, trunc, _ = self.env.step(action.item())

                done = term or trunc

                next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                reward = torch.tensor([reward], dtype=torch.float32)
                done_t = torch.tensor([not done], dtype=torch.float32)
                self.memory.push(state, action, reward, next_state, done_t)

                score += reward
                state = next_state

                with self.global_step.get_lock():
                    self.global_step.value += 1
                    global_step = self.global_step.value
                    if global_step % steps_per_epoch == 0:
                        self.global_epoch.value += 1
                        self.record(
                            self.global_epoch.value,
                            True,
                            epsilon,
                        )

                steps += 1

                # cycle_pos = ((steps/period) + phase)%1.0
                # norm_value = 1 - abs(2*cycle_pos - 1)
                # epsilon = epsilon_end + (epsilon_start - epsilon_end)*norm_value

                if steps % epsilon_update_target == 0:
                    epsilon = np.random.uniform(0.05, 0.8)
                    self.record(steps, False, epsilon)

                epsilon *= 0.99999
                epsilon = max(epsilon, epsilon_end)

                if done or steps % async_update_step == 0:
                    if len(self.memory) > 0:
                        s, a, r, s1, done_t = self.memory.sample(async_update_step)
                        self.memory = ReplayMemory(async_update_step)
                        self.optimize_model(s, a, r, s1, done_t)
                if global_step % update_target == 0:
                    self.update_target_model()

        self.res_queue.put(None)
        print(self.name, "steps", steps, "updates", global_update_step)
