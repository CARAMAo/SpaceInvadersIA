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
        global_ep,
        global_ep_r,
        global_step,
        memory_size,
        res_queue,
        init_barrier,
        name,
    ):
        super(Worker, self).__init__(daemon=True)

        self.env = make_env(obs_mode, frame_skip=4, episodic_life=True)

        self.name = "w%i" % name
        (
            self.global_ep,
            self.global_ep_r,
            self.global_step,
            self.res_queue,
            self.init_barrier,
        ) = (global_ep, global_ep_r, global_step, res_queue, init_barrier)
        self.online_net, self.target_net, self.optimizer = (
            online_net,
            target_net,
            optimizer,
        )

        self.frame_stack = deque([], maxlen=4)
        self.memory = ReplayMemory(memory_size)

    def log_state(self, message):
        """
        Log the state of the worker with a custom message, including the timestamp,
        worker name, and the state of online_net and target_net.
        Inoltre, calcola e stampa le differenze tra target_net e online_net.
        """
        import datetime

        # Calcola le differenze tra target_net e online_net
        differences = {}
        for name, param in self.target_net.state_dict().items():
            if name in self.online_net.state_dict():
                diff = torch.sum(torch.abs(param - self.online_net.state_dict()[name]))
                differences[name] = diff.item()

        print(
            f"[{datetime.datetime.now()}] {message} | Worker: {self.name} | "
            # f"Online Net: {self.online_net.state_dict()} | Target Net: {self.target_net.state_dict()} | "
            f"Differences: {differences}"
        )

    def record(self, steps, score, epsilon, loss):
        with self.global_ep.get_lock():
            self.global_ep.value += 1
            self.global_ep_r.value += score
            # with self.global_ep_r.get_lock():
            #     if self.global_ep_r.value == 0.:
            #         self.global_ep_r.value = score
            #     else:
            #         self.global_ep_r.value = 0.99 * self.global_ep_r.value + 0.01 * score
            self.res_queue.put([self.name,steps,None,epsilon,None])
            if self.global_ep.value % log_interval == 0:
                print(
                    "{} , {} step | score: {:.2f}, | epsilon: {:.2f} | loss: {}".format(
                        self.name,
                        steps,
                        self.global_ep_r.value / log_interval,
                        epsilon,
                        loss,
                    )
                )
                self.res_queue.put(
                    [
                        self.name,
                        steps,
                        self.global_ep_r.value / log_interval,
                        epsilon,
                        loss,
                    ]
                )
                self.global_ep_r.value = 0.0

    def update_target_model(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def optimize_model(self, s, a, r, s1, done):

        s = torch.cat(s).to(device)
        a = torch.cat(a).to(device)
        r = torch.cat(r).to(device)
        s1 = torch.cat(s1).to(device)
        done = torch.cat(done).to(device)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.online_net(s).gather(1, a)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size, device=device)
        with torch.no_grad():
            next_state_values = self.target_net(s1).max(1).values
            # Compute the expected Q values
            expected_state_action_values = (done * next_state_values * gamma) + r

        # Compute Huber loss
        criterion = torch.nn.MSELoss()

        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        return loss

    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return torch.tensor(
                [[self.env.action_space.sample()]], device=device, dtype=torch.long
            )
        else:
            with torch.no_grad():
                return self.online_net(state.to(device)).max(1).indices.view(1, 1)

    def run(self):
        self.online_net.random_init(seed=42)
        self.target_net.random_init(seed=42)

        self.init_barrier.wait()

        
        #period = 100_000
        #amp = epsilon_start - epsilon_end
        #phase = np.random.rand()
        steps = 0
        #norm_value = 1 - abs(2*phase - 1)
        #epsilon = epsilon_end + (epsilon_start - epsilon_end)*norm_value
        epsilon = epsilon_start
        while self.global_step.value < max_steps:
            # if self.global_ep_r.value > goal_score:
            #     break
            done = False

            score = 0
            (state, _) = self.env.reset()

            state = torch.Tensor(state).unsqueeze(0)

            while not done:

                action = self.get_action(state, epsilon)
                next_state, reward, term, trunc, _ = self.env.step(action.item())
                # self.frame_stack.append(next_state)

                done = term or trunc

                next_state = torch.Tensor(next_state).unsqueeze(0)
                reward = torch.Tensor([reward])
                done_t = torch.Tensor([not done])
                self.memory.push(state, action, reward, next_state, done_t)

                score += reward
                # if done:
                #     (state,_) = self.env.reset()
                #     state = torch.Tensor(state).to(device).unsqueeze(0)
                # else:
                state = next_state

                if len(self.memory) > pre_training:
                    with self.global_step.get_lock():
                        self.global_step.value += 1
                        global_step = self.global_step.value
                    steps += 1
                    #cycle_pos = ((steps/period) + phase)%1.0
                    #norm_value = 1 - abs(2*cycle_pos - 1)
                    #epsilon = epsilon_end + (epsilon_start - epsilon_end)*norm_value
                    
                    epsilon -= eps_decay
                    epsilon = max(epsilon, epsilon_end)
                    #if steps % 100_000 == 0 and epsilon == epsilon_end and np.random.rand() > .4:
                    #    epsilon = np.random.uniform(high=epsilon_start)

                    if done or steps % async_update_step == 0:
                        s, a, r, s1, done_t = self.memory.sample(batch_size)
                        loss = self.optimize_model(s, a, r, s1, done_t).item()
                        # memory = ReplayMemory(async_update_step)
                        if done:
                            self.record(global_step, score.item(), epsilon, loss)
                            break
                    if global_step % update_target == 0:
                        self.update_target_model()


        self.res_queue.put(None)
