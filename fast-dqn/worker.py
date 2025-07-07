import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack
import torch
import torch.multiprocessing as mp
import numpy as np
from model import QNet
from memory import ReplayMemory, Transition, PrioritizedReplayMemory
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
        init_seed,
        name,
    ):
        super(Worker, self).__init__(daemon=True)
        if obs_mode == "frame":
            self.env = gym.make("SpaceInvadersNoFrameskip-v4")
            self.env = AtariPreprocessing(self.env)
            self.env = FrameStack(self.env, 4)
        else:
            self.env = gym.make("SpaceInvaders-ramNoFrameskip-v4")
            self.env = SIWrapper(
                self.env, frame_skip=4, episodic_life=False, obs_mode="frames"
            )
        self.init_barrier = init_barrier
        self.name = "w%i" % name
        self.global_ep, self.global_ep_r, self.global_step, self.res_queue = (
            global_ep,
            global_ep_r,
            global_step,
            res_queue,
        )
        self.online_net, self.target_net, self.optimizer = (
            online_net,
            target_net,
            optimizer,
        )
        self.init_seed = init_seed
        if prioritized_memory:
            self.memory = PrioritizedReplayMemory(memory_size)
        else:
            self.memory = ReplayMemory(memory_size)

    def record(self, steps, log_type, epsilon=None, loss=None, lr=None):

        self.res_queue.put(
            [
                self.name,
                steps,
                log_type,
                epsilon,
                loss,
                lr,
            ]
        )

    def update_target_model(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def optimize_model(self, beta=0.4):

        if len(self.memory) < batch_size:
            return 0

        if isinstance(self.memory, PrioritizedReplayMemory):
            s, a, r, s1, done, indices, weights = self.memory.sample(
                batch_size, beta=beta
            )
            s = torch.cat(list(s))
            a = torch.cat(list(a))
            r = torch.cat(list(r))
            s1 = torch.cat(list(s1))
            done = torch.cat(list(done))
            weights = torch.tensor(weights, device=device).unsqueeze(1)
        else:
            s, a, r, s1, done = self.memory.sample(batch_size)
            s = torch.cat(list(s))
            a = torch.cat(list(a))
            r = torch.cat(list(r))
            s1 = torch.cat(list(s1))
            done = torch.cat(list(done))
            weights = None

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.online_net(s).gather(1, a)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            if double_dqn:
                next_actions = self.online_net(s1).max(1).indices.unsqueeze(1)
                next_state_values = (
                    self.target_net(s1).gather(1, next_actions).squeeze(1)
                )
            else:
                next_state_values = self.target_net(s1).max(1).values
            # Compute the expected Q values
            expected_state_action_values = (done * next_state_values * gamma) + r

        # Compute loss
        criterion = torch.nn.MSELoss(reduction="none")

        losses = criterion(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )
        if weights is not None:
            loss = (losses * weights).mean()
        else:
            loss = losses.mean()

        loss_value = loss.item()
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        if isinstance(self.memory, PrioritizedReplayMemory):
            td_errors = (
                (
                    state_action_values.detach()
                    - expected_state_action_values.unsqueeze(1)
                )
                .abs()
                .cpu()
                .numpy()
            )
            new_priorities = td_errors + 1e-6
            self.memory.update_priorities(indices, new_priorities.flatten())

        return loss_value

    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return torch.tensor(
                [[self.env.action_space.sample()]], device=device, dtype=torch.long
            )
        else:
            with torch.no_grad():
                return self.online_net(state).max(1).indices.view(1, 1)

    def run(self):
        epsilon = epsilon_start
        epsilon_min = np.random.choice(epsilon_end, p=epsilon_distribution)
        steps = 0
        global_step = 0
        loss = 0
        self.online_net.init(seed=self.init_seed)
        self.target_net.init(seed=self.init_seed)
        self.init_barrier.wait()

        while global_step // epoch_steps < max_epochs:

            done = False

            score = 0
            (state, _) = self.env.reset()

            state = torch.Tensor(state.array()).to(device).unsqueeze(0)

            while not done:

                action = self.get_action(state, epsilon)
                next_state, reward, term, trunc, _ = self.env.step(action.item())
                # self.frame_stack.append(next_state)

                done = term or trunc

                next_state = torch.Tensor(next_state.array()).to(device).unsqueeze(0)
                reward = torch.Tensor([reward]).to(device)
                done_t = torch.Tensor([not done]).to(device)
                self.memory.push(state, action, reward, next_state, done_t)

                score += reward
                # if done:
                #     (state,_) = self.env.reset()
                #     state = torch.Tensor(state).to(device).unsqueeze(0)
                # else:
                state = next_state

                if len(self.memory) >= pre_training:
                    with self.global_step.get_lock():
                        self.global_step.value += 1
                        global_step = self.global_step.value
                        # if global_step >= 20 * epoch_steps:
                        #    self.optimizer.param_groups[0]["lr"] = lr / (
                        #        10
                        #        * (
                        #            (global_step - 20 * epoch_steps)
                        #            // (20 * epoch_steps)
                        #            + 1
                        #        )
                        #    )
                        if global_step % epoch_steps == 0:
                            self.record(
                                global_step // epoch_steps, "epoch_end", loss=loss
                            )

                    steps += 1
                    if steps % epsilon_update_period == 0:
                        epsilon_min = np.random.choice(
                            epsilon_end, p=epsilon_distribution
                        )

                    epsilon = epsilon_min + (epsilon_start - epsilon_min) * (
                        1.0 - global_step / exploration_frames
                    )
                    epsilon = max(epsilon, epsilon_min)

                    beta = min(1.0, 0.4 + (global_step / (50 * epoch_steps)) * 0.6)

                    if done or steps % async_update_step == 0:
                        # s, a, r, s1, done_t = self.memory.sample(batch_size)
                        loss = self.optimize_model(beta=beta)

                        # self.record(
                        #    steps,
                        #    "loss",
                        #    epsilon=epsilon,
                        #    loss=loss,
                        #    lr=self.optimizer.param_groups[0]["lr"],
                        # )
                        # for g in self.optimizer.param_groups:
                        #    g["lr"] = lr * (1. - global_step / (max_epochs * epoch_steps))

                    if global_step % update_target == 0:
                        self.update_target_model()

            # score = score if score == 500.0 else score + 1

        self.res_queue.put(None)
