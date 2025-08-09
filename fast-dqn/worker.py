import gym
from gym.wrappers.atari_preprocessing import AtariPreprocessing
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers import TransformReward
import torch
import torch.multiprocessing as mp
import numpy as np
from model import QNet
from memory import ReplayMemory, Transition, PrioritizedReplayMemory
from si_wrappers import *
from copy import deepcopy
from config import *
from collections import deque

def clip_reward(x):
    return min(max(x,0.),1.)

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
            self.env = AtariPreprocessing(self.env,terminal_on_life_loss=True)
            self.env = FrameStack(self.env, 4)
            self.env = TransformReward(self.env,clip_reward)
        else:
            self.env = gym.make("SpaceInvaders-ramNoFrameskip-v4")
            self.env = SIWrapper(
                self.env, frame_skip=4, episodic_life=True
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
        self.memory_size = memory_size

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
            s = torch.cat(list(s)).to(device)
            a = torch.cat(list(a)).to(device)
            r = torch.cat(list(r)).to(device)
            s1 = torch.cat(list(s1)).to(device)
            done = torch.cat(list(done)).to(device)
            weights = torch.tensor(weights,device=device).unsqueeze(1)
        else:
            s, a, r, s1, done = self.memory.sample(batch_size)
            s = torch.cat(list(s)).to(device)
            a = torch.cat(list(a)).to(device)
            r = torch.cat(list(r)).to(device)
            s1 = torch.cat(list(s1)).to(device)
            done = torch.cat(list(done)).to(device)
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
        del s,a,r,s1,done
        return loss_value

    def optimize_model_sarsa(self, transitions_batch):
        # Unpack batch: list of (s, a, r, s', a', done)
        states, actions, rewards, next_states, next_actions, dones = zip(*transitions_batch)

        # Convert lists to tensors
        states = torch.cat(states).to(device)
        actions = torch.cat(actions).to(device)
        rewards = torch.cat(rewards).to(device)
        next_states = torch.cat(next_states).to(device)
        next_actions = torch.cat(next_actions).to(device)
        dones = torch.cat(dones).to(device)

        # Q(s, a)
        q_pred = self.online_net(states).gather(1, actions)

        with torch.no_grad():
            q_next = self.online_net(next_states).gather(1, next_actions)

        target = rewards + gamma * q_next.squeeze(1) * dones

        criterion = torch.nn.MSELoss()
        loss = criterion(q_pred.squeeze(1), target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()


    def get_action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return torch.tensor(
                [[self.env.action_space.sample()]], device=device, dtype=torch.long
            )
        else:
            with torch.no_grad():
                return self.online_net(state.to(device)).max(1).indices.view(1, 1)

    def run(self):
        if prioritized_memory:
            self.memory = PrioritizedReplayMemory(self.memory_size)
        else:
            self.memory = ReplayMemory(self.memory_size)
        epsilon = epsilon_start
        epsilon_min = np.random.choice(epsilon_end, p=epsilon_distribution)
        steps = 0
        global_step = 0
        loss = 0
        self.online_net.init(seed=self.init_seed)
        self.target_net.init(seed=self.init_seed)
        self.init_barrier.wait()
        curr_epoch = 0
        sarsa_batch = []
        while global_step // epoch_steps < max_epochs:

            done = False

            score = 0
            (state, _) = self.env.reset()

            state = torch.Tensor(np.array(state)).unsqueeze(0)

            while not done:

                action = self.get_action(state, epsilon)
                next_state, reward, term, trunc, _ = self.env.step(action.item())
                # self.frame_stack.append(next_state)

                done = term or trunc

                next_state = torch.Tensor(np.array(next_state)).unsqueeze(0)
                reward = torch.Tensor([reward])
                done_t = torch.Tensor([not done])
                next_action = self.get_action(next_state, epsilon)

                if sarsa:
                    sarsa_batch.append((state, action, reward, next_state, next_action, done_t))
                else:
                    self.memory.push(state, action, reward, next_state, done_t)
                

                score += reward

                state = next_state

                if sarsa or len(self.memory) >= pre_training:
                    with self.global_step.get_lock():
                        self.global_step.value += 1
                        global_step = self.global_step.value
                        if curr_epoch != global_step // epoch_steps:
                            curr_epoch = global_step //epoch_steps

                        
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


                    if done or steps % async_update_step == 0:
                        # s, a, r, s1, done_t = self.memory.sample(batch_size)
                        if sarsa:
                            if sarsa_batch:
                                loss = self.optimize_model_sarsa(sarsa_batch)
                                sarsa_batch.clear()
                        else:
                            loss = self.optimize_model()



                        self.record(
                            steps,
                            "loss",
                            epsilon=epsilon,
                            loss=loss,
                            lr=self.optimizer.param_groups[0]["lr"],
                        )


                    if global_step % update_target == 0:
                        self.update_target_model()


        self.res_queue.put(None)
