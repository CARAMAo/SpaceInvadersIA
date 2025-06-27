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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dqn import DQN
from prm import Transition,PrioritizedReplayMemory


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


print(matplotlib.get_backend())
plt.ion()
# if GPU is to be used
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print("Using Cuda" if torch.cuda.is_available() else "CPU" )

warnings.filterwarnings("ignore", category=DeprecationWarning)


RENDER=False

env = gym.make("SpaceInvaders-ramNoFrameskip-v4",render_mode= "human" if RENDER else "rgb_array")

env = SIWrapper(DiscreteSI(env),skip_pauses=True)

# env = gym.make("CartPole-v1")

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 500_000
SOFT_UPDATE= False
TAU = 0.005
H_TAU = 10_000
LR =  0.00025

USE_PRM = True
#Prioritized Replay Memory Parameters
#PRM_ALFA 0. uniform sampling - 1. priority sampling
#BETA_SCHEDULE lambda function computing BETA (sampling bias correction: 0. none - 1. full correction) based on the episode number 
PRM_ALFA = .5 if USE_PRM else .0
BETA_SCHEDULE = (lambda i_episode: .5) if USE_PRM else (lambda i_episode: 0.)

DOUBLE_DQN=False

env.metadata["render_fps"] = 120
# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
(state,_) = env.reset()
n_observations = len(state)
print(n_observations)
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = PrioritizedReplayMemory(1000000,PRM_ALFA)


steps_done = 0
eps_threshold=0.

@torch.compile
def select_action(state,evaluating=False):
    global steps_done,eps_threshold
    sample = random.random()
    if evaluating:
        if sample > 0.05:
          with torch.no_grad():
              # t.max(1) will return the largest column value of each row.
              # second column on max result is index of where max element was
              # found, so we pick action with the larger expected reward.
              return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    else:
        # eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        #     math.exp(-1. * steps_done / EPS_DECAY)
        r = steps_done / EPS_DECAY

        eps_threshold = EPS_END + (EPS_START - EPS_END)*max(0.,(1.-r))
        steps_done+=1
    
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []
episode_rewards = []
episode_max_q = []
episode_eps = []

total_q = 0.
total_loss = 0.
episode_loss = []


def plot_durations(show_result=False):
    plt.figure(1)

    durations_t = torch.tensor(episode_loss, dtype=torch.float)
    # rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    q_t = torch.tensor(episode_max_q,dtype=torch.float)
    eps_t = torch.tensor(episode_eps,dtype=torch.float)

    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')

    plt.subplot(411)
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())


    plt.subplot(412)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    min_rewards,max_rewards,avg_rewards = zip(*episode_rewards)
    plt.plot(min_rewards)
    plt.plot(max_rewards)
    plt.plot(avg_rewards)


    plt.subplot(413)
    plt.xlabel('Episode')
    plt.ylabel('Q-Max')
    plt.plot(q_t.numpy())
    # Take 100 episode averages and plot them too
    if len(q_t) >= 100:
        means = q_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.subplot(414)
    plt.xlabel('Episode')
    plt.ylabel('Eps')
    plt.plot(eps_t.numpy())
    # Take 100 episode averages and plot them too
    if len(eps_t) >= 100:
        means = eps_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.01)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

@torch.compile
def optimize_model():
    global total_loss,total_q
    if len(memory) < BATCH_SIZE:
        return
    
    beta = BETA_SCHEDULE(i_episode)

    idx,transitions,sampling_weights = memory.sample(BATCH_SIZE,beta)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    total_q += torch.max(state_action_values).item()
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        if DOUBLE_DQN:
            #Double DQN:
            q_actions = policy_net(non_final_next_states).argmax(1,keepdim=True)
            n = target_net(non_final_next_states).gather(1, q_actions)
            next_state_values[non_final_mask] = n.squeeze()
        else:
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # # Compute Huber loss
    # criterion = nn.SmoothL1Loss()
    # # criterion = nn.MSELoss()

    # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    '''PRIORITIZED EXPERIENCE START'''
    deltas = expected_state_action_values.unsqueeze(1) - state_action_values

    priorities = (deltas.abs()
                        .cpu()
                        .detach()
                        .numpy()
                        .flatten())
    
    memory.update_priorities(idx, priorities + 1e-6) # priorities must be positive!
    
    # compute the mean squared loss
    _sampling_weights = (torch.tensor(sampling_weights,device=device)
                                .view((-1, 1)))
    loss = torch.mean((deltas * _sampling_weights)**2)
    '''PRIORITIZED EXPERIENCE END'''

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    total_loss += loss.item()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 10)
    optimizer.step()
   
if torch.cuda.is_available():
    num_episodes = 5
else:
    num_episodes = 5

steps_per_episode = 5_000

total_steps = num_episodes * steps_per_episode
i_episode = 0

start = datetime.datetime.now()
preprocessing_length = 0
preprocessing_i = 0
while preprocessing_i < preprocessing_length:
    (state, _) = env.reset()
    state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
    for t in count():
        action = torch.tensor([[env.action_space.sample()]],device=device,dtype=torch.long)
        r = env.step(action.item())
        observation, reward, terminated, truncated, info = r
        reward = torch.tensor([reward], device=device)
        preprocessing_i+=1
        done = terminated or preprocessing_i >= preprocessing_length
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0)
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state
        if done:
            print(f"Pre-training Episode ended {preprocessing_i}/{preprocessing_length}")
            break
        
       
def evaluate_agent():
    scores = []
    for i in range(5):  # Gioca per 5 episodi
        (state,_) = env.reset()  # Inizializza lo stato dell'ambiente
        total_reward = 0.
        done = False  # Inizializza il flag di terminazione dell'episodio
        while not done:  # Continua a giocare finché l'episodio non è terminato
            state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
            action = select_action(state,evaluating=True)
            # Esegui l'azione sull'ambiente e ottieni l'osservazione successiva
            state, reward, terminated, truncated,info  = env.step(action)
            total_reward+=reward
            done = terminated or truncated
        scores.append(total_reward)
    return (np.min(scores),np.max(scores),np.mean(scores))


while i_episode < num_episodes:
    i = 0
    total_reward = 0.
    total_loss = 0.
    total_q = 0.
    games_completed = 0
    terminated = False
    while i < steps_per_episode:
        (state,_) = env.reset()
        state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
        done = False
        while not done:
            action = select_action(state)
            r = env.step(action.item())
            observation, reward, terminated, truncated, info = r

            total_reward += reward

            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0)
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            
            # Move to the next state
            state = next_state

            
            # Perform one step of the optimization (on the policy network)
            optimize_model()

            if SOFT_UPDATE:
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)
            # Hard update of the target network's weights
            # θ′ ← θ
            elif (steps_done+1) % H_TAU == 0:
                target_net.load_state_dict(policy_net.state_dict())   
            i+=1
        games_completed+=1

    episode_eps.append(eps_threshold)
    r = evaluate_agent()
    episode_rewards.append( r )
    episode_loss.append( total_loss/i )
    episode_max_q.append(total_q/i)
    
    print(f"Episode {i_episode+1}: Frame Played {i} Loss {total_loss/i} Avg. Reward {r[-1]} Eps. {eps_threshold} Steps Done {steps_done+1}/{total_steps}")
    plot_durations()
    if (i_episode+1)%10 == 0:
        torch.save( 
                    # policy_net.state_dict(),
                    {
                    'optimizer':optimizer.state_dict(),
                    'model':policy_net.state_dict(),
                    'episode':i_episode,
                    'steps_done':steps_done,
                    },
                    f"model{i_episode+1}")
    i_episode+=1
        

print('Complete')
torch.save({'optimizer':optimizer.state_dict(),'model':policy_net.state_dict(),'steps_done':steps_done},f"model-final")
plot_durations(show_result=True)
plt.ioff()
plt.show()
print(datetime.datetime.now() - start)


# while steps_done < total_steps:
#     # Initialize the environment and get its state
#     (state,_) = env.reset()
#     state = torch.tensor(state, dtype=torch.float, device=device).unsqueeze(0)
#     total_reward = 0.
#     total_loss = 0.
#     total_q = 0.
#     games_completed = 0.
#     # if i_episode%TAU == 0:
#     #   target_net.load_state_dict(policy_net.state_dict())

#     for t in count():

#         action = select_action(state)
#         r = env.step(action.item())
#         observation, reward, terminated, truncated, info = r

#         total_reward += reward

#         reward = torch.tensor([reward], device=device)
#         done = terminated or truncated
#         if terminated:
#             next_state = None
#         else:
#             next_state = torch.tensor(observation, dtype=torch.float, device=device).unsqueeze(0)
#         # Store the transition in memory
#         memory.push(state, action, next_state, reward)
        
#         # Move to the next state
#         state = next_state

    
#         # Perform one step of the optimization (on the policy network)
#         optimize_model()

#         if SOFT_UPDATE:
#         # Soft update of the target network's weights
#         # θ′ ← τ θ + (1 −τ )θ′
#             target_net_state_dict = target_net.state_dict()
#             policy_net_state_dict = policy_net.state_dict()
#             for key in policy_net_state_dict:
#                 target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
#             target_net.load_state_dict(target_net_state_dict)
#         # Hard update of the target network's weights
#         # θ′ ← θ
#         elif (steps_done+1) % H_TAU == 0:
#             target_net.load_state_dict(policy_net.state_dict())

#         if done:
#             episode_durations.append(t + 1)
#             episode_rewards.append(total_reward)
#             episode_loss.append( total_loss/(t+1) )
#             episode_max_q.append(total_q/(t+1))
#             episode_eps.append(eps_threshold)
#             print(f"Episode {i_episode+1}: Loss {total_loss/(t+1)} Total Reward {total_reward} Eps. {eps_threshold} Steps Done {steps_done+1}/{total_steps}")
#             plot_durations()
#             if (i_episode+1)%100 == 0:
#                 torch.save({'optimizer':optimizer,'model':policy_net,'steps_done':steps_done,'eps':eps_threshold},f"model{i_episode+1}")
#             i_episode+=1
#             break

# print('Complete')
# torch.save({'optimizer':optimizer,'model':policy_net,'steps_done':steps_done,'eps':eps_threshold},f"model-final")
# plot_durations(show_result=True)
# plt.ioff()
# plt.show()
# print(datetime.datetime.now() - start)