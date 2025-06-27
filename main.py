from collections import defaultdict
import gym.wrappers
import torch

from gym_wrappers import *
import numpy as np
from gym.utils import play
import sys
import os
from ale_py import ALEInterface, ALEState
import warnings
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import numpy as np
import cv2
from torchinfo import summary
from dqn import DQN

from gym.wrappers import RecordVideo

print(np.__version__)
print(sys.argv[1])
VERSION = sys.argv[1]

print(torch.cuda.device_count())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", category=DeprecationWarning)


RENDER = True
env = gym.make(
    "SpaceInvaders-ramNoFrameskip-v4", render_mode="human" if RENDER else "rgb_array"
)

env.metadata["render_fps"] = 60


# env = RecordVideo(SIWrapper(DiscreteSI(env),skip_pauses=False),video_folder="./v2",name_prefix=f"model{VERSION}", episode_trigger= lambda _ : True)
env = SIWrapper(env, normalize_reward=False)

# env = gym.make("CartPole-v1",render_mode="human")
frames = 0

address_change_count = [0] * 128

i = 0

print(env.observation_space.shape)
cuda = torch.device("cuda:0")
print(device)

(state, _) = env.reset()
model = DQN(len(state), env.action_space.n).to(device)

checkpoint = torch.load(f"{VERSION}")
model.load_state_dict(checkpoint["model"])

print(summary(model))


scores = []

game_frames = []

max = 0

for i in range(100):  # Gioca per 5 episodi
    (state, _) = env.reset()  # Inizializza lo stato dell'ambiente
    tmp_game_frames = []
    total_reward = 0.0
    done = False  # Inizializza il flag di terminazione dell'episodio
    while not done:  # Continua a giocare finché l'episodio non è terminato
        # Effettua una predizione utilizzando il modello
        with torch.no_grad():
            action = (
                env.action_space.sample()
                if np.random.rand() < 0.05
                else model(torch.tensor([state], device=cuda, dtype=torch.float32))
                .max(1)
                .indices.view(1, 1)
                .item()
            )

        # Esegui l'azione sull'ambiente e ottieni l'osservazione successiva
        state, reward, done, truncated, info = env.step(action)
        tmp_game_frames.extend(env.render())
        # print(state)
        total_reward += reward
        # Visualizza l'ambiente (opzionale)
        # env.render()
    print(f"Episode {i}, reward {total_reward}")
    scores.append(total_reward)
    if total_reward > max:
        max = total_reward
        game_frames = tmp_game_frames

print(f"Mean Score: {np.mean(scores)}({np.std(scores)}) Best Score: {np.max(scores)}")


env.close()

writer = cv2.VideoWriter(
    "outputvideo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 60, game_frames[0].shape[1::-1]
)
for frame in game_frames:
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
writer.release()

# MAX & MIN positions

"""
enemy x: 23 50
enemy y: 16 147
player: 35 117
"""

"""
enemy x: 23 50
enemy y: 17 145
player: 35 117
"""
