import os
import sys
from time import sleep
import torch
import gym
from model import CNNQNet,QNet
from gym.utils import play
from si_wrappers import SIWrapper, DiscreteSI
import cv2
import numpy as np

device = torch.device('cuda')

def play_game(env, net, num_games=1000):
    scores = []
    best_frames = []
    max_score = 0
    for i in range(num_games):
        state, _ = env.reset()
        # frame_stack = deque([state]*4,maxlen=4)
        total_reward = 0.
        steps = 0
        state = torch.Tensor(np.array(state)).to(device).unsqueeze(0)
        done = False
        while not done:
            action = (
                env.action_space.sample()
                if np.random.rand() < 1.0
                else net(state).max(1).indices.view(1, 1).item()
            )
            next_state, reward, term, trunc, _ = env.step(action)
            # frame_stack.append(next_state)
            state = torch.Tensor(np.array(next_state)).to(device).unsqueeze(0)
            total_reward += reward
            steps += 1
            done = term or trunc
        print(f"Game {i} completed, score: {total_reward}")
        if total_reward > max_score:
            max_score = total_reward
            best_frames = env.render()
        scores.append(total_reward)
    return (np.mean(scores),np.max(scores),best_frames)

if __name__ == "__main__":
    checkpoint = torch.load(sys.argv[1], map_location="cuda")
    obs_mode = "frame"

    if obs_mode == "frame":
        env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="rgb_array")
        env.metadata['render_fps'] = 120
        env = gym.wrappers.RenderCollection(env,pop_frames=False)
        env = gym.wrappers.atari_preprocessing.AtariPreprocessing(env)
        env = gym.wrappers.frame_stack.FrameStack(env, 4)
        
        net = CNNQNet()
    else:
        env = gym.make("SpaceInvaders-ramNoFrameskip-v4", render_mode="rgb_array")
        env = SIWrapper(
            env,
            frame_skip=4,
            random_starts=False,
            normalize_reward=False,
        )
        env = gym.wrappers.RenderCollection(env,pop_frames=False)
        net = QNet(71,6)

    num_inputs = env.observation_space.shape
    num_outputs = env.action_space.n

    print(num_inputs)
    (state, _) = env.reset()
    state_dict = checkpoint['model']
    
    net.load_state_dict(state_dict)
    net.eval()
    net.to(torch.device("cuda"))

    mean,best,frames = play_game(env,net)
    print("avg score:",mean,"best score:",best)
    if frames:
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter("best_episode.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 120, (width, height))
        for frame in frames:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Converti da RGB a BGR per OpenCV
            out.write(bgr)
        out.release()
