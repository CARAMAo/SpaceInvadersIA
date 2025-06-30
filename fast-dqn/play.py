import sys
import torch
import gym
from model import QNet
from gym.utils import play
from si_wrappers import SIWrapper,DiscreteSI


if __name__ == "__main__":
    checkpoint = torch.load(sys.argv[1], map_location="cuda")
    env = gym.make("SpaceInvaders-ramNoFrameskip-v4", render_mode="human")
    env.metadata['render_fps'] = 240
    env = SIWrapper(env,frame_skip=4,render=True,random_starts=False)
    
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.n

    net = QNet(num_inputs, num_outputs)
    net.load_state_dict(checkpoint["model"])
    net.eval()

    epsilon = 0.05

    for episode in range(5):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            if torch.rand(1).item() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = net(state_tensor).max(1).indices.item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            state = next_state
            done = terminated or truncated
        print(f"Episode {episode+1}: score={total_reward}")
