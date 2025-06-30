import gym
from gym.spaces import *
import numpy as np
from ale_py import ALEInterface, ALEState
import warnings
import cv2

addresses = dict(
    invaders_left_count=17,
    # player_score=[102,104], #4 half-bytes, each representing a digit (0000-9999)
    # num_lives=73,
    player_x=28,  # player x coordinate [35,117]
    enemies_x=26,  # invaders x coordinate (left most column) [23,130]
    enemies_y=16,  # invaders y coordinate (from the top)
    ball_y=85,  # player shot y coordinate
    ball_x=87,  # player shot x coordinate
    missile1_y=81,  # invader shot 1 y coordinate
    missile1_x=83,  # invader shot 1 x coordinate
    missile2_y=82,  # invader shot 2 y coordinate
    missile2_x=84,  # invader shot 2 x coordinate
    # spaceship_x=30,
    # obstacles=[43,70], #3 sequences of 9 bytes representing each one the obstacle pixels (0 destroyed, 1 intact),
    # (relevant bits are only 56 out of 72(9 * 8 bits) total, we could save 2 bytes for each obstacle representation
    # this would shrink the possible states for obstacles from 2^216 to 2^168)
    # obstacle0=43,
    # obstacle1=44,
    # obstacle2=45,
    # obstacle3=46,
    # obstacle4=47,
    # obstacle5=48,
    # obstacle6=49,
    # obstacle7=50,
    # obstacle8=51,
    # obstacle9=52,
    # obstacle10=53,
    # obstacle11=54,
    # obstacle12=55,
    # obstacle13=56,
    # obstacle14=57,
    # obstacle15=58,
    # obstacle16=59,
    # obstacle17=60,
    # obstacle18=61,
    # obstacle19=62,
    # obstacle20=63,
    # obstacle21=64,
    # obstacle22=65,
    # obstacle23=66,
    # obstacle24=67,
    # obstacle25=68,
    # obstacle26=69,
    row1=18,
    row2=19,  # 6 bytes, each bit represents an enemy in the corresponding row,
    row3=20,  # (same optimization as above? 6 bytes => 5 bytes or 2^48 to 2^40)
    row4=21,
    row5=22,
    row6=23,
    # bo=24,# 1st bit: player_is_invaded
    # 2nd bit: show obstacles or not
    # 3-4 bit: player-1 and player-2 is playing ?
    # 5-6 bit: ?
    # 7th bit: enemy movement direction     1=left-to-right 0=right-to-left
    # 8th bit: spaceship movement direction 1=left-to-right 0=right-to-left
    # player_is_shooting=77, # 0b11 if player is shooting 0b100 if spaceship is on screen
    # player_is_exploding=42,
)

FRAME_SKIP = 3


class DiscreteSI(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # observation_shape = [255]*(len(addresses.keys())-3) + [255]*9*3 + [255]*6 + [2,2]
        self.observation_space = Box(low=0.0, high=1.0, shape=(72,), dtype=np.float32)

    def observation(self, obs):
        # gray_image = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        # gray_image = gray_image[105:-21,35:119]
        # return gray_image
        # return [obs[i]/255. for i in range(len(obs))]
        res = []
        rows = []
        obstacles_showing = (obs[24] >> 6) & 1
        for key in addresses.keys():
            address = addresses[key]
            if "_x" in key or "_y" in key:
                value = int(obs[address])
                res.append( ((value-23)/(130-23))*2. - 1.)
                #res.append((value / 255) * 2.0 - 1.0)
            elif key == "invaders_left_count":
                value = int(obs[address])
                res.append((1.0 - (value - 1.0) / 36.0))
            elif "row" in key:
                row = obs[address]
                res.extend([(row >> i) & 1 for i in range(6)])
            else:
                res.append(obs[address] / 255)

        # invaders direction
        res.append(-1.0 if (obs[24] >> 1) & 1 == 0 else 1.0)

        # spaceship direction
        # res.append(-1.0 if obs[24] & 1 == 0 else 1.0)

        # res.append(0.0 if obs[77] & 0b11 > 0 else 1.0)  # shooting

        if obstacles_showing:
            for i in range(3):
                r_h = 0
                for row in obs[43 + i * 9 : 52 + i * 9]:
                    r_h = r_h | row
                res.extend([(r_h >> i) & 1 for i in range(8)])
        else:
            res.extend([0] * 24)
        return res


class SIWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        random_starts=True,
        frame_skip=4,
        normalize_reward=True,
        negative_reward=False,
        episodic_life=False,
    ):
        super().__init__(DiscreteSI(env))
        self.player_is_exploding = False
        self.random_starts = random_starts
        self.shooting = False
        self.frame_skip = frame_skip
        self.normalize_reward = normalize_reward
        self.negative_reward = negative_reward
        self.episodic_life = episodic_life

    def reset(self, *args, **kwargs):
        (state, info) = self.env.reset(*args, **kwargs)
        self.env.unwrapped.ale.setRAM(42, 0)

        noops = np.random.randint(1, 80) if self.random_starts else 1

        for i in range(noops):
            (state, _, _, _, info) = self.step(0)

        return state, info

    def step(self, action):
        obs, total_reward, terminated, truncated, info = None, 0.0, False, False, {}
        ale_interface = self.env.unwrapped.ale

        for t in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            hit = ale_interface.getRAM()[42] & 4
            invaded = (ale_interface.getRAM()[24] >> 7) & 1
            if hit or invaded:
                if self.negative_reward:
                    total_reward += -1.0
                if self.episodic_life:
                    terminated = True

            if terminated or truncated:
                break

        total_reward = (
            total_reward
            if not self.normalize_reward
            else max(-1.0, min(1.0, total_reward))
        )
        return obs, total_reward, terminated, truncated, info
<<<<<<< HEAD
=======


def make_env(
    obs_mode="condensed_ram", frame_skip=3, frame_stack=4, render=False, **kwargs
):

    env_kwargs = {"render_mode": "human" if render else "rgb_array"}

    if obs_mode == "image":
        env = gym.make("SpaceInvadersNoFrameskip-v4", **env_kwargs)
        env = ImageObservationWrapper(env)
        env = gym.wrappers.FrameStack(env, frame_stack)
    elif obs_mode == "ram":
        env = gym.make("SpaceInvaders-ramNoFrameskip-v4", **env_kwargs)
        env = RawRAMWrapper(env)
    elif obs_mode == "condensed_ram":
        env = gym.make("SpaceInvaders-ramNoFrameskip-v4", **env_kwargs)
        env = DiscreteSI(env)
    else:
        raise ValueError(f"Unknown obs_mode {obs_mode}")

    env = SIWrapper(env, frame_skip=frame_skip, **kwargs)
    return env
>>>>>>> 0b76251b7216a175cb80cb28bed7b7a530c2e116
