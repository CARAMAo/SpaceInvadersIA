import gym
from gym.utils import play
import gym_wrappers
from ale_py import ALEInterface

env = gym.make("SpaceInvaders-ram-v4",render_mode="rgb_array")

env = env
env.metadata['render_fps'] = 240
env.reset()
old = None
counter = [0]*128
l = []

def f(obs_t,obs_tp,action,reward,terminated,truncated,info):
    global env

    ale: ALEInterface = env.unwrapped.ale

    for i in (81,82,83,84):
        ale.setRAM(i,255)
    # global env
    # if terminated:
    #     exit()
    
    # ale: ALEInterface = env.unwrapped.ale

    # b = ale.getRAM()
    # hit = b[42] & 4
    # ale.setRAM(42,0)
    # if hit:
    #     ale.setRAM(28,35)
    #     ale.setRAM(73,info['lives']-1)
    
    
       
play.play(env,zoom=2.,callback=f)

