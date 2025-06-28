import gym
from gym.utils import play



def callback(*args,**kwargs):
	print(args[0][81:85])


env = gym.make("SpaceInvaders-ramNoFrameskip-v4",render_mode="rgb_array")

play.play(env,zoom=2,fps=10,callback=callback)