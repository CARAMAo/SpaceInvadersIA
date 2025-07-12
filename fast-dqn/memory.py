import random
from collections import namedtuple, deque
import numpy as np
from torchrl.data import ReplayBuffer,LazyMemmapStorage
from tensordict import TensorDict
from config import batch_size,device
import torch

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


scratch_dir = 'D:/dump/'

def transform(x):
    return {k:torch.cat(list(v.to(device))) for k,v in x.items()}

class ReplayMemory(object):

    def __init__(self, capacity,path=''):
        self.memory = ReplayBuffer(storage=LazyMemmapStorage(capacity,scratch_dir=scratch_dir+path),batch_size=batch_size,pin_memory=False,transform=transform)

    def push(self, *args):
        self.memory.add(Transition(*args)._asdict())

    def sample(self, batch_size=None):
        sample = self.memory.sample(batch_size)
        return sample.values()

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = []
        self.pos = 0

    def push(self, *args):
        """Save a transition with max priority."""
        max_prio = max(self.priorities, default=1.0)
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
            self.priorities.append(max_prio)
        else:
            self.memory[self.pos] = Transition(*args)
            self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == 0:
            return [], [], [], [], [], []

        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        # Importance-sampling weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        return (*batch, indices, weights)

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)
