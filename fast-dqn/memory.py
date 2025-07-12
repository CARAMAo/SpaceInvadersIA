import random
from collections import namedtuple, deque
import numpy as np

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return zip(*self.memory)
        else:
            return zip(*random.sample(self.memory, batch_size))

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
