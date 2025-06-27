import typing
import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class PrioritizedReplayMemory(object):
    """Fixed-size buffer to store priority, Transition tuples."""

    def __init__(self,
                    buffer_size: int,
                    alpha: float = 0.0,
                    random_state: np.random.RandomState = None) -> None:
        """
        Initialize an PrioritizedReplayMemory object.

        Parameters:
        -----------
        buffer_size (int): maximum size of buffer
        alpha (float): Strength of prioritized sampling. Default to 0.0 (i.e., uniform sampling).
        random_state (np.random.RandomState): random number generator.
        
        """
        self._buffer_size = buffer_size
        self._buffer_length = 0 # current number of prioritized transition tuples in buffer
        self._buffer = np.empty(self._buffer_size, dtype=[("priority", np.float32), ("transition", Transition)])
        self._alpha = alpha
        self._random_state = np.random.RandomState() if random_state is None else random_state
        
    def __len__(self) -> int:
        """Current number of prioritized transition tuple stored in buffer."""
        return self._buffer_length

    @property
    def alpha(self):
        """Strength of prioritized sampling."""
        return self._alpha

    @property
    def buffer_size(self) -> int:
        """Maximum number of prioritized transition tuples stored in buffer."""
        return self._buffer_size

    def push(self, *args) -> None:
        """Add a new transition to memory."""
        priority = 1.0 if self.is_empty() else self._buffer["priority"].max()
        if self.is_full():
            if priority > self._buffer["priority"].min():
                idx = self._buffer["priority"].argmin()
                self._buffer[idx] = (priority, Transition(*args))
            else:
                pass # low priority transitions should not be included in buffer
        else:
            self._buffer[self._buffer_length] = (priority, Transition(*args))
            self._buffer_length += 1

    def is_empty(self) -> bool:
        """True if the buffer is empty; False otherwise."""
        return self._buffer_length == 0

    def is_full(self) -> bool:
        """True if the buffer is full; False otherwise."""
        return self._buffer_length == self._buffer_size

    def sample(self, batch_size: int, beta: float) -> typing.Tuple[np.array, np.array, np.array]:
        """Sample a batch of transitions from memory."""
        # use sampling scheme to determine which transitions to use for learning
        ps = self._buffer[:self._buffer_length]["priority"]
        sampling_probs = ps**self._alpha / np.sum(ps**self._alpha)
        idxs = self._random_state.choice(np.arange(ps.size),
                                            size=batch_size,
                                            replace=True,
                                            p=sampling_probs)
        
        # select the transitions and compute sampling weights
        transitions = self._buffer["transition"][idxs]        
        weights = (self._buffer_length * sampling_probs[idxs])**-beta
        normalized_weights = weights / weights.max()
        
        return idxs, transitions, normalized_weights

    def update_priorities(self, idxs: np.array, priorities: np.array) -> None:
        """Update the priorities associated with particular transitions."""
        self._buffer["priority"][idxs] = priorities