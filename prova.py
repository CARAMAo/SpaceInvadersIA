import torch.multiprocessing as mp
from dqn import DQN
from torch import optim
import torch
import time
import numpy as np


def child(i):
    for i in range(5):
        print(i, ":", np.random.rand())


if __name__ == "__main__":
    num_processes = 2
    torch.manual_seed(500)

    for i in range(num_processes):
        p = mp.Process(target=child, args=(i,))
        p.start()

    for _ in range(num_processes):
        p.join()
