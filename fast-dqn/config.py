
import torch

gamma = 0.99
batch_size = 128
memory_size = 500_000
pre_training = memory_size//10 # 10% of memory_size
lr = 2.5e-5
epsilon_start = .9
epsilon_end = .1
eps_decay =  (epsilon_start-epsilon_end)/500_000.

log_interval = 100
update_target = 10_000
async_update_step = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_steps = 10_000_000
