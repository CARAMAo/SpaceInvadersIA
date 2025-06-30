import torch

gamma = 0.99
batch_size = 128
memory_size = 10_000
pre_training = memory_size
lr = 2.5e-4
epsilon_start = 1.0
epsilon_end = 0.1
exploration_frames = 5_000_000

update_target = 40_000
async_update_step = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch_steps = 250_000
max_epochs = 200
