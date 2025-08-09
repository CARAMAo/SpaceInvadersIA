import torch

gamma = 0.99


lr = 1e-5

update_target = 10_000
async_update_step = 4
memory_size = 5_000
batch_size = 32
pre_training = memory_size // 10

# exploration
epsilon_start = 1.0
epsilon_end = [0.1, 0.01, 0.5]
epsilon_distribution = [0.4, 0.3, 0.3]
epsilon_update_period = 100_000  # local steps/frames
exploration_frames = 1_000_000  # global steps/frames


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch_steps = 250_000
max_epochs = 500

double_dqn = False
prioritized_memory = False

obs_mode = 'frame'

sarsa = False
