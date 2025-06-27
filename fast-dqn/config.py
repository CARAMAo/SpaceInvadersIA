import os
import torch


def getenv(key, default, cast):
    return cast(os.environ.get(key, default))


lr = getenv("LR", 1e-4, float)
update_target = getenv("UPDATE_TARGET", 10_000, int)
memory_size = getenv("MEMORY_SIZE", 100_000, int)
batch_size = getenv("BATCH_SIZE", 128, int)
obs_mode = os.environ.get("OBS_MODE", "condensed_ram")
layer_size = getenv("LAYER_SIZE",512,int)

gamma = 0.99
pre_training = memory_size // 10
epsilon_start = 0.99
epsilon_end = 0.1
eps_decay = (epsilon_start - epsilon_end) / 500_000.0
log_interval = 500
async_update_step = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_steps = 20_000_000
max_evaluation_steps = 5 * 60 * 60
frame_skip = 3 if obs_mode == 'image' else 4

