import os
import torch


def getenv(key, default, cast):
    return cast(os.environ.get(key, default))


lr = getenv("LR", 2.5e-4, float)
update_target = getenv("UPDATE_TARGET", 5_000, int)
memory_size = getenv("MEMORY_SIZE", 100_000, int)
batch_size = getenv("BATCH_SIZE", 32, int)
obs_mode = os.environ.get("OBS_MODE", "condensed_ram")
layer_size = getenv("LAYER_SIZE", 128, int)

gamma = 0.99
pre_training = 10_000
epsilon_start = 0.99
epsilon_end = 0.1
eps_decay = (epsilon_start - epsilon_end) / 1_000_000.0
log_interval = 500
async_update_step = 8
device = torch.device("cuda")
max_steps = 20_000_000

frame_skip = 3 if obs_mode == "image" else 4
max_evaluation_steps = 100_000 // frame_skip

steps_per_epoch = 250_000
training_epochs = 100
