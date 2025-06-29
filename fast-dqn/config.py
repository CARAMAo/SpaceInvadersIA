import os
import torch


def getenv(key, default, cast):
    return cast(os.environ.get(key, default))


lr = getenv("LR", 1e-4, float)
update_target = getenv("UPDATE_TARGET", 10_000, int)
async_update_step = 4
memory_size = getenv("MEMORY_SIZE", async_update_step, int)
obs_mode = os.environ.get("OBS_MODE", "condensed_ram")
layer_size = getenv("LAYER_SIZE", 128, int)

gamma = 0.99
pre_training = 10_000
epsilon_start = 0.9
epsilon_end = 0.1
log_interval = 500

device = torch.device("cpu")
max_steps = 20_000_000

frame_skip = 3 if obs_mode == "image" else 4
max_evaluation_steps = 100_000 // frame_skip

steps_per_epoch = 250_000
training_epochs = 100
