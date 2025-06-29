import os
import torch


def getenv(key, default, cast):
    return cast(os.environ.get(key, default))


lr = getenv("LR", 1e-3, float)
update_target = getenv("UPDATE_TARGET", 1_000, int)
async_update_step = 4
obs_mode = os.environ.get("OBS_MODE", "condensed_ram")
layer_size = getenv("LAYER_SIZE", 128, int)

gamma = 0.99
epsilon_start = 0.9
epsilon_end = 0.1

device = torch.device("cpu")

frame_skip = 3 if obs_mode == "image" else 4

steps_per_epoch = 250_000
training_epochs = 100
