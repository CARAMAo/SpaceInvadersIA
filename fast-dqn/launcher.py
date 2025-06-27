import os
import subprocess
import itertools

# ---------- IPERPARAMETRI ----------
# Fase 1
lr_list = [2.5e-4]
update_target_list = [10000]
default_memory_size = 100_000
default_batch_size = 128
layer_size_list = [128,256,512]

# Fase 2
memory_sizes = [50_000, 100_000]
batch_sizes = [32, 64, 128]

# ---------- OSSERVAZIONE ----------
obs_mode = "condensed_ram"  # o "image", "condensed_ram"


# ---------- ESPERIMENTI ----------
def run_experiment(env_vars):
    print(f"Launching experiment: {env_vars}")
    subprocess.run(["python", "train.py"], env=env_vars)


def run_phase_1():
    print("▶ Phase 1")
    for lr, layer_size in itertools.product(lr_list, layer_size_list):
        env_vars = os.environ.copy()
        env_vars["LR"] = str(lr)
        env_vars["UPDATE_TARGET"] = str(update_target_list[0])
        env_vars["MEMORY_SIZE"] = str(default_memory_size)
        env_vars["BATCH_SIZE"] = str(default_batch_size)
        env_vars["OBS_MODE"] = obs_mode
        env_vars["LAYER_SIZE"] = str(layer_size)
        run_experiment(env_vars)


def run_phase_2(best_lr, best_update_target):
    print("▶ Phase 2")
    for memory_size, batch_size in itertools.product(memory_sizes, batch_sizes):
        env_vars = os.environ.copy()
        env_vars["LR"] = str(best_lr)
        env_vars["UPDATE_TARGET"] = str(best_update_target)
        env_vars["MEMORY_SIZE"] = str(memory_size)
        env_vars["BATCH_SIZE"] = str(batch_size)
        env_vars["OBS_MODE"] = obs_mode
        run_experiment(env_vars)


# ---------- MAIN ----------
if __name__ == "__main__":
    run_phase_1()

    # Aggiorna con i valori migliori trovati dopo Fase 1
    #BEST_LR = 1e-4
    #BEST_UPDATE_TARGET = 10000
    #run_phase_2(BEST_LR, BEST_UPDATE_TARGET)
