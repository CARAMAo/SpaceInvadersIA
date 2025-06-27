import torch
import torch.multiprocessing as mp
import datetime
from torch import nn


class SharedModel(nn.Module):
    def __init__(self):
        super(SharedModel, self).__init__()
        # Definizione di un tensore 2x2 come parametro del modello
        self.shared_tensor = nn.Parameter(torch.rand((2, 2), device="cuda"))

    def forward(self):
        pass  # Non necessario per questo esempio


def worker_process(shared_model, process_id):
    """
    Funzione eseguita da ciascun processo.
    Ogni processo aggiorna il tensore condiviso con valori pari al proprio process_id + 1.
    """
    print(f"[{datetime.datetime.now()}] Worker {process_id} started.")
    print(
        f"[{datetime.datetime.now()}] Worker {process_id} initial tensor:\n{shared_model.shared_tensor.data}"
    )

    # Creazione di un nuovo stato con valori pari al process_id + 1
    new_state = {
        "shared_tensor": torch.full_like(
            shared_model.shared_tensor, fill_value=process_id + 1
        )
    }
    shared_model.load_state_dict(new_state)  # Aggiorna il tensore condiviso

    print(
        f"[{datetime.datetime.now()}] Worker {process_id} updated tensor:\n{shared_model.shared_tensor.data}"
    )


def main():
    # Creazione del modello condiviso e allocazione su CUDA
    shared_model = SharedModel().to("cuda")
    shared_model.shared_tensor.data.share_memory_()  # Condivisione del tensore

    print(
        f"[{datetime.datetime.now()}] Main process initial tensor:\n{shared_model.shared_tensor.data}"
    )

    # Creazione di due processi
    processes = []
    for process_id in range(2):
        p = mp.Process(target=worker_process, args=(shared_model, process_id))
        processes.append(p)
        p.start()

    # Attesa della terminazione dei processi
    for p in processes:
        p.join()

    print(
        f"[{datetime.datetime.now()}] Main process final tensor:\n{shared_model.shared_tensor.data}"
    )


if __name__ == "__main__":
    mp.set_start_method("spawn")  # Metodo di avvio per torch.multiprocessing
    main()
