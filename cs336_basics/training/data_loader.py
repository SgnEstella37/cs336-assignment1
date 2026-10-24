import numpy as np
import torch


def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    D = len(x)
    m = context_length
    start_indices = np.random.randint(0, D - m, size=batch_size)

    inputs = []
    targets = []

    for i in start_indices:
        inputs.append(x[i : i + context_length])
        targets.append(x[i + 1 : i + 1 + context_length])

    inputs = np.stack(inputs)
    targets = np.stack(targets)

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs, targets
