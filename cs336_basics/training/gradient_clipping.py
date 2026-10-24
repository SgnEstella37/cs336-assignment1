import torch


def gradient_clipping(parameters: list[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    total_squared = torch.tensor(0.0)
    for p in parameters:
        if p.grad == None:
            continue
        total_squared += torch.sum(p.grad**2)
    total_norm = torch.sqrt(total_squared)
    total_norm_float = total_norm.item()
    if total_norm_float <= max_l2_norm:
        return
    scale = max_l2_norm / (total_norm + eps)
    for p in parameters:
        if p.grad is None:
            continue

        p.grad.mul_(scale)
