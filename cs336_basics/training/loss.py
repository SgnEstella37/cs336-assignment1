import torch


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    max_values = logits.max(dim=-1, keepdim=True).values
    logits = logits - max_values
    exp_logits = torch.exp(logits)
    sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)
    log_sum_exp = torch.log(sum_exp)

    correct_logits = torch.gather(logits, dim=-1, index=target.unsqueeze(-1))

    loss = log_sum_exp - correct_logits

    return loss.mean()
