import torch
from cs336_basics.model.layers import Linear


class SwiGLU(torch.nn.Module):
    W_1: Linear
    W_2: Linear
    W_3: Linear
    d_model: int
    d_ff: int

    def __init__(self, d_model: int, d_ff: int | None = None):
        super().__init__()

        # initialize d_ff
        if d_ff is None:
            d_ff = round((8 / 3 * d_model) / 64) * 64

        # initialize W
        self.W_1 = Linear(d_model, d_ff)
        self.W_2 = Linear(d_ff, d_model)
        self.W_3 = Linear(d_model, d_ff)

        self.d_model = d_model
        self.d_ff = d_ff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def silu(x):
            return x * torch.sigmoid(x)

        y_1 = silu(self.W_1.forward(x))
        y_3 = self.W_3.forward(x)
        y = y_1 * y_3
        outcome = self.W_2.forward(y)

        return outcome
