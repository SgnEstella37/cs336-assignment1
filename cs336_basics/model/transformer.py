from turtle import forward
import torch
from cs336_basics.model.attention import MultiHeadSelfAttention
from cs336_basics.model.feedforward import SwiGLU
from cs336_basics.model.layers import RMSNorm


class TransformerBlock(torch.nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    norm1: RMSNorm
    norm2: RMSNorm
    MHA: MultiHeadSelfAttention
    FF: SwiGLU

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        eps1: float = 1e-5,
        eps2: float = 1e-5,
        theta: float | None = None,
        max_seq_len: int = 256,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.norm1 = RMSNorm(d_model, eps1)
        self.norm2 = RMSNorm(d_model, eps2)

        self.MHA = MultiHeadSelfAttention(d_model, num_heads, theta, max_seq_len)

        self.FF = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # MHA
        y = x + self.MHA(self.norm1(x), token_positions)

        # FF
        z = y + self.FF(self.norm2(y))

        return z

