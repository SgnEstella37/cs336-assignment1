import torch
from cs336_basics.model.layers import softmax, Linear
from tests.conftest import d_model
from einops import rearrange


# implement RoPE
class RotaryPositionalEmbedding(torch.nn.Module):
    theta: float
    max_seq_len: int
    d_k: int
    cos_cache: torch.Tensor
    sin_cache: torch.Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"

        # store the cos and sin
        positions = torch.arange(max_seq_len, device=device)
        freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device) / d_k))
        angles = positions[:, None] * freq[None, :]
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_k, "Last dimension of x must equal d_k"
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        output = torch.stack((rot_even, rot_odd), dim=-1)
        output = output.flatten(start_dim=-2)

        return output


# implement scaled dot product attention
def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    # mask's elements are True or False
    scores = Q @ K.transpose(-2, -1)
    scores = scores / (Q.shape[-1] ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    return softmax(scores, dim=-1) @ V


# implement multi-head self attention
class MultiHeadSelfAttention(torch.nn.Module):
    d_model: int
    d_k: int
    num_heads: int
    theta: float | None = None
    rope: RotaryPositionalEmbedding
    T: int
    max_seq_len: int
    W_Q: torch.Tensor
    W_K: torch.Tensor
    W_V: torch.Tensor
    W_O: torch.Tensor

    def __init__(self, d_model: int, num_heads: int, theta: float| None = None, max_seq_len: int=256):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.rope = None

        if theta is not None:
            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
            )
        # Initialize projection matrices
        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # multihead
        Q = rearrange(Q, "b t (h d) -> b h t d", h=self.num_heads)
        K = rearrange(K, "b t (h d) -> b h t d", h=self.num_heads)
        V = rearrange(V, "b t (h d) -> b h t d", h=self.num_heads)

        # RoPE
        T = x.shape[-2]
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(T, device=x.device)

            assert T <= self.max_seq_len, "Sequence length exceeds max_seq_len"
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # causal masking
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))

        # attention
        outcomes = scaled_dot_product_attention(Q, K, V, mask)
        outcomes = rearrange(outcomes, "b h t d -> b t (h d)")

        outcomes = self.W_O(outcomes)
        return outcomes
