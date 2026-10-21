
import torch
from cs336_basics.model.attention import MultiHeadSelfAttention
from cs336_basics.model.feedforward import SwiGLU
from cs336_basics.model.layers import RMSNorm, Linear, Embedding


# implement Block
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

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        # MHA
        y = x + self.MHA(self.norm1(x), token_positions)

        # FF
        z = y + self.FF(self.norm2(y))

        return z


# fulfill the whole TransformerLM
class TransformerLM(torch.nn.Module):
    d_model: int
    vocab_size: int
    context_length: int
    num_layers: int
    blocks: torch.nn.ModuleList
    final_norm: RMSNorm
    LM_Head: Linear
    embedding:Embedding

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = Embedding(vocab_size, d_model)
        
        self.blocks = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=rope_theta,
                max_seq_len=context_length,
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)
        
        self.LM_Head = Linear(d_model, vocab_size)
        
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        embedding_T=self.embedding(x)
        
        for block in self.blocks:
            embedding_T= block(embedding_T)
            
        embedding_T= self.final_norm.forward(embedding_T)
        
        output=self.LM_Head(embedding_T)

        return output