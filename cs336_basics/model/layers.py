import torch


# Linear Module
class Linear(torch.nn.Module):
    in_features: int
    out_features: int
    W: torch.nn.Parameter

    def __init__(
        self,
        in_features,
        out_features,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        W_p = torch.nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        W_var = 2 / (in_features + out_features)
        W_std = W_var**0.5

        # initialize W
        torch.nn.init.trunc_normal_(
            W_p,
            mean=0.0,
            std=W_std,
            a=-3 * W_std,
            b=3 * W_std,
        )
        self.W = W_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.W.T
        return y


# Embedding Module
class Embedding(torch.nn.Module):
    num_embeddings: int
    embedding_dim: int
    embedding_W: torch.nn.Parameter

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        W_e = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        # initialize W
        torch.nn.init.trunc_normal_(
            W_e,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

        self.embedding_W = W_e

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        output = self.embedding_W[token_ids]
        return output


# RMSNorm Module
class RMSNorm(torch.nn.Module):
    d_model: int
    eps: float
    gamma_para: torch.nn.Parameter

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        gp = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

        self.gamma_para = gp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (((x**2).mean(dim=-1, keepdim=True)) + self.eps) ** 0.5
        result = (x / rms) * self.gamma_para.to(torch.float32)

        return result.to(in_dtype)


# Implement Softmax
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_values = torch.max(x, dim=dim, keepdim=True).values
    shifted_x = x - max_values
    exp_s_x = torch.exp(shifted_x)
    sum_exp = torch.sum(exp_s_x, dim=dim, keepdim=True)
    output = exp_s_x / sum_exp
    return output
