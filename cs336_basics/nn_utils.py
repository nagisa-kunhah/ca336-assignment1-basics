import torch
from torch import Tensor,nn
from typing import Optional, Dict
from jaxtyping import Bool, Float, Int
import math
from cs336_basics.position_encoder import RoPE

position_encoders = {
    "RoPE": RoPE
}

class Linear(nn.Module):
    input_dim: int
    output_dim: int
    weight: nn.Parameter

    def __init__(self, input_dim: int, output_dim: int, device=None, dtype=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(
            torch.empty(
                size=(output_dim, input_dim),
                device = device,
                dtype=dtype
            )
        )
        std = math.sqrt(2 / (input_dim + output_dim))
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        )

    
    def forward(self, input: Float[Tensor, " ... d_in"]):
        return input @ self.weight.T


class Embedding(nn.Module):
    vocal_size: int
    d_model: int
    weight: nn.Parameter

    def __init__(self, vocal_size: int, d_model: int, device=None, dtype=None):
        super().__init__()
        self.vocal_size = vocal_size
        self.d_model = d_model
        self.weight = nn.Parameter(
            torch.empty(
                size=(vocal_size, d_model),
                device = device,
                dtype=dtype
            )
        )

    def forward(self, token_ids: Int[Tensor, " ..."]):
        return self.weight[token_ids]


class SiLu(nn.Module):
    def __init__(self):
        super().__init__()

    def sigmoid(self, x: Float[Tensor, " ... d_model"]):
        return 1/(1 + torch.exp(-x))
    
    def forward(self, x: Float[Tensor, " ..."]):
        return x*self.sigmoid(x)

class SwigGLU(nn.Module):
    d_model: int
    d_ff: int
    w1: Linear
    w2: Linear
    w3: Linear
    silu: nn.Module
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(input_dim=d_model, output_dim=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(input_dim=d_ff, output_dim=d_model, device=device, dtype=dtype)
        self.w3 = Linear(input_dim=d_model, output_dim=d_ff, device=device, dtype=dtype)
        self.silu = SiLu()


    def forward(self, x: Float[Tensor, " ... d_model"]):
        h1 = self.silu(self.w1(x))
        h3 = self.w3(x)
        return self.w2((h1*h3))

class Softmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, " ..."], dim: int):
        zi = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
        mu = torch.sum(torch.exp(x - torch.max(x, dim=dim, keepdim=True).values), dim=dim, keepdim=True)
        return zi/mu

class ScaledDotProductAttention(nn.Module):
    softmax: nn.Module
    def __init__(self):
        super().__init__()
        self.softmax = Softmax()
    
    def forward(self, 
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... keys d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    )->Float[Tensor, " ... queries d_v"]:
        scale = math.sqrt(Q.shape[-1]) 
        attention_score = Q@K.transpose(-2,-1)/scale # (..., queries, d_k)@(..., d_k, keys) =  (...,  queries, keys)
        if mask is not None:
            attention_score = attention_score.masked_fill(~mask, float('-inf'))
        weights = self.softmax(attention_score, -1)
        return weights@V #(...,  queries, keys) @ (..., keys, d_v) = (..., queries, d_v)

class MultiheadSelfAttention(nn.Module):
    softmax: nn.Module
    q_proj: nn.Module
    k_proj: nn.Module
    v_proj: nn.Module
    output_proj: nn.Module
    n_head: int
    d_model: int
    position_encoder: nn.Module | None
    device: torch.device
    dtype: torch.dtype
    def __init__(self, d_model: int, num_heads: int, position_encoder: nn.Module|None = None, device=None, dtype=None):
        super().__init__()
        self.softmax = Softmax()
        self.d_model = d_model
        self.n_head = num_heads
        self.position_encoder = position_encoder
        self.q_proj = Linear(input_dim=d_model, output_dim=d_model, device=device, dtype=dtype)
        self.k_proj = Linear(input_dim=d_model, output_dim=d_model, device=device, dtype=dtype)
        self.v_proj = Linear(input_dim=d_model, output_dim=d_model, device=device, dtype=dtype)
        self.output_proj = Linear(input_dim=d_model, output_dim=d_model, device=device, dtype=dtype)
    
    def forward(self,
        in_features: Float[Tensor, " ... sequence_length d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ):
        sequence_length = in_features.shape[-2]
        d_model = in_features.shape[-1]
        num_heads = self.n_head
        d_head = d_model // num_heads
        scale = math.sqrt(d_head)
        Q = self.q_proj(in_features)
        K = self.k_proj(in_features)
        V = self.v_proj(in_features)
        Q = Q.unflatten(-1, (num_heads, d_head)).transpose(-2,-3) # (..., num_head, sequence_length, d_head)
        K = K.unflatten(-1, (num_heads, d_head)).transpose(-2,-3) # (..., num_head, sequence_length, d_head)
        if self.position_encoder is not None and token_positions is not None:
            Q = self.position_encoder(Q, token_positions)
            K = self.position_encoder(K, token_positions)
        V = V.unflatten(-1, (num_heads, d_head)).transpose(-2,-3) # (..., num_head, sequence_length, d_head)
        mask = torch.tril(torch.ones(sequence_length,sequence_length, dtype = torch.bool, device = in_features.device)) # (sequence_length, sequence_length)
        attention_score = Q @ K.transpose(-2,-1) / scale # (..., num_head, sequence_length, sequence_length)
        attention_score = attention_score.masked_fill(mask=~mask, value=float('-inf'))
        weights:Float[Tensor, " ... num_heads sequence_length sequence_length"] = self.softmax(attention_score, -1)
        ret = weights @ V # (..., num_head, sequence_length, d_head)
        ret = ret.transpose(-3,-2).flatten(start_dim=-2) # (..., sequence_length, d_model)
        return self.output_proj(ret)

class RMSNorm(nn.Module):
    d_model: int
    eps: float
    device: torch.device
    dtype: torch.dtype
    weight: nn.Parameter
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(d_model))

    def forward(self, x: Float[Tensor, " ... d_model"]):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True)/self.d_model + self.eps)
        res = x / rms * self.weight
        res = res.to(in_dtype)
        return res
