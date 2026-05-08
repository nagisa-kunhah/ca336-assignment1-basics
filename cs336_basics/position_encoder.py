import torch
from torch import Tensor,nn
from jaxtyping import Bool, Float, Int


class RoPE(nn.Module):
    d_k: int
    theta: float
    max_seq_len: int
    sin_cache: Float[Tensor, " max_seq_len d_k // 2"]
    cos_cache: Float[Tensor, " max_seq_len d_k // 2"]
    device: torch.device
    dtype: torch.dtype
    def __init__(self, d_k:int, theta: float, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        dim_half = d_k // 2
        k = torch.arange(dim_half, dtype=dtype, device=device) # (dim_half)
        freq = theta ** (2*k / d_k)
        freq = freq.unsqueeze(0) #(1, dim_half)
        pos = torch.arange(max_seq_len, dtype=dtype, device=device).unsqueeze(1) # (max_seq_len, 1)
        angle = pos / freq #(max_seq_len, d_k // 2)
        self.sin_cache = torch.sin(angle).to(device=device)
        self.cos_cache = torch.cos(angle).to(device=device)
        self.device = device
        self.dtype = dtype


    def forward(self, input: Float[Tensor, " ... sequence_length d_k"], token_positions: Int[Tensor, " ... sequence_length"]):
        cos = self.cos_cache[token_positions] # (sequence_length, d_k // 2)
        sin = self.sin_cache[token_positions] # (sequence_length, d_k // 2)
        x_even = input[..., 0::2] # (..., sequence_length, d_k // 2)
        x_odd = input[..., 1::2] # (..., sequence_length, d_k // 2)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.empty(input.shape, device=self.device, dtype=self.dtype)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out