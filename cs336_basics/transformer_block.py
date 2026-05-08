import torch
from torch import Tensor,nn
from cs336_basics.nn_utils import MultiheadSelfAttention, RMSNorm, Linear, RoPE, SwigGLU
from jaxtyping import Bool, Float, Int


class TransformerBlock(nn.Module):
    d_model: int
    num_heads: int
    d_ff: int
    max_seq_len: int
    theta: float
    attn: MultiheadSelfAttention
    ln1: RMSNorm
    ffn: SwigGLU
    ln2: RMSNorm
    device: torch.device
    dtype: torch.dtype
    def __init__(self, d_model:int, num_head: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_head
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype
        self.attn = MultiheadSelfAttention(
            d_model=d_model,
            num_heads=num_head,
            position_encoder=RoPE(
                d_k=d_model // num_head,
                max_seq_len=max_seq_len,
                theta=theta,
                device=device,
                dtype=dtype
            ),
            device=device,
            dtype=dtype
        )
        self.ln1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ffn = SwigGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
    
    def forward(self, x: Float[Tensor, " batch sequence_length d_model"]):
        sequence_length = x.size(-2)
        batch_shape = x.shape[:-2]
        position = torch.arange(sequence_length, dtype=torch.int, device=self.device).expand(*batch_shape, sequence_length)
        x = self.attn(self.ln1(x), position) + x
        x = self.ffn(self.ln2(x)) + x
        return x
