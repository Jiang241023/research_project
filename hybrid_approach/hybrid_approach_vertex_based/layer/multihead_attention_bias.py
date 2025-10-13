# layer/multihead_attention_bias.py
import math
import torch
from torch import nn

class MultiheadSelfAttentionWithBias(nn.Module):
    """
    Standard multi-head self-attention but accepts an additive bias [B,N,N]
    added to the attention logits BEFORE softmax.
    """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        """
        x: [B,N,d_model]
        attn_bias: [B,N,N] (additive to logits). Can contain -inf at pads.
        key_padding_mask: [B,N] (True = keep, False = pad); if provided, pads set to -inf.
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,N,d]
        scale = 1.0 / math.sqrt(self.d_head)
        attn = (q @ k.transpose(-2, -1)) * scale  # [B,H,N,N]

        if attn_bias is not None:
            attn = attn + attn_bias[:, None, :, :]  # broadcast to heads

        if key_padding_mask is not None:
            mask = ~key_padding_mask  # False where pad
            attn = attn.masked_fill(mask[:, None, None, :], float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        y = attn @ v  # [B,H,N,d]
        y = y.transpose(1, 2).reshape(B, N, D)
        y = self.proj_drop(self.proj(y))
        return y
