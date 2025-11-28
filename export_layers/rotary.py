import math
from typing import Optional, Tuple

import torch
from torch import nn


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotary embedding 应用到 query/key。
    q/k: [B, num_heads, seq, head_dim]
    cos/sin: [B, seq, head_dim]
    """

    cos = cos.unsqueeze(1)  # [B,1,seq,head_dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class ExportRotaryEmbedding(nn.Module):
    """
    简化版 RoPE，包含常见的 base frequency 与可选 scaling。
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float = 10000.0,
        rope_scaling: Optional[dict] = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.rope_scaling = rope_scaling or {}

        inv_freq = 1.0 / (
            (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: [B, seq]
        seq_len: 当前序列长度（历史+本步），用于裁剪 cos/sin
        """

        # [B, seq, head_dim/2]
        inv_freq = self.inv_freq.unsqueeze(0).unsqueeze(0)
        theta = position_ids.float().unsqueeze(-1) * inv_freq
        emb = torch.cat((theta, theta), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos[:, -seq_len:], sin[:, -seq_len:]
