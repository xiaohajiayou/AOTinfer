from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn

from .rotary import apply_rotary_pos_emb
from .utils import repeat_kv


@dataclass
class ExportAttentionConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention_dropout: float = 0.0


class ExportAttention(nn.Module):
    """
    Export-friendly 自注意力模块。KV cache 由调用方提供，mask 为 4D tensor。
    """

    def __init__(
        self,
        config: ExportAttentionConfig,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        o_proj: nn.Linear,
    ):
        super().__init__()
        self.config = config
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.scaling = config.head_dim**-0.5
        self.num_groups = config.num_attention_heads // config.num_key_value_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cache_len: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [B, N, hidden]
            cos/sin: [B, total_seq, head_dim]
            key_cache/value_cache: [B, num_kv_heads, cache_len, head_dim]
            cache_len: scalar tensor
            attention_mask: [B, 1, N, cache_len+N]
        """

        bsz, n_step, _ = hidden_states.shape
        dtype = hidden_states.dtype

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.head_dim

        q = q.view(bsz, n_step, num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, n_step, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, n_step, num_kv_heads, head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = torch.cat([key_cache, k], dim=2)
        v = torch.cat([value_cache, v], dim=2)
        new_cache_len = cache_len + n_step
        new_key_cache = k
        new_value_cache = v

        k_full = repeat_kv(k, self.num_groups)
        v_full = repeat_kv(v, self.num_groups)

        q = q * self.scaling
        attn_weights = torch.matmul(q, k_full.transpose(2, 3))
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)
        attn_output = torch.matmul(attn_weights, v_full)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, n_step, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, new_key_cache, new_value_cache, new_cache_len
