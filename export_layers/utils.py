import torch
from typing import Optional


def repeat_kv(hidden_states: torch.Tensor, num_groups: int) -> torch.Tensor:
    """
    Expand KV heads to match查询头数量。

    hidden_states: [B, num_kv_heads, seq, head_dim]
    返回: [B, num_kv_heads * num_groups, seq, head_dim]
    """

    if num_groups == 1:
        return hidden_states

    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, num_groups, seq_len, head_dim
    )
    return hidden_states.reshape(bsz, num_kv_heads * num_groups, seq_len, head_dim)

