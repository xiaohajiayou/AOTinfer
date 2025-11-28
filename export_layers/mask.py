from __future__ import annotations

import torch


def build_causal_mask(
    batch_size: int,
    n_step: int,
    past_key_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    构造形状为 [B, 1, n_step, past_key_len + n_step] 的因果 mask。
    仅使用张量算子，避免对 cache_len 执行 .item() 或 Python 控制流，
    以便 torch.export 能够处理动态长度。
    """

    total_k = past_key_len + n_step
    large_neg = torch.finfo(dtype).min
    mask = torch.full(
        (batch_size, 1, n_step, total_k),
        fill_value=large_neg,
        dtype=dtype,
        device=device,
    )
    query_pos = torch.arange(n_step, device=device).view(1, 1, n_step, 1)
    key_pos = torch.arange(total_k, device=device).view(1, 1, 1, total_k)
    mask = mask.masked_fill(key_pos <= (past_key_len + query_pos), 0.0)
    return mask
