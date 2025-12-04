from typing import Optional

import torch


def scatter_vision_tokens(
    text_embeds: torch.Tensor,
    vision_tokens: torch.Tensor,
    image_bound: torch.Tensor,
    max_tokens: Optional[int] = None,
) -> torch.Tensor:
    """
    Replace placeholder positions in text embeddings with vision tokens.

    Args:
        text_embeds: [B, T, H]
        vision_tokens: [N, L, H] where N = num_slices (flattened across batch)
        image_bound: [B, K, 2] start/end indices per slice (exclusive end), padded with zeros
        max_tokens: optional cap on how many vision tokens to consume (for safety)
    Returns:
        new_embeds: [B, T, H]
    """
    new_embeds = text_embeds.clone()
    B, _, H = text_embeds.shape
    _, K, _ = image_bound.shape

    vidx = 0
    for b in range(B):
        for k in range(K):
            start, end = image_bound[b, k]
            if end <= start:
                continue  # padded
            length = int(end - start)
            if max_tokens is not None:
                length = min(length, max_tokens)
            if length <= 0:
                continue
            if vidx >= vision_tokens.shape[0]:
                break
            vt = vision_tokens[vidx]
            vidx += 1
            if vt.shape[0] < length:
                pad_len = length - vt.shape[0]
                vt = torch.cat([vt, torch.zeros(pad_len, H, device=vt.device, dtype=vt.dtype)], dim=0)
            new_embeds[b, start:end] = vt[:length]
    return new_embeds
