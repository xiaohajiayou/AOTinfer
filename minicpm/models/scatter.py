import torch
from typing import Optional


def scatter_vision_tokens(
    text_embeds: torch.Tensor,
    vision_tokens: Optional[torch.Tensor],
    image_bound: Optional[torch.Tensor],
    export_mode: bool = False,
) -> torch.Tensor:
    """
    Replace placeholder positions in text embeddings with vision tokens.
    Args:
        text_embeds: [B, T, H]
        vision_tokens: [N, 64, H] flattened per slice
        image_bound: [B, K, 2] start/end indices per slice (exclusive end)
        export_mode: if True, assume单张图单区间，走静态路径
    """
    if vision_tokens is None or image_bound is None:
        return text_embeds

    new_embeds = text_embeds.clone()
    if export_mode:
        start = int(image_bound[0, 0, 0])
        end = int(image_bound[0, 0, 1])
        length = max(end - start, 0)
        if length > 0:
            new_embeds[0, start:end] = vision_tokens[0, :length].to(new_embeds.dtype)
        return new_embeds

    B = text_embeds.size(0)
    H = text_embeds.size(-1)
    vt = vision_tokens.to(new_embeds.dtype)
    offset = 0
    for b in range(B):
        bounds = image_bound[b]
        for m in range(bounds.size(0)):
            start = int(bounds[m, 0])
            end = int(bounds[m, 1])
            length = max(end - start, 0)
            if length > 0:
                new_embeds[b, start:end] = vt[offset, :length]
                offset += length
    return new_embeds
