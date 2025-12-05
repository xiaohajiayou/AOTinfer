import torch
from typing import Optional


def scatter_vision_tokens(
    text_embeds: torch.Tensor,
    vision_tokens: Optional[torch.Tensor],
    image_bound: Optional[torch.Tensor],
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

    # 可能没有任何视觉区间，直接返回
    if image_bound.numel() == 0 or vision_tokens.numel() == 0:
        return text_embeds

    new_embeds = text_embeds.clone()

    # 避免 torch.export 时的 python int 转换，全部使用张量运算
    start = image_bound[0, 0, 0]
    end = image_bound[0, 0, 1]
    seq_len = new_embeds.shape[1]
    length = torch.clamp(end - start, min=0)

    # 仅单图单区间：vision_tokens[0] 长度固定 64，超出部分由 mask 剪裁
    max_vis = vision_tokens.shape[1]
    idx = torch.arange(max_vis, device=new_embeds.device, dtype=start.dtype)
    target_pos = start + idx  # 相对起点的写入位置
    valid = (idx < length) & (target_pos < seq_len)
    target_indices = target_pos[valid].long()
    src = vision_tokens[0].to(new_embeds.dtype)[valid]
    # 空集时 index_copy_ 是 no-op
    new_embeds[0].index_copy_(0, target_indices, src)
    return new_embeds

    # B = text_embeds.size(0)
    # H = text_embeds.size(-1)
    # vt = vision_tokens.to(new_embeds.dtype)
    # offset = 0
    # for b in range(B):
    #     bounds = image_bound[b]
    #     for m in range(bounds.size(0)):
    #         start = int(bounds[m, 0])
    #         end = int(bounds[m, 1])
    #         length = max(end - start, 0)
    #         if length > 0:
    #             new_embeds[b, start:end] = vt[offset, :length]
    #             offset += length
    # return new_embeds
