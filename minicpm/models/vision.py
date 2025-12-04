"""
SigLIP 视觉编码器（自研前向，固定形状，使用 HF 权重）
- 固定 patch_size=14，输入 [S,3,980,980] -> patch 网格 70x70，总 L=4900
- 去掉动态 position_id 计算，直接使用 learned position_embedding
"""

from typing import Optional

import torch
import torch.nn as nn


class SiglipVisionExport(nn.Module):
    def __init__(self, patch_embedding: nn.Conv2d, position_embedding: nn.Embedding, layers: nn.ModuleList, post_layernorm: nn.Module):
        super().__init__()
        self.patch_embedding = patch_embedding
        self.position_embedding = position_embedding
        self.layers = layers
        self.post_layernorm = post_layernorm
        self.num_patches = position_embedding.num_embeddings

    @classmethod
    def from_hf(cls, hf_vpm: nn.Module):
        return cls(
            patch_embedding=hf_vpm.embeddings.patch_embedding,
            position_embedding=hf_vpm.embeddings.position_embedding,
            layers=hf_vpm.encoder.layers,
            post_layernorm=hf_vpm.post_layernorm,
        )

    def forward(self, pixel_values: torch.Tensor, tgt_sizes: Optional[torch.Tensor] = None):
        """
        Args:
            pixel_values: [S,3,H,W]，H/W 为 patch_size 的整数倍（可变）
            tgt_sizes: [S,2]，每个切片的 patch 网格 (H/patch, W/patch)
        Returns:
            hidden: [S, L, hidden_size], L = H/patch * W/patch
        """
        bsz = pixel_values.size(0)
        x = self.patch_embedding(pixel_values)  # [S, hidden, H', W']
        H_, W_ = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [S, L, hidden], L = H'*W'

        # 位置编码：按照 HF bucketize 方式将实际 patch 网格映射到 70x70 的 embedding 索引
        if tgt_sizes is None:
            nb_patches_h = torch.full((bsz,), H_, device=x.device, dtype=torch.int32)
            nb_patches_w = torch.full((bsz,), W_, device=x.device, dtype=torch.int32)
        else:
            nb_patches_h = tgt_sizes[:, 0]
            nb_patches_w = tgt_sizes[:, 1]

        num_per_side = int(self.position_embedding.num_embeddings**0.5)  # 70
        boundaries = torch.arange(1 / num_per_side, 1.0, 1 / num_per_side, device=x.device)
        position_ids = torch.zeros((bsz, H_ * W_), device=x.device, dtype=torch.long)

        for b in range(bsz):
            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h[b], device=x.device)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w[b], device=x.device)
            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
            pos_ids = (bucket_coords_h[:, None] * num_per_side + bucket_coords_w).flatten()
            position_ids[b, : pos_ids.numel()] = pos_ids

        x = x + self.position_embedding(position_ids)
        attn_mask = None
        for layer in self.layers:
            x = layer(x, attention_mask=attn_mask, output_attentions=False)[0]
        x = self.post_layernorm(x)
        return x
