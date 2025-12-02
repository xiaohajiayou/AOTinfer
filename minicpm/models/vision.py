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
            pixel_values: [S,3,980,980]，H/W 固定，可 pad
        Returns:
            hidden: [S, L, hidden_size], L=4900
        """
        bsz = pixel_values.size(0)
        x = self.patch_embedding(pixel_values)  # [S, hidden, H', W']
        x = x.flatten(2).transpose(1, 2)  # [S, L, hidden]
        # 直接使用顺序位置
        pos_ids = torch.arange(self.num_patches, device=x.device).unsqueeze(0).expand(bsz, -1)
        x = x + self.position_embedding(pos_ids)
        attn_mask = None  # 固定形状，不使用动态 mask
        for layer in self.layers:
            x = layer(x, attention_mask=attn_mask, output_attentions=False)[0]
        x = self.post_layernorm(x)
        return x
