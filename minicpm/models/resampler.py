"""
Resampler（自研前向，使用 HF 权重）：
- 复刻 HF resampler 的单层 cross-attn + 线性投影逻辑，避免动态控制流。
"""

from typing import Optional

import torch
import torch.nn as nn


class ResamplerExport(nn.Module):
    def __init__(
        self,
        queries: nn.Parameter,
        kv_proj: nn.Module,
        attn: nn.MultiheadAttention,
        ln_q: nn.Module,
        ln_kv: nn.Module,
        ln_post: nn.Module,
        proj: nn.Parameter,
        num_heads: int,
        pos_embed: torch.Tensor,
    ):
        super().__init__()
        self.queries = queries
        self.kv_proj = kv_proj
        self.attn = attn
        self.ln_q = ln_q
        self.ln_kv = ln_kv
        self.ln_post = ln_post
        self.proj = proj
        self.num_heads = num_heads
        self.head_dim = queries.shape[-1] // num_heads
        self.num_queries = queries.shape[0]
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    @classmethod
    def from_hf(cls, hf_resampler: nn.Module, num_heads: int):
        return cls(
            queries=hf_resampler.query,
            kv_proj=hf_resampler.kv_proj,
            attn=hf_resampler.attn,
            ln_q=hf_resampler.ln_q,
            ln_kv=hf_resampler.ln_kv,
            ln_post=hf_resampler.ln_post,
            proj=hf_resampler.proj,
            num_heads=num_heads,
            pos_embed=hf_resampler.pos_embed,
        )

    def forward(self, vision_hidden: torch.Tensor, tgt_sizes: Optional[torch.Tensor] = None):
        assert tgt_sizes is not None
        bs = vision_hidden.shape[0]
        device = vision_hidden.device
        dtype = vision_hidden.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]
        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype))
            key_padding_mask[i, patch_len[i] :] = True
        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)

        x = self.kv_proj(vision_hidden)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.queries)  # Q * D
        out = self.attn(
            self._repeat(q, bs),
            x + pos_embed,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]  # Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)
