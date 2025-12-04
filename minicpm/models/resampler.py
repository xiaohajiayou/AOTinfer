"""
Resampler（自研前向，使用 HF 权重）：
- 复刻 HF resampler 的单层 cross-attn + 线性投影逻辑，避免动态控制流。
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, vision_hidden: torch.Tensor, tgt_sizes: Optional[torch.Tensor] = None, export_mode: bool = False):
        assert tgt_sizes is not None
        bs = vision_hidden.shape[0]
        device = vision_hidden.device
        orig_dtype = vision_hidden.dtype
        compute_dtype = orig_dtype
        head_dim = self.head_dim
        num_heads = self.num_heads

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]
        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(compute_dtype))
            key_padding_mask[i, patch_len[i] :] = True
        # L * B * D
        pos_embed = torch.nn.utils.rnn.pad_sequence(pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)

        # 线性投影 K/V，保持与 HF 的 batch_first=False 逻辑一致
        if isinstance(self.kv_proj, nn.Linear):
            kv_w = self.kv_proj.weight.to(compute_dtype)
            kv_b = self.kv_proj.bias.to(compute_dtype) if self.kv_proj.bias is not None else None
            x = F.linear(vision_hidden.to(compute_dtype), kv_w, kv_b)
        else:
            x = self.kv_proj(vision_hidden.to(compute_dtype))
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        # Q: (Q, B, D)
        q_in = self.ln_q(self.queries.to(compute_dtype))
        q_in = q_in.unsqueeze(1).expand(-1, bs, -1)  # Q,B,D

        k_in = x + pos_embed  # L,B,D
        v_in = x  # L,B,D

        # 统一走 math 路径，复刻 HF multi_head_attention_forward（need_weights=False）
        attn_out, _ = self.attn.multi_head_attention_forward(
            q_in,
            k_in,
            v_in,
            embed_dim_to_check=self.attn.embed_dim,
            num_heads=self.attn.num_heads,
            in_proj_weight=self.attn.in_proj_weight.to(compute_dtype),
            in_proj_bias=self.attn.in_proj_bias.to(compute_dtype) if self.attn.in_proj_bias is not None else None,
            bias_k=self.attn.bias_k.to(compute_dtype) if self.attn.bias_k is not None else None,
            bias_v=self.attn.bias_v.to(compute_dtype) if self.attn.bias_v is not None else None,
            add_zero_attn=self.attn.add_zero_attn,
            dropout_p=0.0,
            out_proj_weight=self.attn.out_proj.weight.to(compute_dtype),
            out_proj_bias=self.attn.out_proj.bias.to(compute_dtype) if self.attn.out_proj.bias is not None else None,
            training=False,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=None,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            average_attn_weights=True,
            is_causal=False,
        )
        attn_out = attn_out.permute(1, 0, 2)  # B,Q,D

        attn_out = F.layer_norm(
            attn_out,
            self.ln_post.normalized_shape,
            self.ln_post.weight.to(compute_dtype),
            self.ln_post.bias.to(compute_dtype) if self.ln_post.bias is not None else None,
            self.ln_post.eps,
        )
        attn_out = attn_out @ self.proj.to(compute_dtype)
        return attn_out.to(orig_dtype)

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)
