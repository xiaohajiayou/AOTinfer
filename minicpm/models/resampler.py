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

    # def forward(self, vision_hidden: torch.Tensor, tgt_sizes: Optional[torch.Tensor] = None):
    def forward(self, vision_hidden: torch.Tensor, tgt_sizes: Optional[torch.Tensor] = None):
        # 1. 固定导出用的网格尺寸（转为静态数值，消除动态性）
        TGT_H = 25
        TGT_W = 41
        tgt_sizes = torch.tensor([[TGT_H, TGT_W]], dtype=torch.int32, device=vision_hidden.device)
        assert tgt_sizes is not None
        
        # 2. 基础维度计算 + 强制统一dtype（核心修复点）
        bs = vision_hidden.shape[0]
        device = vision_hidden.device
        orig_dtype = vision_hidden.dtype
        # 强制指定compute_dtype为bfloat16，避免dtype不一致
        compute_dtype = torch.bfloat16 if orig_dtype in [torch.bfloat16, torch.float32] else orig_dtype

        # 3. 批量计算patch长度（无循环 + 统一dtype）
        patch_len = torch.tensor([TGT_H * TGT_W], device=device, dtype=torch.int32).repeat(bs)
        max_patch_len = patch_len[0].item()

        # 4. 批量生成key_padding_mask（统一device/dtype）
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)
        idx_matrix = torch.arange(max_patch_len, device=device, dtype=torch.int64).unsqueeze(0).expand(bs, -1)
        patch_len_expand = patch_len.unsqueeze(1).expand(-1, max_patch_len).to(torch.int64)
        key_padding_mask = idx_matrix >= patch_len_expand

        # 5. 静态生成pos_embed（强制统一dtype）
        pos_embed_slice = self.pos_embed[:TGT_H, :TGT_W, :].to(compute_dtype)
        pos_embed_slice = pos_embed_slice.reshape((TGT_H * TGT_W, -1))
        pos_embed = pos_embed_slice.unsqueeze(0).repeat(bs, 1, 1).to(compute_dtype)
        pos_embed = pos_embed.permute(1, 0, 2)

        # 6. 线性投影 K/V（强制统一所有张量dtype）
        vision_hidden = vision_hidden.to(compute_dtype)  # 核心：先转换输入dtype
        if isinstance(self.kv_proj, nn.Linear):
            kv_w = self.kv_proj.weight.to(compute_dtype)
            kv_b = self.kv_proj.bias.to(compute_dtype) if self.kv_proj.bias is not None else None
            x = F.linear(vision_hidden, kv_w, kv_b)
        else:
            x = self.kv_proj(vision_hidden)
        x = self.ln_kv(x).to(compute_dtype)  # 确保LayerNorm输出dtype一致
        x = x.permute(1, 0, 2)  # L * B * D

        # 7. Q/K/V 构造（强制统一所有张量dtype）
        q_in = self.ln_q(self.queries.to(compute_dtype))
        q_in = q_in.unsqueeze(1).expand(-1, bs, -1).to(compute_dtype)  # Q,B,D
        k_in = (x + pos_embed).to(compute_dtype)  # 确保加法后dtype一致
        v_in = x.to(compute_dtype)

        # 8. 多头注意力计算（强制所有参数统一dtype）
        # 预处理注意力参数，确保dtype完全一致
        attn_kwargs = {
            "embed_dim_to_check": self.attn.embed_dim,
            "num_heads": self.attn.num_heads,
            "in_proj_weight": self.attn.in_proj_weight.to(compute_dtype),
            "in_proj_bias": self.attn.in_proj_bias.to(compute_dtype) if self.attn.in_proj_bias is not None else None,
            "bias_k": self.attn.bias_k.to(compute_dtype) if self.attn.bias_k is not None else None,
            "bias_v": self.attn.bias_v.to(compute_dtype) if self.attn.bias_v is not None else None,
            "add_zero_attn": self.attn.add_zero_attn,
            "dropout_p": 0.0,
            "out_proj_weight": self.attn.out_proj.weight.to(compute_dtype),
            "out_proj_bias": self.attn.out_proj.bias.to(compute_dtype) if self.attn.out_proj.bias is not None else None,
            "training": False,
            "key_padding_mask": key_padding_mask,
            "need_weights": False,
            "attn_mask": None,
            "use_separate_proj_weight": False,
            "q_proj_weight": None,
            "k_proj_weight": None,
            "v_proj_weight": None,
            "average_attn_weights": True,
            "is_causal": False,
        }
        # 确保key_padding_mask的device与输入一致
        attn_kwargs["key_padding_mask"] = attn_kwargs["key_padding_mask"].to(device)

        attn_out, _ = self.attn.multi_head_attention_forward(
            q_in.to(compute_dtype),
            k_in.to(compute_dtype),
            v_in.to(compute_dtype),
            **attn_kwargs
        )

        # 9. 后处理（统一dtype + 最终转回原dtype）
        attn_out = attn_out.permute(1, 0, 2).to(compute_dtype)  # B,Q,D
        attn_out = F.layer_norm(
            attn_out,
            self.ln_post.normalized_shape,
            self.ln_post.weight.to(compute_dtype),
            self.ln_post.bias.to(compute_dtype) if self.ln_post.bias is not None else None,
            self.ln_post.eps,
        )
        # 矩阵乘法前强制统一dtype（核心修复点）
        attn_out = attn_out.to(compute_dtype)
        proj = self.proj.to(compute_dtype)
        attn_out = attn_out @ proj

        # 最终转回原dtype，确保输出兼容
        return attn_out.to(orig_dtype)

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)
