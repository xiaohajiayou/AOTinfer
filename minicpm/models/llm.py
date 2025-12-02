"""
MiniCPM LLM 组装：复用 qwen2_export block，但用显式 ExportRotaryEmbedding 计算 cos/sin，支持 inputs_embeds。
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from export_layers.rotary import ExportRotaryEmbedding
from qwen.models.qwen2_block import Qwen2ExportBlock


class MiniCPMDecoder(nn.Module):
    """
    以 inputs_embeds 作为入口的解码器，内部使用 Qwen2ExportBlock（与 qwen2 适配保持一致）。
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        norm: nn.Module,
        lm_head: nn.Linear,
        rotary_emb: ExportRotaryEmbedding,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.rotary_emb = rotary_emb
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ):
        hidden_states = inputs_embeds
        bsz, seqlen, _ = hidden_states.shape
        device = hidden_states.device

        if past_key_values is None:
            past_key_values = []
            for _ in range(len(self.layers)):
                past_key_values.append(
                    (
                        torch.zeros(bsz, self.num_kv_heads, 0, self.head_dim, device=device, dtype=hidden_states.dtype),
                        torch.zeros(bsz, self.num_kv_heads, 0, self.head_dim, device=device, dtype=hidden_states.dtype),
                    )
                )

        # 当前 cache_len 由第一层 kv 判断
        cache_len = past_key_values[0][0].shape[2] if len(past_key_values) > 0 else 0
        position_ids = torch.arange(cache_len, cache_len + seqlen, device=device).view(1, seqlen)
        cos, sin = self.rotary_emb(position_ids, seqlen)

        new_kvs = []
        for layer, (k_cache, v_cache) in zip(self.layers, past_key_values):
            layer_cache_len = torch.tensor(k_cache.shape[2], device=device, dtype=torch.long)
            hidden_states, k_cache, v_cache, layer_cache_len = layer(
                hidden_states, cos, sin, k_cache, v_cache, layer_cache_len
            )
            new_kvs.append((k_cache, v_cache))
            cache_len = layer_cache_len  # 更新 cache_len 供后续层使用

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, tuple(new_kvs)


def build_llm_from_hf(hf_llm: nn.Module):
    """
    使用 HF Qwen2ForCausalLM 子模块权重构建 decoder（输入为 inputs_embeds）。
    """
    cfg = hf_llm.config
    layers = nn.ModuleList([Qwen2ExportBlock(hf_layer, cfg) for hf_layer in hf_llm.model.layers])
    norm = hf_llm.model.norm
    lm_head = hf_llm.lm_head
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // num_heads
    max_pos = getattr(cfg, "max_position_embeddings", 32768)
    rope_theta = getattr(cfg, "rope_theta", 10000.0)
    rotary = ExportRotaryEmbedding(head_dim, max_position_embeddings=max_pos, base=rope_theta)
    decoder = MiniCPMDecoder(layers, norm, lm_head, rotary, num_kv_heads=num_kv_heads, head_dim=head_dim)
    embed_tokens = hf_llm.model.embed_tokens
    scale_emb = getattr(cfg, "scale_emb", 1.0)
    return decoder, embed_tokens, scale_emb
