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
        rotary_emb,
        num_kv_heads: int,
        head_dim: int,
    ):
        super().__init__()
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.rotary = rotary_emb
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        key_cache: List[torch.Tensor],
        value_cache: List[torch.Tensor],
        cache_len: torch.Tensor,
    ):
        hidden_states = inputs_embeds
        B, N, _ = hidden_states.shape
        device = hidden_states.device

        # position_ids = torch.arange(0, N, device=device).view(1, N) + cache_len.unsqueeze(-1)
        # cos, sin = self.rotary(position_ids, N)
        position_ids = torch.arange(0, N, device=device).view(1, N) + cache_len.unsqueeze(-1)
        cos, sin = self.rotary(position_ids, N, hidden_states.dtype)
        
        # position_ids = torch.arange(
        #     cache_len.item(), cache_len.item() + N, device=device
        # ).view(1, N)
        # cos, sin = self.rotary(hidden_states, position_ids)
        new_key = []
        new_value = []
        for i, block in enumerate(self.layers):
            layer_cache_len = torch.tensor(
                key_cache[i].shape[2], dtype=torch.long, device=device
            )
            hidden_states, key_layer, value_layer, layer_cache_len = block(
                hidden_states,
                cos,
                sin,
                key_cache[i],
                value_cache[i],
                layer_cache_len,
            )
            new_key.append(key_layer)
            new_value.append(value_layer)
            cache_len = layer_cache_len

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, new_key, new_value, cache_len


def build_llm_from_hf(hf_llm: nn.Module):
    """
    使用 HF Qwen2ForCausalLM 子模块权重构建 decoder（输入为 inputs_embeds）。
    """
    cfg = hf_llm.config
    layers = nn.ModuleList([Qwen2ExportBlock(hf_layer, cfg) for hf_layer in hf_llm.model.layers])
    norm = hf_llm.model.norm
    max_pos = getattr(cfg, "max_position_embeddings", 32768)
    rope_theta = getattr(cfg, "rope_theta", 10000.0)
    
    lm_head = hf_llm.lm_head
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // num_heads
    rotary = ExportRotaryEmbedding(head_dim, max_position_embeddings=max_pos, base=rope_theta)
    # rotary = hf_llm.model.layers[0].self_attn.rotary_emb
    decoder = MiniCPMDecoder(layers, norm, lm_head, rotary, num_kv_heads=num_kv_heads, head_dim=head_dim)
    return decoder
