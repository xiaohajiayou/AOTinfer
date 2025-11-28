from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from export_layers import ExportAttention, ExportAttentionConfig, build_causal_mask


class Qwen2ExportBlock(nn.Module):
    def __init__(self, hf_layer: nn.Module, config):
        super().__init__()
        self.norm1 = hf_layer.input_layernorm
        self.norm2 = hf_layer.post_attention_layernorm
        attn_conf = ExportAttentionConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            attention_dropout=0.0,
        )
        self.attn = ExportAttention(
            attn_conf,
            hf_layer.self_attn.q_proj,
            hf_layer.self_attn.k_proj,
            hf_layer.self_attn.v_proj,
            hf_layer.self_attn.o_proj,
        )
        self.mlp = hf_layer.mlp

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        key_cache_layer: torch.Tensor,
        value_cache_layer: torch.Tensor,
        cache_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        attn_mask = build_causal_mask(
            batch_size=hidden_states.shape[0],
            n_step=hidden_states.shape[1],
            past_key_len=key_cache_layer.shape[2],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        attn_output, new_key, new_value, new_cache_len = self.attn(
            hidden_states,
            cos,
            sin,
            key_cache_layer,
            value_cache_layer,
            cache_len,
            attn_mask,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_key, new_value, new_cache_len


class Qwen2ExportModel(nn.Module):
    def __init__(self, hf_model: nn.Module):
        super().__init__()
        config = hf_model.config
        self.embed = hf_model.model.embed_tokens
        self.rotary = hf_model.model.rotary_emb
        self.layers = nn.ModuleList(
            [Qwen2ExportBlock(layer, config) for layer in hf_model.model.layers]
        )
        self.norm = hf_model.model.norm
        self.lm_head = hf_model.lm_head
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        key_cache: List[torch.Tensor],
        value_cache: List[torch.Tensor],
        cache_len: torch.Tensor,
    ):
        B, N = input_ids.shape
        device = input_ids.device
        hidden_states = self.embed(input_ids)

        position_ids = torch.arange(
            cache_len.item(), cache_len.item() + N, device=device
        ).view(1, N)
        cos, sin = self.rotary(hidden_states, position_ids)
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
