from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn

from export_layers import ExportAttention, ExportAttentionConfig, build_causal_mask


def _get_attr(obj, path: str):
    """Safely walk dotted attr path, returning None if any link is missing."""
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _text_backbone(hf_model: nn.Module):
    """
    Try to locate the text backbone inside a Qwen2.5-VL HF model.
    Expected structures (may vary by version):
      - model.language_model (common)
      - model (already language model)
    """
    for candidate in ("model.language_model", "language_model", "model"):
        got = _get_attr(hf_model, candidate)
        if got is not None and hasattr(got, "layers"):
            return got
    raise ValueError("Cannot locate language_model inside provided HF Qwen2.5-VL model")


def _lm_head(hf_model: nn.Module):
    for candidate in ("lm_head", "model.lm_head", "language_model.lm_head"):
        got = _get_attr(hf_model, candidate)
        if got is not None:
            return got
    raise ValueError("Cannot locate lm_head inside provided HF Qwen2.5-VL model")


class Qwen2_5_VLExportBlock(nn.Module):
    def __init__(self, hf_layer: nn.Module, config):
        super().__init__()
        self.norm1 = hf_layer.input_layernorm
        self.norm2 = hf_layer.post_attention_layernorm
        mrope_section = None
        if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
            mrope_section = config.rope_scaling.get("mrope_section", None)
        attn_conf = ExportAttentionConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            attention_dropout=0.0,
            mrope_section=mrope_section,
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


class Qwen2_5_VLExportModel(nn.Module):
    """
    Export-friendly Qwen2.5-VL model.

    Assumptions:
      - Visual tokens are provided as projected embeddings (B, V, hidden).
      - No explicit cross-attention; vision + text tokens are concatenated.
    """

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        text_model = _text_backbone(hf_model)
        config = text_model.config
        self.embed = text_model.embed_tokens
        self.rotary = text_model.rotary_emb
        self.layers = nn.ModuleList(
            [Qwen2_5_VLExportBlock(layer, config) for layer in text_model.layers]
        )
        self.norm = text_model.norm
        self.lm_head = _lm_head(hf_model)
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        key_cache: List[torch.Tensor],
        value_cache: List[torch.Tensor],
        cache_len: torch.Tensor,
        vision_hidden_states: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            input_ids: [B, N_text]
            key_cache/value_cache: list of [B, num_kv_heads, cache_len, head_dim]
            cache_len: scalar tensor
            vision_hidden_states: optional [B, N_vision, hidden] projected vision tokens
        """
        B, N_text = input_ids.shape
        device = input_ids.device
        hidden_states = self.embed(input_ids)

        vision_len = 0
        if vision_hidden_states is not None:
            hidden_states = torch.cat([vision_hidden_states, hidden_states], dim=1)
            vision_len = vision_hidden_states.shape[1]

        total_len = hidden_states.shape[1]
        # Qwen2.5-VL rotary expects position_ids shape [3, B, total_len]
        base_pos = torch.arange(total_len, device=device).view(1, 1, -1)
        cache_offset = cache_len.view(1, 1, 1)
        position_ids = base_pos + cache_offset
        position_ids = position_ids.expand(3, B, -1)
        cos, sin = self.rotary(hidden_states, position_ids)
        new_key: List[torch.Tensor] = []
        new_value: List[torch.Tensor] = []
        for i, block in enumerate(self.layers):
            layer_cache_len = torch.tensor(key_cache[i].shape[2], dtype=torch.long, device=device)
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
        logits = self.lm_head(hidden_states[:, vision_len:, :])
        return logits, new_key, new_value, cache_len
