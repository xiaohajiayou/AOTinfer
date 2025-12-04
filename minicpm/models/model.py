"""
MiniCPM top-level model skeleton.
TODO: replace placeholders with actual Qwen2 backbone and full vision/resampler impl.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .scatter import scatter_vision_tokens
from .vision import SiglipVisionExport
from .resampler import ResamplerExport
from .llm import build_llm_from_hf


@dataclass
class MiniCPMInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    pixel_values: Optional[torch.Tensor] = None
    tgt_sizes: Optional[torch.Tensor] = None
    image_bound: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple] = None
    position_ids: Optional[torch.Tensor] = None


class MiniCPMModel(nn.Module):
    def __init__(self, vision: nn.Module, resampler: nn.Module, decoder: nn.Module, embed_tokens: nn.Embedding, scale_emb: float = 1.0):
        super().__init__()
        self.vision = vision
        self.resampler = resampler
        self.decoder = decoder
        self.embed_tokens = embed_tokens
        self.scale_emb = scale_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        tgt_sizes: Optional[torch.Tensor] = None,
        image_bound: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        vision -> resampler -> scatter -> decoder.
        Assumes batch=1; shapes must be pre-padded by caller.
        """
        text_embeds = self.embed_tokens(input_ids)
        text_embeds = text_embeds * self.scale_emb
        inputs_embeds = text_embeds

        if pixel_values is not None and image_bound is not None:
            vision_hidden = self.vision(pixel_values, tgt_sizes=tgt_sizes)
            vision_tokens = self.resampler(vision_hidden, tgt_sizes=tgt_sizes)
            vision_tokens = vision_tokens.to(text_embeds.dtype)
            inputs_embeds = scatter_vision_tokens(text_embeds, vision_tokens, image_bound)

        logits, new_kvs = self.decoder(inputs_embeds, past_key_values)
        return {"logits": logits, "past_key_values": new_kvs}


def build_from_hf(hf_model: nn.Module) -> MiniCPMModel:
    """
    构建自研模型，复用 HF 权重，但前向由我们控制。
    """
    cfg = hf_model.config
    # 判断是否包含视觉模块
    has_vision = hasattr(hf_model, "vpm") or hasattr(hf_model, "vision_model")
    vision = None
    resampler = None
    if has_vision:
        vpm = hf_model.vpm if hasattr(hf_model, "vpm") else hf_model.vision_model
        vision = SiglipVisionExport.from_hf(vpm)
        resampler = ResamplerExport.from_hf(hf_model.resampler, num_heads=cfg.vision_config.num_attention_heads)
    # 语言塔直接使用 Qwen2 导出实现（MiniCPM 背骨即 Qwen2）
    hf_llm = hf_model.llm if hasattr(hf_model, "llm") else hf_model
    decoder, embed_tokens, scale_emb = build_llm_from_hf(hf_llm)
    return MiniCPMModel(vision, resampler, decoder, embed_tokens, scale_emb=scale_emb)
