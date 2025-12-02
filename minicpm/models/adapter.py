"""
Adapter wrapper for MiniCPM, aligned with qwen2_5_vl style.
Provides a class interface and a scatter helper.

TODO: wire to actual HF MiniCPM model and export-friendly implementation.
"""

from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


def scatter_vision_tokens(
    text_embeds: torch.Tensor,
    vision_tokens: torch.Tensor,
    image_bound: torch.Tensor,
    max_tokens: Optional[int] = None,
) -> torch.Tensor:
    """
    Replace placeholder positions in text embeddings with vision tokens.

    Args:
        text_embeds: [B, T, H]
        vision_tokens: [N, L, H] where N = num_slices (flattened across batch)
        image_bound: [B, K, 2] start/end indices per slice (exclusive end), padded with zeros
        max_tokens: optional cap on how many vision tokens to consume (for safety)
    Returns:
        new_embeds: [B, T, H]
    """
    new_embeds = text_embeds.clone()
    B, _, H = text_embeds.shape
    _, K, _ = image_bound.shape

    vidx = 0
    for b in range(B):
        for k in range(K):
            start, end = image_bound[b, k]
            if end <= start:
                continue  # padded
            length = int(end - start)
            if max_tokens is not None:
                length = min(length, max_tokens)
            if length <= 0:
                continue
            if vidx >= vision_tokens.shape[0]:
                break
            vt = vision_tokens[vidx]
            vidx += 1
            if vt.shape[0] < length:
                pad_len = length - vt.shape[0]
                vt = torch.cat([vt, torch.zeros(pad_len, H, device=vt.device, dtype=vt.dtype)], dim=0)
            new_embeds[b, start:end] = vt[:length]
    return new_embeds


class MiniCPMAdapter:
    """
    Wrap HF MiniCPM into an export-friendly interface (placeholder skeleton).
    Pattern mirrors qwen2_5_vl.models.adapter.Qwen2_5_VLAdapter.
    """

    def __init__(self, hf_model: nn.Module, export_model: Optional[nn.Module] = None):
        self.hf_model = hf_model
        self.config = getattr(hf_model, "config", None)
        self.export_model = export_model  # will be MiniCPMModel instance

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        hf_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        adapter = cls(hf_model)
        adapter.tokenizer = tokenizer
        return adapter

    def init_kv_cache(
        self,
        batch_size: int,
        cache_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> List[torch.Tensor]:
        # Placeholder; real impl depends on MiniCPM block structure
        num_layers = getattr(self.config, "num_hidden_layers", 0)
        num_kv_heads = getattr(self.config, "num_key_value_heads", 0)
        head_dim = getattr(self.config, "hidden_size", 0) // getattr(self.config, "num_attention_heads", 1)
        cache = []
        for _ in range(num_layers):
            cache.append(torch.zeros(batch_size, num_kv_heads, cache_len, head_dim, dtype=dtype, device=device))
        return cache

    def build_export_model(self, vision: nn.Module, resampler: nn.Module, llm: nn.Module):
        # 延迟导入以避免循环引用
        from minicpm.models.model import MiniCPMModel

        self.export_model = MiniCPMModel(vision, resampler, llm, hidden_size=getattr(self.config, "hidden_size", 0))
        return self.export_model
