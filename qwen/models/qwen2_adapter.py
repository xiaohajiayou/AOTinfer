from __future__ import annotations

from typing import List

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from qwen.models.qwen2_block import Qwen2ExportModel


class Qwen2Adapter:
    """
    构建 export-friendly Qwen2 模型，并提供 KV cache 初始化工具。
    """

    def __init__(self, hf_model: nn.Module):
        self.hf_model = hf_model
        self.config = hf_model.config
        self.export_model = Qwen2ExportModel(hf_model).to(next(hf_model.parameters()).device)

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).eval()
        return cls(hf_model)

    def init_kv_cache(
        self,
        batch_size: int,
        cache_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        device = torch.device(device)
        num_layers = len(self.export_model.layers)
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        cache = []
        for _ in range(num_layers):
            cache.append(
                torch.zeros(
                    batch_size,
                    num_kv_heads,
                    cache_len,
                    head_dim,
                    dtype=dtype,
                    device=device,
                )
            )
        return cache
