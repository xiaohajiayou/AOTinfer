from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForVision2Seq

from qwen2_5_vl.models.qwen2_5_vl_block import Qwen2_5_VLExportModel


class Qwen2_5_VLAdapter:
    """
    Wrap HF Qwen2.5-VL to an export-friendly model.
    Vision encoder is expected to be handled externally; we ingest projected vision tokens.
    """

    def __init__(self, hf_model: nn.Module):
        self.hf_model = hf_model
        text_config = getattr(hf_model, "config", None)
        if hasattr(text_config, "text_config"):
            text_config = text_config.text_config
        self.config = text_config
        self.export_model = Qwen2_5_VLExportModel(hf_model).to(next(hf_model.parameters()).device)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        model_class: Optional[str] = None,
    ):
        hf_model = None
        if model_class == "vision2seq":
            hf_model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                device_map=device,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).eval()
        else:
            try:
                hf_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                ).eval()
            except Exception:
                hf_model = AutoModelForVision2Seq.from_pretrained(
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
