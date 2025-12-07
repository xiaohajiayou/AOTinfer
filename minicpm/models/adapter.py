"""
MiniCPM Adapter：
- 提供 MiniCPMAdapter，加载 HF 模型并拆出子模块：vision_resampler / embed / llm
"""

from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


from minicpm.models.vision import SiglipVisionExport
from minicpm.models.resampler import ResamplerExport
from minicpm.models.llm import build_llm_from_hf
from minicpm.models.scatter import scatter_vision_tokens


class VisionResamplerWrapper(nn.Module):
    """SigLIP 视觉塔 + Resampler 合并，便于导出。"""

    def __init__(self, vision: nn.Module, resampler: nn.Module):
        super().__init__()
        self.vision = vision
        self.resampler = resampler

    def forward(self, pixel_values: torch.Tensor, tgt_sizes: torch.Tensor):
        hidden = self.vision(pixel_values, tgt_sizes=tgt_sizes)
        return self.resampler(hidden, tgt_sizes=tgt_sizes)





class MiniCPMAdapter:
    """
    加载 HF MiniCPM，拆出子模块：vision_resampler / embed / llm
    """

    def __init__(self, hf_model: nn.Module, tokenizer: Optional[AutoTokenizer] = None):
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.config = getattr(hf_model, "config", None)

        self.vision_resampler: Optional[nn.Module] = None
        self.embed: Optional[nn.Module] = None
        self.llm: Optional[nn.Module] = None
        self.scale_emb: float = 1.0

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        local_files_only: bool = True,
    ):
        hf_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device,
            # attn_implementation="eager",
            local_files_only=local_files_only,
            init_vision=True,
            init_audio=False,
            init_tts=False,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=local_files_only
        )
        adapter = cls(hf_model, tokenizer)
        adapter.build_submodules()
        return adapter

    def build_submodules(self):
        # 视觉塔 + Resampler
        if hasattr(self.hf_model, "vpm") or hasattr(self.hf_model, "vision_model"):
            vpm = self.hf_model.vpm if hasattr(self.hf_model, "vpm") else self.hf_model.vision_model
            vision_export = SiglipVisionExport.from_hf(vpm)
            res_export = ResamplerExport.from_hf(
                self.hf_model.resampler, num_heads=self.hf_model.config.vision_config.num_attention_heads
            )
            self.vision_resampler = VisionResamplerWrapper(vision_export, res_export)

        # LLM 与 embedding
        if hasattr(self.hf_model, "llm"):
            hf_llm = self.hf_model.llm 
            model = build_llm_from_hf(hf_llm)
            self.embed = hf_llm.model.embed_tokens
            self.llm = model

    def get_export_modules(self):
        """
        返回 (vision_resampler, embed, llm)，供导出或 wrapper 使用。
        """
        if self.vision_resampler is None or self.embed is None or self.llm is None:
            self.build_submodules()
        return self.vision_resampler, self.embed, self.llm

    def init_kv_cache(
        self,
        batch_size: int,
        cache_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        device = torch.device(device)
        # if self.llm is None:
        #     self.build_submodules()
        num_layers = len(self.llm.layers)
        num_kv_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        cache = []
        for _ in range(num_layers):
            cache.append(
                torch.zeros(
                    (batch_size, num_kv_heads, cache_len, head_dim),
                    dtype=dtype,
                    device=device,
                )
            )
        return cache
