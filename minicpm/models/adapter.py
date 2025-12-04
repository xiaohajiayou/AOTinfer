"""
MiniCPM Adapter：
- 提供 MiniCPMAdapter，加载 HF 模型并拆出三段子模块（vision_resampler / embed_scatter / llm）
"""

from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from minicpm.models.model import build_from_hf
from minicpm.models.vision import SiglipVisionExport
from minicpm.models.resampler import ResamplerExport
from minicpm.models.llm import build_llm_from_hf
from minicpm.models.scatter import scatter_vision_tokens


class VisionResamplerWrapper(nn.Module):
    """将 SiglipVisionExport + ResamplerExport 合并，便于导出/调用。"""

    def __init__(self, vision: nn.Module, resampler: nn.Module):
        super().__init__()
        self.vision = vision
        self.resampler = resampler

    def forward(self, pixel_values: torch.Tensor, tgt_sizes: torch.Tensor):
        hidden = self.vision(pixel_values, tgt_sizes=tgt_sizes)
        tokens = self.resampler(hidden, tgt_sizes=tgt_sizes)
        return tokens


class MiniCPMEmbedScatter(nn.Module):
    """文本 embedding + 可选 scatter（无图则原样返回）。"""

    def __init__(self, embed_tokens: nn.Embedding, scale_emb: float = 1.0):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.scale_emb = scale_emb

    def forward(
        self,
        input_ids: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None,
        image_bound: Optional[torch.Tensor] = None,
    ):
        embeds = self.embed_tokens(input_ids) * self.scale_emb
        if vision_tokens is not None and image_bound is not None:
            embeds = scatter_vision_tokens(embeds, vision_tokens.to(embeds.dtype), image_bound)
        return embeds


class MiniCPMAdapter:
    """
    加载 HF MiniCPM，拆出三段子模块，供导出或 wrapper 使用。
    """

    def __init__(self, hf_model: nn.Module, tokenizer: Optional[AutoTokenizer] = None):
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.config = getattr(hf_model, "config", None)
        # 子模块占位
        self.vision_resampler: Optional[nn.Module] = None
        self.embed_scatter: Optional[nn.Module] = None
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
            local_files_only=local_files_only,
            init_vision=True,
            init_audio=False,
            init_tts=False,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=local_files_only)
        adapter = cls(hf_model, tokenizer)
        adapter.build_submodules()
        return adapter

    def build_submodules(self):
        # 视觉/Resampler
        if hasattr(self.hf_model, "vpm") or hasattr(self.hf_model, "vision_model"):
            vpm = self.hf_model.vpm if hasattr(self.hf_model, "vpm") else self.hf_model.vision_model
            vision_export = SiglipVisionExport.from_hf(vpm)
            res_export = ResamplerExport.from_hf(
                self.hf_model.resampler, num_heads=self.hf_model.config.vision_config.num_attention_heads
            )
            self.vision_resampler = VisionResamplerWrapper(vision_export, res_export)
        # LLM
        hf_llm = self.hf_model.llm if hasattr(self.hf_model, "llm") else self.hf_model
        decoder, embed_tokens, scale_emb = build_llm_from_hf(hf_llm)
        self.embed_scatter = MiniCPMEmbedScatter(embed_tokens, scale_emb=scale_emb)
        self.llm = decoder
        self.scale_emb = scale_emb

    def get_export_modules(self):
        """
        返回三段子模块 (vision_resampler, embed_scatter, llm) 供导出使用。
        """
        if self.vision_resampler is None or self.embed_scatter is None or self.llm is None:
            self.build_submodules()
        return self.vision_resampler, self.embed_scatter, self.llm
