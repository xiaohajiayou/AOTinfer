"""
Reusable tensor-only building blocks for export-friendly transformer models.

These modules intentionally avoid any torch.export unfriendly patterns such as
DynamicCache, masking_utils, or data-dependent Python control flow.
"""

from .embedding import ExportEmbedding
from .mask import build_causal_mask
from .rotary import ExportRotaryEmbedding, apply_rotary_pos_emb
from .utils import repeat_kv
from .norm import ExportRMSNorm
from .mlp import ExportMLP
from .attention import ExportAttention, ExportAttentionConfig

__all__ = [
    "ExportEmbedding",
    "build_causal_mask",
    "ExportRotaryEmbedding",
    "apply_rotary_pos_emb",
    "repeat_kv",
    "ExportRMSNorm",
    "ExportMLP",
    "ExportAttention",
    "ExportAttentionConfig",
]
