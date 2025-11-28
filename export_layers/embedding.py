import torch
from torch import nn


class ExportEmbedding(nn.Module):
    """
    Thin wrapper around nn.Embedding that optionally applies a scaling factor.

    Some models（例如 Gemma/Qwen）在推理时需要对嵌入乘一个常数，
    这里把逻辑统一起来，方便不同模型的适配器复用。
    """

    def __init__(self, embed: nn.Embedding, embed_scale: float = 1.0):
        super().__init__()
        self.embed = embed
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch, seq] 的整型张量
        Returns:
            embeddings: [batch, seq, hidden]
        """
        hidden = self.embed(input_ids)
        if self.embed_scale != 1.0:
            hidden = hidden * self.embed_scale
        return hidden
