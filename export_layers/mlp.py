import torch
from torch import nn


class ExportMLP(nn.Module):
    """
    通用的 gated MLP：down_proj(act(gate_proj(x)) * up_proj(x))
    """

    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = getattr(torch.nn.functional, activation)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        down = self.down_proj(gated * up)
        return down
