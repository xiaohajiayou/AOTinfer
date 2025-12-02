"""
Export MiniCPM using HF submodules + fixed forward for torch.export.
Exports two graphs: text-only and multimodal.
"""

import os
import torch
from transformers import AutoModel

from minicpm.models.model import build_from_hf

# constraints
T_MAX = 4096
S_MAX = 10
H = W = 980  # 14 * 70


def build_export_model(model_dir: str, device: str, dtype: torch.dtype):
    hf_model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="eager",
    ).eval()
    return build_from_hf(hf_model).to(device)


def export_text(model, out_dir: str):
    T = min(16, T_MAX)
    input_ids = torch.zeros((1, T), dtype=torch.long, device=next(model.parameters()).device)
    attention_mask = torch.ones((1, T), dtype=torch.bool, device=input_ids.device)
    traced = torch.export.export(
        model,
        (input_ids, attention_mask, None, None, None, None, None),
    )
    path = os.path.join(out_dir, "minicpm_text.pt2")
    traced.save(path)
    print(f"[export] text-only saved to {path}")


def export_multimodal(model, out_dir: str):
    T = min(16, T_MAX)
    device = next(model.parameters()).device
    input_ids = torch.zeros((1, T), dtype=torch.long, device=device)
    attention_mask = torch.ones((1, T), dtype=torch.bool, device=device)
    pixel_values = torch.zeros((S_MAX, 3, H, W), dtype=torch.float32, device=device)
    tgt_sizes = torch.zeros((S_MAX, 2), dtype=torch.int64, device=device)
    image_bound = torch.zeros((1, S_MAX, 2), dtype=torch.long, device=device)
    image_bound[0, 0] = torch.tensor([0, 64], device=device)

    traced = torch.export.export(
        model,
        (input_ids, attention_mask, pixel_values, tgt_sizes, image_bound, None, None),
    )
    path = os.path.join(out_dir, "minicpm_mm.pt2")
    traced.save(path)
    print(f"[export] multimodal saved to {path}")


def main():
    model_dir = "/home/liwenxiao/models/minicpm-o-2.6"
    out_dir = os.path.dirname(__file__)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = build_export_model(model_dir, device, dtype)
    export_text(model, out_dir)
    export_multimodal(model, out_dir)


if __name__ == "__main__":
    main()
