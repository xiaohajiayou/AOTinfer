"""
三段导出（支持动态维度）：
- vision_resampler: pixel_values [S,3,patch_h,patch_w_flat], tgt_sizes [S,2]
- embed_scatter: input_ids [B,T], vision_tokens [S,64,H], image_bound [B,K,2]
- llm: inputs_embeds [B,T,H], past_kv 动态 cache_len
导出同时写出元数据（num_layers/num_kv_heads/head_dim）供 runner 初始化 KV。
"""

import os
import json
import argparse
import torch
from torch.export import Dim

from minicpm.models.adapter import MiniCPMAdapter

# 约束（符号维度）
T = Dim("seq_len")
S = Dim("num_slices")
K = Dim("num_slices")
C = Dim("hidden_size")


def export_vision_resampler(module, out_dir: str):
    device = next(module.parameters()).device
    # 固定前处理当前网格：25x41 patch => 输入分辨率 350x574
    pixel_values = torch.zeros((1, 3, 350, 574), dtype=torch.float32, device=device)
    tgt_sizes = torch.tensor([[25, 41]], dtype=torch.int32, device=device)
    exported = torch.export.export(module, (pixel_values, tgt_sizes))
    os.makedirs(out_dir, exist_ok=True)
    package_path = os.path.join(out_dir, "minicpm_vision_resampler.pt2")
    print(f"[export] vision_resampler compiling to {package_path}")
    torch._inductor.aoti_compile_and_package(exported, package_path=package_path)


def export_embed_scatter(module, out_dir: str):
    device = next(module.parameters()).device
    input_ids = torch.zeros((1, 4), dtype=torch.long, device=device)
    vision_tokens = torch.zeros(
        (1, 64, module.embed_tokens.weight.shape[1]), dtype=module.embed_tokens.weight.dtype, device=device
    )
    image_bound = torch.zeros((1, 1, 2), dtype=torch.long, device=device)

    exported = torch.export.export(
        module,
        (input_ids, vision_tokens, image_bound, True),
        dynamic_shapes={
            "input_ids": {1: T},
        },
    )
    os.makedirs(out_dir, exist_ok=True)
    package_path = os.path.join(out_dir, "minicpm_embed_scatter.pt2")
    print(f"[export] embed_scatter compiling to {package_path}")
    torch._inductor.aoti_compile_and_package(exported, package_path=package_path)



def export_llm(module, out_dir: str):
    device = next(module.parameters()).device
    hidden_size = module.lm_head.weight.shape[1]
    inputs_embeds = torch.zeros((1, 4, hidden_size), device=device, dtype=module.lm_head.weight.dtype)
    num_layers = len(module.layers)
    num_kv_heads = module.num_kv_heads
    head_dim = module.head_dim
    past_key_values = tuple(
        (
            torch.zeros(1, num_kv_heads, 2, head_dim, device=device, dtype=inputs_embeds.dtype),
            torch.zeros(1, num_kv_heads, 2, head_dim, device=device, dtype=inputs_embeds.dtype),
        )
        for _ in range(num_layers)
    )
    # 为每一层 KV 指定动态 cache_len
    past_kv_dyn = tuple(({2: Dim("cache_len")}, {2: Dim("cache_len")}) for _ in range(num_layers))
    exported = torch.export.export(
        module,
        (inputs_embeds, past_key_values),
        dynamic_shapes={
            "inputs_embeds": {1: T},
            "past_key_values": past_kv_dyn if num_layers > 0 else None,
        },
    )
    os.makedirs(out_dir, exist_ok=True)
    package_path = os.path.join(out_dir, "minicpm_llm.pt2")
    print(f"[export] llm compiling to {package_path}")
    torch._inductor.aoti_compile_and_package(exported, package_path=package_path)
    
    # 元数据
    meta = {"num_layers": num_layers, "num_kv_heads": num_kv_heads, "head_dim": head_dim, "hidden_size": hidden_size}
    with open(os.path.join(out_dir, "minicpm_export_meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"[export] meta -> {os.path.join(out_dir, 'minicpm_export_meta.json')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=str, default="/home/liwenxiao/models/minicpm_o_2_6")
    ap.add_argument("--out-dir", type=str, default=os.path.dirname(__file__))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    adapter = MiniCPMAdapter.from_pretrained(args.model_dir, device=args.device, dtype=dtype)
    vision_resampler, embed_scatter, llm = adapter.get_export_modules()

    # export_vision_resampler(vision_resampler.to(args.device), args.out_dir)
    export_embed_scatter(embed_scatter.to(args.device), args.out_dir)
    # export_llm(llm.to(args.device), args.out_dir)


if __name__ == "__main__":
    main()
