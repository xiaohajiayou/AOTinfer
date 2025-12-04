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
import sys
import torch
from torch.export import Dim

# add project root to sys.path so `minicpm` can be imported when running directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from minicpm.models.adapter import MiniCPMAdapter

# 约束（符号维度）
T = Dim("seq_len")
S = Dim("num_slices")
K = Dim("num_slices")
C = Dim("hidden_size")


def export_vision_resampler(module, out_dir: str, args):
    torch_dtype = getattr(torch, args.dtype)  # 如 "float32" → torch.float32
    torch_device = torch.device(args.device)  # 如 "cuda:0" → torch.device("cuda:0")
    device_str = args.device  # 设备字符串（用于编译选项）
    # 固定前处理当前网格：25x41 patch => 输入分辨率 350x574
    pixel_values = torch.zeros((1, 3, 350, 574), dtype=torch_dtype, device=torch_device)
    tgt_sizes = torch.tensor([[25, 41]], dtype=torch.int64, device=torch_device)
    
    # ========== 导出逻辑 ==========
    exported = torch.export.export(
        module,
        (pixel_values, tgt_sizes),
    )
    os.makedirs(out_dir, exist_ok=True)
    package_path = os.path.join(out_dir, "minicpm_vision_resampler.pt2")
    print(f"[export] vision_resampler compiling to {package_path}")

    # ========== 修复 2：编译选项按设备适配 ==========
    compile_options = {
        "device": device_str,
        "dtype": torch_dtype,
        "disable_mixed_mm": True if device_str == "cpu" else False,
        "cpu_fallback": True if device_str == "cpu" else False,
        "triton": False if device_str == "cpu" else True,
        "disable_cudagraphs": True if device_str == "cpu" else False,
    }
    torch._inductor.aoti_compile_and_package(
        exported, 
        package_path=package_path,
        # options=compile_options
    )


def export_embed_scatter(module, out_dir: str, args):
    torch_dtype = getattr(torch, args.dtype)  # 如 "float32" → torch.float32
    torch_device = torch.device(args.device)  # 如 "cuda:0" → torch.device("cuda:0")
    device_str = args.device  # 设备字符串（用于编译选项）
    input_ids = torch.zeros((1, 4), dtype=torch.int64, device=torch_device)
    vision_tokens = torch.zeros(
        (1, 64, module.embed_tokens.weight.shape[1]), dtype=torch_dtype, device=torch_device
    )
    image_bound = torch.zeros((1, 1, 2), dtype=torch.int64, device=torch_device)

    T = Dim("seq_len")  # 显式定义动态维度
    exported = torch.export.export(
        module,
        (input_ids, vision_tokens, image_bound, True),
        # use positional list to avoid requiring top-level keys for unused args
        dynamic_shapes=[
            {1: T},  # input_ids
            None,    # vision_tokens
            None,    # image_bound
            None,    # export_mode flag
        ],
    )
    os.makedirs(out_dir, exist_ok=True)
    package_path = os.path.join(out_dir, "minicpm_embed_scatter.pt2")
    print(f"[export] embed_scatter compiling to {package_path}")
    # ========== 修复 2：编译选项按设备适配 ==========
    compile_options = {
        "device": device_str,
        "dtype": torch_dtype,
        "disable_mixed_mm": True if device_str == "cpu" else False,
        "cpu_fallback": True if device_str == "cpu" else False,
        "triton": False if device_str == "cpu" else True,
        "disable_cudagraphs": True if device_str == "cpu" else False,
    }
    torch._inductor.aoti_compile_and_package(
        exported, 
        package_path=package_path,
        # options=compile_options
    )



def export_llm(module, out_dir: str, args):
    torch_dtype = getattr(torch, args.dtype)  # 如 "float32" → torch.float32
    torch_device = torch.device(args.device)  # 如 "cuda:0" → torch.device("cuda:0")
    device_str = args.device  # 设备字符串（用于编译选项）
    hidden_size = module.lm_head.weight.shape[1]
    inputs_embeds = torch.zeros((1, 4, hidden_size), device=torch_device, dtype=torch_dtype)
    num_layers = len(module.layers)
    num_kv_heads = module.num_kv_heads
    head_dim = module.head_dim
    past_key_values = tuple(
        (
            torch.zeros(1, num_kv_heads, 2, head_dim, device=torch_device, dtype=torch_dtype),
            torch.zeros(1, num_kv_heads, 2, head_dim, device=torch_device, dtype=torch_dtype),
        )
        for _ in range(num_layers)
    )
    T = Dim("seq_len")
    cache_len = Dim("cache_len")
    # 为每一层 KV 指定动态 cache_len
    past_kv_dyn = tuple(({2: cache_len}, {2: cache_len}) for _ in range(num_layers))
    
    # ========== 导出逻辑 ==========
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
    # ========== 修复 2：编译选项按设备适配 ==========
    compile_options = {
        "device": device_str,
        "dtype": torch_dtype,
        "disable_mixed_mm": True if device_str == "cpu" else False,
        "cpu_fallback": True if device_str == "cpu" else False,
        "triton": False if device_str == "cpu" else True,
        "disable_cudagraphs": True if device_str == "cpu" else False,
    }
    torch._inductor.aoti_compile_and_package(
        exported, 
        package_path=package_path,
        # options=compile_options
    )
    
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
    torch_dtype = getattr(torch, args.dtype)  # 如 "float32" → torch.float32
    torch_device = torch.device(args.device)  # 如 "cuda:0" → torch.device("cuda:0")

    adapter = MiniCPMAdapter.from_pretrained(args.model_dir, device=torch_device, dtype=torch_dtype)
    vision_resampler, embed_scatter, llm = adapter.get_export_modules()

    export_vision_resampler(vision_resampler.to(torch_device, torch_dtype), args.out_dir, args)
    export_embed_scatter(embed_scatter.to(torch_device, torch_dtype), args.out_dir, args)
    export_llm(llm.to(torch_device, torch_dtype), args.out_dir, args)


if __name__ == "__main__":
    main()
