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
    input_ids = torch.zeros((1, 128), dtype=torch.int32, device=torch_device)

    vision_tokens = torch.zeros(
        (1, 0, module.embed_tokens.weight.shape[1]), dtype=torch_dtype, device=torch_device
    )
    image_bound = torch.zeros((1, 0, 2), dtype=torch.int64, device=torch_device)

    T = Dim("seq_len")  # 显式定义动态维度
    V = Dim("num_vision", min=0, max=64)
    B = Dim("num_bounds", min=0, max=64)
    exported = torch.export.export(
        module,
        (input_ids, vision_tokens, image_bound),
        dynamic_shapes=[
            {1: T},  # input_ids
            {1: V},  # vision_tokens length
            {1: B},  # image_bound count
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



def export_llm(module, adapter, out_dir: str, args):
    torch_dtype = getattr(torch, args.dtype)  # 如 "float32" → torch.float32
    torch_device = torch.device(args.device)  # 如 "cuda:0" → torch.device("cuda:0")
    hidden_size = module.lm_head.weight.shape[1]
    num_layers = len(module.layers)
    num_kv_heads = module.num_kv_heads
    head_dim = module.head_dim
    
    # example inputs
    example_cache_len = args.example_cache_len
    example_seq = args.example_seq
    inputs_embeds = torch.zeros((1, example_seq, hidden_size), device=torch_device, dtype=torch_dtype)
    key_cache = adapter.init_kv_cache(1, example_cache_len, torch_dtype, torch_device)
    value_cache = adapter.init_kv_cache(1, example_cache_len, torch_dtype, torch_device)
    cache_len = torch.tensor(example_cache_len, dtype=torch.long, device=args.device)

    config = adapter.config
    max_pos = getattr(config, "max_position_embeddings", 32768)
    N_dim = torch.export.Dim("N_text", min=1, max=max_pos)
    cache_dim = torch.export.Dim("cache_len", min=0, max=max_pos)
    dynamic_shapes = (
        {1: N_dim},                 # inputs_embeds
        [{2: cache_dim} for _ in range(num_layers)],  # key_cache
        [{2: cache_dim} for _ in range(num_layers)],  # value_cache
        {}                          # cache_len
    )
    
    # ========== 导出逻辑 ==========
    exported = torch.export.export(
        module,
        (inputs_embeds, key_cache, value_cache, cache_len),
        dynamic_shapes=dynamic_shapes
    )
    
    os.makedirs(out_dir, exist_ok=True)
    package_path = os.path.join(out_dir, "minicpm_llm.pt2")
    print(f"[export] llm compiling to {package_path}")

    torch._inductor.aoti_compile_and_package(
        exported, 
        package_path=package_path
    )
    
    # 元数据
    meta = {"num_layers": num_layers, "num_kv_heads": num_kv_heads, "head_dim": head_dim, "hidden_size": hidden_size}
    with open(os.path.join(out_dir, "minicpm_export_meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"[export] meta -> {os.path.join(out_dir, 'minicpm_export_meta.json')}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/home/liwenxiao/models/minicpm_o_2_6")
    parser.add_argument("--out-dir", type=str, default=os.path.dirname(__file__))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--example_cache_len", type=int, default=16)
    parser.add_argument("--example_seq", type=int, default=4, help="example text tokens for export")
    args = parser.parse_args()
    torch_dtype = getattr(torch, args.dtype)  # 如 "float32" → torch.float32
    torch_device = torch.device(args.device)  # 如 "cuda:0" → torch.device("cuda:0")

    adapter = MiniCPMAdapter.from_pretrained(args.model_dir, device=torch_device, dtype=torch_dtype)
    vision_resampler, embed_scatter, llm = adapter.get_export_modules()

    # export_vision_resampler(vision_resampler.to(torch_device, torch_dtype), args.out_dir, args)
    export_embed_scatter(embed_scatter.to(torch_device, torch_dtype), args.out_dir, args)
    # export_llm(llm.to(torch_device, torch_dtype), adapter, args.out_dir, args)


if __name__ == "__main__":
    main()
