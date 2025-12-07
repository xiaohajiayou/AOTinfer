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
from transformers import AutoTokenizer
from minicpm.models.adapter import MiniCPMAdapter

class Minicpm_Qwen2AOTIWrapper(torch.nn.Module):
    def __init__(self, export_model):
        super().__init__()
        self.export_model = export_model
        self.num_layers = len(export_model.layers)

    def forward(self, *flat_inputs):
        input_embed = flat_inputs[0]
        cache_inputs = flat_inputs[1:]
        key_cache = list(cache_inputs[: self.num_layers])
        value_cache = list(cache_inputs[self.num_layers : 2 * self.num_layers])
        cache_len = cache_inputs[-1]
        logits, new_key, new_value, new_cache_len = self.export_model(
            input_embed, key_cache, value_cache, cache_len
        )
        return (logits, *new_key, *new_value, new_cache_len)

def flatten_inputs(input_ids, key_cache, value_cache, cache_len):
    flat = [input_ids]
    flat.extend(key_cache)
    flat.extend(value_cache)
    flat.append(cache_len)
    return tuple(flat)


def unflatten_outputs(outputs, num_layers):
    logits = outputs[0]
    key_cache = list(outputs[1 : 1 + num_layers])
    value_cache = list(outputs[1 + num_layers : 1 + 2 * num_layers])
    cache_len = outputs[-1]
    return logits, key_cache, value_cache, cache_len


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


def export_embed(module, out_dir: str, args):
    torch_dtype = getattr(torch, args.dtype)
    torch_device = torch.device(args.device)

    input_ids = torch.zeros((1, args.example_seq), dtype=torch.long, device=torch_device)
    T = Dim("seq_len")

    exported = torch.export.export(
        module,
        (input_ids,),
        dynamic_shapes=({1: T},),
    )

    os.makedirs(out_dir, exist_ok=True)
    package_path = os.path.join(out_dir, "minicpm_embed.pt2")
    print(f"[export] embed compiling to {package_path}")
    torch._inductor.aoti_compile_and_package(exported, package_path=package_path)




def export_llm(llm, adapter, out_dir: str, args):
    # wrapper = Minicpm_Qwen2AOTIWrapper(llm).to(args.device)
    wrapper = llm
    torch_dtype = getattr(torch, args.dtype)  # 如 "float32" → torch.float32
    torch_device = torch.device(args.device)  # 如 "cuda:0" → torch.device("cuda:0")
    hidden_size = llm.lm_head.weight.shape[1]
    num_layers = len(llm.layers)
    num_kv_heads = llm.num_kv_heads
    head_dim = llm.head_dim
    
    # example inputs
    example_cache_len = args.example_cache_len
    example_seq = args.example_seq
    inputs_embeds = torch.randint(
        0,
        adapter.config.max_position_embeddings,
        (1, example_seq, hidden_size),
        dtype=torch_dtype,
        device=torch_device
    )
    # inputs_embeds = torch.zeros((1, example_seq, hidden_size), device=torch_device, dtype=torch_dtype)
    key_cache = adapter.init_kv_cache(1, example_cache_len, adapter.embed.weight.dtype, device=args.device)
    value_cache = adapter.init_kv_cache(1, example_cache_len, adapter.embed.weight.dtype, device=args.device)
    cache_len_tensor = torch.tensor(example_cache_len, dtype=torch.long, device=args.device)
    # flat_example = flatten_inputs(inputs_embeds, key_cache, value_cache, cache_len_tensor)

    N_dim = torch.export.Dim("N", min=1, max=adapter.config.max_position_embeddings)
    cache_dim = torch.export.Dim("cache_len", min=0, max=adapter.config.max_position_embeddings)
    # per_input_shapes = []
    # per_input_shapes.append({1: N_dim})
    # for _ in range(num_layers):
    #     per_input_shapes.append({2: cache_dim})
    # for _ in range(num_layers):
    #     per_input_shapes.append({2: cache_dim})
    # per_input_shapes.append({})
    # dynamic_shapes = {"flat_inputs": tuple(per_input_shapes)}
    
    dynamic_shapes = (
        {1: N_dim},                 # inputs_embeds
        [{2: cache_dim} for _ in range(num_layers)],  # key_cache
        [{2: cache_dim} for _ in range(num_layers)],  # value_cache
        {}                          # cache_len
    )
    
    # ========== 导出逻辑 ==========
    exported = torch.export.export(
        wrapper,
        (inputs_embeds, key_cache, value_cache, cache_len_tensor),
        dynamic_shapes=dynamic_shapes
    )
    # exported = torch.export.export(
    #     wrapper,
    #     flat_example,
    #     dynamic_shapes=dynamic_shapes
    # )
    
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

def verify_llm(args, adapter, llm, package_path):
    # wrapper = Minicpm_Qwen2AOTIWrapper(llm).to(args.device)
    wrapper = llm
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    compiled = torch._inductor.aoti_load_package(package_path, device_index=args.device_index)
    loader = compiled.loader
    num_layers = len(llm.layers)
    dtype = adapter.embed.weight.dtype

    def run_wrapper(inputs_embeds, key_cache, value_cache, cache_len):
        # out = wrapper(*flatten_inputs(inputs_embeds, key_cache, value_cache, cache_len))
        # return unflatten_outputs(out, num_layers)
        return wrapper(inputs_embeds, key_cache, value_cache, cache_len)

    def run_loader(inputs_embeds, key_cache, value_cache, cache_len):
        # out = loader.run(flatten_inputs(inputs_embeds, key_cache, value_cache, cache_len))
        # return unflatten_outputs(out, num_layers)
        return compiled(inputs_embeds, key_cache, value_cache, cache_len)

    prompt = args.prompt
    tok = tokenizer(prompt, return_tensors="pt").to(args.device)
    key_cache_w = adapter.init_kv_cache(1, 0, dtype, args.device)
    value_cache_w = adapter.init_kv_cache(1, 0, dtype, args.device)
    key_cache_l = [t.clone() for t in key_cache_w]
    value_cache_l = [t.clone() for t in value_cache_w]
    cache_len_w = torch.tensor(0, dtype=torch.long, device=args.device)
    cache_len_l = cache_len_w.clone()
    inputs_embeds = adapter.embed(tok["input_ids"])
    logits_w, key_cache_w, value_cache_w, cache_len_w = run_wrapper(
        inputs_embeds, key_cache_w, value_cache_w, cache_len_w
    )
    logits_l, key_cache_l, value_cache_l, cache_len_l = run_loader(
        inputs_embeds, key_cache_l, value_cache_l, cache_len_l
    )
    compare_logits(logits_w[:, -1, :], logits_l[:, -1, :], step="prefill")

    next_token = torch.argmax(logits_w[:, -1, :], dim=-1, keepdim=True)

    for step in range(args.decode_steps):
        next_embed = adapter.embed(next_token)
        logits_w, key_cache_w, value_cache_w, cache_len_w = run_wrapper(
            next_embed, key_cache_w, value_cache_w, cache_len_w
        )
        logits_l, key_cache_l, value_cache_l, cache_len_l = run_loader(
            next_embed, key_cache_l, value_cache_l, cache_len_l
        )
        compare_logits(logits_w[:, -1, :], logits_l[:, -1, :], step=f"decode-{step}")
        next_token = torch.argmax(logits_w[:, -1, :], dim=-1, keepdim=True)


def compare_logits(logits_a, logits_b, step):
    diff = (logits_a.float() - logits_b.float()).abs().max().item()
    arg_a = torch.argmax(logits_a, dim=-1).item()
    arg_b = torch.argmax(logits_b, dim=-1).item()
    print(
        f"[{step}] max_abs_diff={diff:.3e}, arg_wrapper={arg_a}, arg_aoti={arg_b}, equal={arg_a == arg_b}"
    )

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-dir", type=str, default="/home/liwenxiao/models/minicpm_o_2_6")
    parser.add_argument("--model-dir", type=str, default="/root/autodl-tmp/models/MiniCPM_o_2_6")
    parser.add_argument("--out-dir", type=str, default=os.path.dirname(__file__))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device_index", type=int, default=0, help="Use -1 for CPU, GPU index for CUDA")
    parser.add_argument("--example_cache_len", type=int, default=16)
    parser.add_argument("--example_seq", type=int, default=4, help="example text tokens for export")
    parser.add_argument("--prompt", type=str, default="你好，简单介绍一下你自己。")
    parser.add_argument("--decode_steps", type=int, default=4)
    args = parser.parse_args()
    torch_dtype = getattr(torch, args.dtype)  # 如 "float32" → torch.float32
    torch_device = torch.device(args.device)  # 如 "cuda:0" → torch.device("cuda:0")

    adapter = MiniCPMAdapter.from_pretrained(
        args.model_dir,
        device=torch_device,
        dtype=torch_dtype
    )
    vision_resampler, embed, llm = adapter.get_export_modules()

    export_vision_resampler(vision_resampler.to(torch_device, torch_dtype), args.out_dir, args)
    export_embed(embed.to(torch_device, torch_dtype), args.out_dir, args)
    export_llm(llm.to(torch_device, torch_dtype), adapter, args.out_dir, args)
    # package_path = os.path.join(args.out_dir, "minicpm_llm.pt2")
    # verify_llm(args, adapter, llm, package_path)


if __name__ == "__main__":
    main()


# python /root/autodl-tmp/AOTinfer/minicpm/export_minicpm_aoti.py \
#   --model-dir /root/autodl-tmp/models/MiniCPM_o_2_6 \
#     --out-dir /root/autodl-tmp/AOTinfer/minicpm \
#     --device cuda \
#     --dtype bfloat16