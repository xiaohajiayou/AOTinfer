import argparse
import os

import torch
from transformers import AutoTokenizer
from qwen.models.qwen2_adapter import Qwen2Adapter


class Qwen2AOTIWrapper(torch.nn.Module):
    def __init__(self, export_model):
        super().__init__()
        self.export_model = export_model
        self.num_layers = len(export_model.layers)

    def forward(self, *flat_inputs):
        input_ids = flat_inputs[0]
        cache_inputs = flat_inputs[1:]
        key_cache = list(cache_inputs[: self.num_layers])
        value_cache = list(cache_inputs[self.num_layers : 2 * self.num_layers])
        cache_len = cache_inputs[-1]
        logits, new_key, new_value, new_cache_len = self.export_model(
            input_ids, key_cache, value_cache, cache_len
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


def export_and_package(args):
    adapter = Qwen2Adapter.from_pretrained(
        args.model_path, device=args.device, dtype=getattr(torch, args.torch_dtype)
    )
    export_model = adapter.export_model
    wrapper = Qwen2AOTIWrapper(export_model).to(args.device)
    num_layers = wrapper.num_layers

    example_cache_len = args.example_cache_len
    example_seq = args.example_seq
    example_ids = torch.randint(
        0,
        adapter.config.vocab_size,
        (1, example_seq),
        dtype=torch.long,
        device=args.device,
    )
    key_cache = adapter.init_kv_cache(1, example_cache_len, export_model.embed.weight.dtype, args.device)
    value_cache = adapter.init_kv_cache(1, example_cache_len, export_model.embed.weight.dtype, args.device)
    cache_len_tensor = torch.tensor(example_cache_len, dtype=torch.long, device=args.device)
    flat_example = flatten_inputs(example_ids, key_cache, value_cache, cache_len_tensor)

    N_dim = torch.export.Dim("N", min=1, max=adapter.config.max_position_embeddings)
    cache_dim = torch.export.Dim("cache_len", min=0, max=adapter.config.max_position_embeddings)
    per_input_shapes = []
    per_input_shapes.append({1: N_dim})
    for _ in range(num_layers):
        per_input_shapes.append({2: cache_dim})
    for _ in range(num_layers):
        per_input_shapes.append({2: cache_dim})
    per_input_shapes.append({})
    dynamic_shapes = {"flat_inputs": tuple(per_input_shapes)}

    print("[export] tracing model to ExportedProgram")
    exported = torch.export.export(wrapper, flat_example, dynamic_shapes=dynamic_shapes)
    os.makedirs(args.out_dir, exist_ok=True)
    package_path = os.path.join(args.out_dir, args.package_name)
    print(f"[export] compiling to {package_path}")
    torch._inductor.aoti_compile_and_package(exported, package_path=package_path)
    return adapter, wrapper, package_path


def verify_with_aoti(args, adapter, wrapper, package_path):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    compiled = torch._inductor.aoti_load_package(package_path, device_index=args.device_index)
    loader = compiled.loader
    num_layers = wrapper.num_layers
    dtype = wrapper.export_model.embed.weight.dtype

    def run_wrapper(input_ids, key_cache, value_cache, cache_len):
        out = wrapper(*flatten_inputs(input_ids, key_cache, value_cache, cache_len))
        return unflatten_outputs(out, num_layers)

    def run_loader(input_ids, key_cache, value_cache, cache_len):
        out = loader.run(flatten_inputs(input_ids, key_cache, value_cache, cache_len))
        return unflatten_outputs(out, num_layers)

    prompt = args.prompt
    tok = tokenizer(prompt, return_tensors="pt").to(args.device)
    key_cache_w = adapter.init_kv_cache(1, 0, dtype, args.device)
    value_cache_w = adapter.init_kv_cache(1, 0, dtype, args.device)
    key_cache_l = [t.clone() for t in key_cache_w]
    value_cache_l = [t.clone() for t in value_cache_w]
    cache_len_w = torch.tensor(0, dtype=torch.long, device=args.device)
    cache_len_l = cache_len_w.clone()

    logits_w, key_cache_w, value_cache_w, cache_len_w = run_wrapper(
        tok["input_ids"], key_cache_w, value_cache_w, cache_len_w
    )
    logits_l, key_cache_l, value_cache_l, cache_len_l = run_loader(
        tok["input_ids"], key_cache_l, value_cache_l, cache_len_l
    )
    compare_logits(logits_w[:, -1, :], logits_l[:, -1, :], step="prefill")

    next_token = torch.argmax(logits_w[:, -1, :], dim=-1, keepdim=True)
    for step in range(args.decode_steps):
        logits_w, key_cache_w, value_cache_w, cache_len_w = run_wrapper(
            next_token, key_cache_w, value_cache_w, cache_len_w
        )
        logits_l, key_cache_l, value_cache_l, cache_len_l = run_loader(
            next_token, key_cache_l, value_cache_l, cache_len_l
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


def parse_args():
    parser = argparse.ArgumentParser(description="Export Qwen2 export model to AOTI package and verify")
    parser.add_argument("-m", "--model_path", type=str, default="/home/cdipc03/models/Qwen/Qwen2-0.5B")
    parser.add_argument("-o", "--out_dir", type=str, default="/home/cdipc03/models/pt2/Qwen2-0.5B")
    parser.add_argument("--package_name", type=str, default="qwen2_export.pt2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--device_index", type=int, default=-1, help="Use -1 for CPU, GPU index for CUDA")
    parser.add_argument("--example_cache_len", type=int, default=16)
    parser.add_argument("--example_seq", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="你好，简单介绍一下你自己。")
    parser.add_argument("--decode_steps", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    adapter, wrapper, package_path = export_and_package(args)
    verify_with_aoti(args, adapter, wrapper, package_path)
