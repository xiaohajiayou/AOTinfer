import os
import argparse

import torch

from qwen.old_code.trace_qwen2_kv import (
    QwenForCausalLMWrapper,
    build_example_inputs,
)
from transformers import AutoModelForCausalLM


def verify_kv_traced(args):
    device = args.device
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"[verify] load model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).eval()
    config = model.config

    wrapper = QwenForCausalLMWrapper(model, config, args).to(device).eval()

    ts_path = os.path.join(args.out_dir, "qwen2_kv_traced.pt")
    print(f"[verify] load traced module from {ts_path}")
    traced = torch.jit.load(ts_path, map_location=device).eval()

    # 使用与 trace 相同的 example_inputs 作为起点
    example_inputs = build_example_inputs(model, config, args)
    (
        input_ids0,
        position_ids0,
        key_cache0,
        value_cache0,
        cache_position0,
    ) = example_inputs

    # 为 eager 和 traced 各准备一份独立 cache
    key_cache_e = [t.clone() for t in key_cache0]
    value_cache_e = [t.clone() for t in value_cache0]
    key_cache_t = [t.clone() for t in key_cache0]
    value_cache_t = [t.clone() for t in value_cache0]

    B, N = input_ids0.shape
    assert B == 1 and N == 1, "trace example assumes batch=1, N=1"

    cache_len = key_cache0[0].shape[2]

    num_steps = args.num_steps
    print(
        f"[verify] start decode loop from cache_len={cache_len}, steps={num_steps}"
    )

    for step in range(num_steps):
        # 每步用一个固定 token（这里沿用 example input 中的 token）
        input_ids_step = input_ids0

        # 根据当前 cache_len 构造 position_ids 和 cache_position
        position_ids_step = torch.arange(
            cache_len,
            cache_len + N,
            dtype=torch.long,
            device=device,
        ).view(B, N)
        cache_position_step = torch.tensor(
            [cache_len], dtype=torch.long, device=device
        )

        # eager wrapper
        with torch.no_grad():
            out_eager = wrapper(
                input_ids_step,
                position_ids_step,
                key_cache_e,
                value_cache_e,
                cache_position_step,
            )

        logits_e = out_eager[0]
        layer_num = len(key_cache_e)
        new_key_e = list(out_eager[1 : 1 + layer_num])
        new_val_e = list(out_eager[1 + layer_num : 1 + 2 * layer_num])

        # traced wrapper
        with torch.no_grad():
            out_traced = traced(
                input_ids_step,
                position_ids_step,
                key_cache_t,
                value_cache_t,
                cache_position_step,
            )

        logits_t = out_traced[0]
        new_key_t = list(out_traced[1 : 1 + layer_num])
        new_val_t = list(out_traced[1 + layer_num : 1 + 2 * layer_num])

        # 对比 logits
        a = logits_e[:, -1, :].float()
        b = logits_t[:, -1, :].float()
        diff = (a - b).abs().max().item()
        argmax_e = torch.argmax(a, dim=-1).item()
        argmax_t = torch.argmax(b, dim=-1).item()
        equal = argmax_e == argmax_t

        print(
            f"[step {step}] max_abs_diff={diff:.3e}, "
            f"argmax_eager={argmax_e}, argmax_traced={argmax_t}, equal={equal}"
        )

        if not equal:
            print("[verify] mismatch found, stop.")
            break

        # 更新 cache 和 cache_len
        key_cache_e = new_key_e
        value_cache_e = new_val_e
        key_cache_t = new_key_t
        value_cache_t = new_val_t
        cache_len = key_cache_e[0].shape[2]

    else:
        print("[verify] all steps matched.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="verify qwen2 KV traced module with multi-step decode"
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="/home/cdipc03/models/Qwen/Qwen2-0.5B",
    )
    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="/home/cdipc03/models/ts/Qwen2-0.5B",
    )
    parser.add_argument(
        "-d", "--device", type=str, choices=["cpu", "cuda"], default="cuda"
    )
    parser.add_argument(
        "-p",
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=4,
        help="number of decode steps to verify",
    )
    return parser.parse_args()


if __name__ == "__main__":
    verify_kv_traced(parse_args())

