import os
import argparse
import logging

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


class QwenForCausalLMWrapper(nn.Module):
    """
    KV cache 显式作为输入/输出的 wrapper，用于 jit.trace。
    前向签名：
      (input_ids, position_ids, key_cache, value_cache, cache_position)
        -> (logits, new_key_cache..., new_value_cache...)
    内部自己构造 4D causal mask，并以 dict 形式传给 Qwen2，
    避免走 masking_utils 的 2D mask 路径。
    """

    def __init__(self, model, config, args):
        super().__init__()
        self.model = model
        self.config = config
        self.args = args
        self.layer_num = len(model.model.layers)
        # 仅用于初始化 DynamicCache._seen_tokens
        self.cache_position_int = 0

    def forward(
        self,
        input_ids,
        position_ids,
        key_cache,
        value_cache,
        cache_position,
    ):
        use_cache = True
        output_attentions = False
        output_hidden_states = False
        return_dict = True
        num_logits_to_keep = 1

        # 构造 4D causal mask: [B, 1, N, total_k]
        B, N = input_ids.shape
        device = input_ids.device
        # 从 KV cache 推出历史长度
        if len(key_cache) > 0:
            cache_len = key_cache[0].shape[2]
        else:
            cache_len = 0
        total_k = cache_len + N
        mask_dtype = self.model.model.embed_tokens.weight.dtype

        large_neg = torch.finfo(mask_dtype).min
        causal = torch.full(
            (N, total_k),
            fill_value=large_neg,
            device=device,
            dtype=mask_dtype,
        )
        i = torch.arange(N, device=device).unsqueeze(-1)
        j = torch.arange(total_k, device=device).unsqueeze(0)
        causal = torch.where(
            j <= cache_len + i,
            torch.tensor(0, dtype=mask_dtype, device=device),
            causal,
        )
        causal = causal.view(1, 1, N, total_k)
        attention_mask = {"full_attention": causal}

        past_key_values = DynamicCache()
        past_key_values.key_cache = key_cache
        past_key_values.value_cache = value_cache
        past_key_values._seen_tokens = self.cache_position_int

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=None,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs.logits
        key_cache_out = [t for t in outputs.past_key_values.key_cache]
        value_cache_out = [t for t in outputs.past_key_values.value_cache]
        return (logits, *key_cache_out, *value_cache_out)


def build_example_inputs(model, config, args):
    """
    构造一组用于 trace 的 example inputs，形状和 onnx 脚本类似。
    """
    device = args.device
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    layer_num = len(model.model.layers)

    hidden_size = config.hidden_size
    head_num = config.num_attention_heads
    head_dim = hidden_size // head_num
    num_kv_heads = config.num_key_value_heads

    batch = 1
    N = 1
    sumN = 38
    lastSum = sumN - N

    input_ids = torch.ones([batch, N], dtype=torch.int64, device=device)
    position_ids = torch.tensor([lastSum], dtype=torch.int64, device=device).reshape(
        batch, N
    )
    cache_position = torch.tensor([lastSum], dtype=torch.int64, device=device)

    kv_cache_shape = [batch, num_kv_heads, lastSum, head_dim]
    key_cache = []
    value_cache = []
    for _ in range(layer_num):
        past_key = torch.randn(kv_cache_shape, dtype=dtype, device=device)
        past_value = torch.randn(kv_cache_shape, dtype=dtype, device=device)
        key_cache.append(past_key)
        value_cache.append(past_value)

    return (input_ids, position_ids, key_cache, value_cache, cache_position)


def trace_qwen2(args):
    device = args.device
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"[trace] load model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).eval()
    config = model.config

    wrapper = QwenForCausalLMWrapper(model, config, args).to(device).eval()
    example_inputs = build_example_inputs(model, config, args)

    print("[trace] begin torch.jit.trace (KV wrapper, 4D mask)")
    traced = torch.jit.trace(wrapper, example_inputs, strict=False)
    traced.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    save_path = os.path.join(args.out_dir, "qwen2_kv_traced.pt")
    traced.save(save_path)
    print(f"[trace] saved traced module to: {save_path}")

    # 简单一致性检查：用同一组 example inputs 比较 eager vs traced
    with torch.no_grad():
        eager_out = wrapper(*example_inputs)[0]
        traced_out = traced(*example_inputs)[0]

    a = eager_out[:, -1, :].float()
    b = traced_out[:, -1, :].float()
    diff = (a - b).abs().max().item()
    argmax_eager = torch.argmax(a, dim=-1).item()
    argmax_traced = torch.argmax(b, dim=-1).item()
    print(
        f"[trace] eager vs traced (example inputs): max_abs_diff={diff:.3e}, "
        f"argmax_eager={argmax_eager}, argmax_traced={argmax_traced}, "
        f"equal={argmax_eager == argmax_traced}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="trace qwen2 KV wrapper with jit.trace")
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
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trace_qwen2(parse_args())
