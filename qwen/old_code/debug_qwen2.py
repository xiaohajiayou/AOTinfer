import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from export_qwen2_pt2 import QwenForCausalLMWrapper


def debug_forward_once(args):
    device = args.device
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"[debug] load model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    config = hf_model.config

    pt2_path = os.path.join(args.out_dir, "qwen2.pt2")
    print(f"[debug] load pt2 package from {pt2_path}")
    compiled = torch._inductor.aoti_load_package(pt2_path, device_index=0)
    loader = compiled.loader
    print("[debug] call_spec:", loader.get_call_spec())

    qwen_wrapper = QwenForCausalLMWrapper(hf_model, config, args).to(device).eval()

    # Case A: 使用与导出时相同的 N=4 随机输入
    B = 1
    N = 4
    cache_len = 0
    total_len = cache_len + N

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (B, N),
        device=device,
        dtype=torch.long,
    )

    attention_mask = torch.ones(B, total_len, dtype=torch.long, device=device)
    position_ids = torch.arange(
        cache_len, cache_len + N, dtype=torch.long, device=device
    ).view(B, N)
    cache_position = torch.tensor([cache_len], dtype=torch.long, device=device)

    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = hidden_size // num_heads
    # layer_num = config.num_hidden_layers
    layer_num = 1

    kv_shape = (B, num_kv_heads, cache_len, head_dim)
    key_cache = [
        torch.empty(kv_shape, dtype=dtype, device=device)
        for _ in range(layer_num)
    ]
    value_cache = [
        torch.empty(kv_shape, dtype=dtype, device=device)
        for _ in range(layer_num)
    ]

    past = DynamicCache()
    past.key_cache = key_cache
    past.value_cache = value_cache
    past._seen_tokens = cache_len

    with torch.no_grad():
        out_hf = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past,
            inputs_embeds=None,
            labels=None,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_position=cache_position,
            num_logits_to_keep=1,
        )
    logits_hf = out_hf.logits

    qwen_wrapper.cache_position_int = cache_len
    with torch.no_grad():
        out_wrap = qwen_wrapper(
            input_ids,
            attention_mask,
            position_ids,
            key_cache,
            value_cache,
            cache_position,
        )
    logits_wrap = out_wrap[0]

    inputs = [input_ids, attention_mask, position_ids]
    inputs += list(key_cache)
    inputs += list(value_cache)
    inputs.append(cache_position)
    with torch.no_grad():
        outputs_pt2 = loader.run(inputs)
    logits_pt2 = outputs_pt2[0]

    def compare(a, b, name):
        a_last = a[:, -1, :].float()
        b_last = b[:, -1, :].float()
        diff = a_last - b_last
        max_abs = diff.abs().max().item()
        argmax_equal = (
            torch.argmax(a_last, dim=-1).item()
            == torch.argmax(b_last, dim=-1).item()
        )
        print(
            f"[debug] {name}: max_abs_diff={max_abs:.3e}, "
            f"argmax_equal={argmax_equal}, "
            f"argmax_a={torch.argmax(a_last, -1).item()}, "
            f"argmax_b={torch.argmax(b_last, -1).item()}",
        )

    compare(logits_hf, logits_wrap, "HF vs wrapper")
    compare(logits_wrap, logits_pt2, "wrapper vs PT2")
    compare(logits_hf, logits_pt2, "HF vs PT2")


def parse_args():
    parser = argparse.ArgumentParser(description="debug qwen2 PT2 forward once")
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
        default="/home/cdipc03/models/pt2/Qwen2-0.5B",
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
    parser.add_argument("--add_topk_warper", action="store_true")
    parser.add_argument("--topk", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    debug_forward_once(parse_args())
