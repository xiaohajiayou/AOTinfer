import argparse
import sys
from typing import List, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from qwen.export_qwen2_aoti import flatten_inputs, unflatten_outputs


class AOTIRunner:
    def __init__(self, package_path: str, num_layers: int, device_index: int):
        compiled = torch._inductor.aoti_load_package(package_path, device_index=device_index)
        self.loader = compiled.loader
        self.num_layers = num_layers

    def run(
        self,
        input_ids: torch.Tensor,
        key_cache: List[torch.Tensor],
        value_cache: List[torch.Tensor],
        cache_len: torch.Tensor,
    ):
        flat = flatten_inputs(input_ids, key_cache, value_cache, cache_len)
        outputs = self.loader.run(flat)
        return unflatten_outputs(outputs, self.num_layers)


class HFRunner:
    def __init__(self, model_path: str, device: str, dtype: torch.dtype):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).eval()
        self.cache = None

    def run(self, input_ids: torch.Tensor):
        outputs = self.model(input_ids=input_ids, use_cache=True, past_key_values=self.cache, return_dict=True)
        self.cache = outputs.past_key_values
        return outputs.logits


def build_logits_processor(args) -> LogitsProcessorList:
    processors = LogitsProcessorList()
    if args.temperature and args.temperature != 1.0:
        processors.append(TemperatureLogitsWarper(args.temperature))
    if args.top_k and args.top_k > 0:
        processors.append(TopKLogitsWarper(args.top_k))
    if args.top_p and args.top_p < 1.0:
        processors.append(TopPLogitsWarper(args.top_p))
    return processors


@torch.no_grad()
def main(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    if not args.package_path:
        raise ValueError("请提供 --package_path 指向导出的 AOTI 包")

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dtype = getattr(torch, args.torch_dtype)
    runner = AOTIRunner(args.package_path, config.num_hidden_layers, args.device_index)
    hf_runner = HFRunner(args.model_path, args.device, dtype) if args.compare_hf else None
    run_joint_stream(tokenizer, runner, config, dtype, args, hf_runner)


def run_joint_stream(tokenizer, runner, config, dtype, args, hf_runner: Optional[HFRunner]):
    tok = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    generated = tok["input_ids"]
    hf_generated = generated.clone()
    batch_size = generated.shape[0]
    prompt_len = generated.shape[1]

    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    device = tok["input_ids"].device

    def init_cache(cache_len: int):
        cache = []
        for _ in range(num_layers):
            cache.append(
                torch.zeros(
                    batch_size,
                    num_kv_heads,
                    cache_len,
                    head_dim,
                    dtype=dtype,
                    device=device,
                )
            )
        return cache

    key_cache = init_cache(0)
    value_cache = init_cache(0)
    cache_len = torch.tensor(0, dtype=torch.long, device=device)

    logits, key_cache, value_cache, cache_len = runner.run(
        generated, key_cache, value_cache, cache_len
    )
    processors = build_logits_processor(args)
    print("=== Streaming Output (AOTI) ===")
    full_text = tokenizer.decode(
        generated[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    sys.stdout.write(full_text)
    sys.stdout.flush()
    prev_len = len(full_text)
    log_messages: List[str] = []
    hf_logits = None
    if hf_runner:
        hf_logits = hf_runner.run(generated)
        log_messages.append(format_logits_diff(hf_logits[:, -1, :], logits[:, -1, :], "prefill"))

    eos_token_id = tokenizer.eos_token_id
    for step in range(args.max_new_tokens):
        step_logits = logits[:, -1, :]
        processed_logits = processors(generated, step_logits)
        if args.greedy or len(processors) == 0:
            next_token = torch.argmax(processed_logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(processed_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)
        full_text = tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        sys.stdout.write("\r" + full_text)
        sys.stdout.write(" " * max(0, prev_len - len(full_text)))
        sys.stdout.flush()
        prev_len = len(full_text)

        logits, key_cache, value_cache, cache_len = runner.run(
            next_token, key_cache, value_cache, cache_len
        )
        if hf_runner:
            hf_generated = torch.cat([hf_generated, next_token], dim=1)
            hf_logits = hf_runner.run(next_token)
            log_messages.append(
                format_logits_diff(hf_logits[:, -1, :], logits[:, -1, :], f"decode-{step}")
            )
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    print()
    sys.stdout.write("\n")
    if hf_runner:
        print("=== HF Reference ===")
        hf_text = tokenizer.decode(
            hf_generated[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        print(hf_text)
        print("=== HF vs AOTI logits ===")
        for msg in log_messages:
            print(msg)


def format_logits_diff(logits_hf, logits_aoti, step: str) -> str:
    diff = (logits_hf.float() - logits_aoti.float()).abs().max().item()
    arg_hf = torch.argmax(logits_hf, dim=-1).item()
    arg_aoti = torch.argmax(logits_aoti, dim=-1).item()
    equal = arg_hf == arg_aoti
    return f"[{step}] max_abs_diff={diff:.3e}, arg_hf={arg_hf}, arg_aoti={arg_aoti}, equal={equal}"


def parse_args():
    parser = argparse.ArgumentParser("Run Qwen2 export model with tokenizer + sampler")
    parser.add_argument("-m", "--model_path", type=str, default="/home/cdipc03/models/Qwen/Qwen2-0.5B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--prompt", type=str, default="你好，简单介绍一下你自己。")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling")
    parser.add_argument("--package_path", type=str, required=True, help="AOTI package path")
    parser.add_argument("--device_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare_hf", action="store_true", help="Compare with HF AutoModel outputs")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
