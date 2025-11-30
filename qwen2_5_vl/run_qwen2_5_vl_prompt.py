import argparse
import time
from typing import List, Optional

import torch
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)

from qwen2_5_vl.export_qwen2_5_vl_aoti import flatten_inputs, unflatten_outputs
from qwen2_5_vl.adapter import Qwen2_5_VLAdapter


class AOTIRunner:
    def __init__(self, package_path: str, num_layers: int, device_index: int):
        # # Torch 2.9 nightly removed torch._inductor.codecache from __init__; add it back for aoti_load_package
        try:
            import importlib

            spec = importlib.util.find_spec("torch._inductor.codecache")
            if spec is not None:
                import torch._inductor.codecache as _cc  # type: ignore

                torch._inductor.codecache = _cc  # type: ignore[attr-defined]
        except Exception:
            pass
        compiled = torch._inductor.aoti_load_package(package_path, device_index=device_index)
        self.loader = compiled.loader
        self.num_layers = num_layers

    def run(
        self,
        input_ids: torch.Tensor,
        vision_tokens: torch.Tensor,
        key_cache: List[torch.Tensor],
        value_cache: List[torch.Tensor],
        cache_len: torch.Tensor,
    ):
        flat = flatten_inputs(input_ids, vision_tokens, key_cache, value_cache, cache_len)
        outputs = self.loader.run(flat)
        return unflatten_outputs(outputs, self.num_layers)


class HFRunner:
    def __init__(self, model_path: str, device: str, dtype: torch.dtype, model=None):
        if model is not None:
            self.model = model
        else:
            # Vision2Seq covers VL checkpoints better than plain CausalLM
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    device_map=device,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                ).eval()
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                ).eval()
        self.cache = None

    def run(self, input_ids: torch.Tensor, vision_tokens: Optional[torch.Tensor]):
        kwargs = dict(
            input_ids=input_ids,
            use_cache=True,
            past_key_values=self.cache,
            return_dict=True,
        )
        if vision_tokens is not None:
            kwargs["vision_tower_hidden_states"] = vision_tokens
        try:
            outputs = self.model(**kwargs)
        except TypeError as e:
            raise RuntimeError(
                "HF runner does not accept vision_tower_hidden_states; disable --compare_hf or adjust inputs"
            ) from e
        self.cache = outputs.past_key_values
        return outputs.logits


class WrapperRunner:
    """
    Use export_model (wrapper) directly instead of AOTI.
    """

    def __init__(self, model_path: str, device: str, dtype: torch.dtype, model=None):
        if model is not None:
            adapter = Qwen2_5_VLAdapter(model)
        else:
            adapter = Qwen2_5_VLAdapter.from_pretrained(model_path, device=device, dtype=dtype)
        self.model = adapter.export_model
        self.num_layers = len(self.model.layers)

    def run(
        self,
        input_ids: torch.Tensor,
        vision_tokens: torch.Tensor,
        key_cache: List[torch.Tensor],
        value_cache: List[torch.Tensor],
        cache_len: torch.Tensor,
    ):
        logits, new_key, new_value, new_cache_len = self.model(
            input_ids, key_cache, value_cache, cache_len, vision_tokens
        )
        return logits, new_key, new_value, new_cache_len


def prepare_vision_tokens(args, dtype: torch.dtype, hidden_size: int, processor=None, vision_model=None):
    """
    If --image is provided, run HF vision tower to get projected vision tokens.
    Otherwise, return random tokens as placeholder.
    """
    if not args.image:
        return torch.randn(
            1,
            args.vision_tokens,
            hidden_size,
            device=args.device,
            dtype=dtype,
        )

    processor = processor or AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    image = Image.open(args.image).convert("RGB")
    proc = processor(images=image, text=args.prompt, return_tensors="pt")
    pixel_values = proc["pixel_values"].to(args.device)
    image_grid_thw = proc.get("image_grid_thw", None)
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(args.device)

    # Only need vision tower to get projected tokens
    vision_model = vision_model or AutoModelForVision2Seq.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=dtype, device_map=args.device
    ).eval()
    vision_feats = vision_model.get_image_features(pixel_values, image_grid_thw=image_grid_thw)
    vision_tokens = torch.cat(vision_feats, dim=0).to(dtype=dtype)
    if vision_tokens.dim() == 2:
        vision_tokens = vision_tokens.unsqueeze(0)
    return vision_tokens


@torch.no_grad()
def main(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    dtype = getattr(torch, args.torch_dtype)

    base_model = None
    processor = None
    if args.image or args.use_wrapper or args.compare_hf:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        try:
            base_model = AutoModelForVision2Seq.from_pretrained(
                args.model_path,
                device_map=args.device,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).eval()
        except Exception:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                device_map=args.device,
                trust_remote_code=True,
                torch_dtype=dtype,
            ).eval()

    if args.use_wrapper:
        runner = WrapperRunner(args.model_path, args.device, dtype, model=base_model)
    else:
        if not args.package_path:
            raise ValueError("必须提供 --package_path 或使用 --use_wrapper")
        runner = AOTIRunner(args.package_path, config.num_hidden_layers, args.device_index)
    hf_runner = HFRunner(args.model_path, args.device, dtype, model=base_model) if args.compare_hf else None

    prompt = args.prompt
    tok = tokenizer(prompt, return_tensors="pt").to(args.device)
    vision_tokens = prepare_vision_tokens(args, dtype, config.hidden_size, processor=processor, vision_model=base_model)

    batch_size = tok["input_ids"].shape[0]
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

    t0 = time.perf_counter()
    logits, key_cache, value_cache, cache_len = runner.run(
        tok["input_ids"], vision_tokens, key_cache, value_cache, cache_len
    )
    aoti_prefill_ms = (time.perf_counter() - t0) * 1000

    hf_prefill_ms = None
    if hf_runner:
        t0 = time.perf_counter()
        hf_logits = hf_runner.run(tok["input_ids"], vision_tokens)
        hf_prefill_ms = (time.perf_counter() - t0) * 1000

    aoti_decode_ms = []
    hf_decode_ms = []
    generated = tok["input_ids"]
    generated_hf = tok["input_ids"].clone()
    prompt_len = tok["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id

    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    for step in range(args.max_new_tokens):
        t1 = time.perf_counter()
        logits, key_cache, value_cache, cache_len = runner.run(
            next_token, torch.zeros(1, 0, config.hidden_size, device=device, dtype=dtype), key_cache, value_cache, cache_len
        )
        aoti_decode_ms.append((time.perf_counter() - t1) * 1000)
        generated = torch.cat([generated, next_token], dim=1)
        if hf_runner:
            t2 = time.perf_counter()
            hf_logits = hf_runner.run(next_token, None)
            hf_decode_ms.append((time.perf_counter() - t2) * 1000)
            generated_hf = torch.cat([generated_hf, next_token], dim=1)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        if eos_id is not None and next_token.item() == eos_id:
            break

    def pct(vals: List[float], q: float):
        if not vals:
            return None
        vals_sorted = sorted(vals)
        idx = (len(vals_sorted) - 1) * q
        lo = int(idx)
        hi = min(lo + 1, len(vals_sorted) - 1)
        frac = idx - lo
        return vals_sorted[lo] * (1 - frac) + vals_sorted[hi] * frac

    def print_stats(prefix: str, prefill_ms: float, decode_ms: List[float]):
        print(f"{prefix} prefill: {prefill_ms:.2f} ms")
        if decode_ms:
            avg = sum(decode_ms) / len(decode_ms)
            total_s = sum(decode_ms) / 1000.0
            throughput = len(decode_ms) / total_s if total_s > 0 else float("nan")
            p50 = pct(decode_ms, 0.5)
            p90 = pct(decode_ms, 0.9)
            p99 = pct(decode_ms, 0.99)
            print(
                f"{prefix} decode avg: {avg:.2f} ms | P50 {p50:.2f} | P90 {p90:.2f} | P99 {p99:.2f} | throughput {throughput:.2f} tok/s"
            )

    print("=== Latency (ms) ===")
    print_stats("AOTI", aoti_prefill_ms, aoti_decode_ms)
    if hf_runner and hf_prefill_ms is not None:
        print_stats("HF", hf_prefill_ms, hf_decode_ms)
    # Print decoded text
    text_runner = tokenizer.decode(
        generated[0, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print("=== Runner Output ===")
    print(text_runner)
    if hf_runner:
        text_hf = tokenizer.decode(
            generated_hf[0, prompt_len:], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        print("=== HF Output ===")
        print(text_hf)


def parse_args():
    parser = argparse.ArgumentParser("Run Qwen2.5-VL export model with tokenizer + sampler")
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--prompt", type=str, default="描述一下图片里的内容。")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--package_path", type=str, help="AOTI package path (when not using wrapper)")
    parser.add_argument("--image", type=str, help="Path to an input image; if set, use real vision tokens")
    parser.add_argument("--device_index", type=int, default=-1, help="Use -1 for CPU, GPU index for CUDA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare_hf", action="store_true", help="Compare with HF AutoModel outputs")
    parser.add_argument("--vision_tokens", type=int, default=64, help="Number of vision tokens (projected)")
    parser.add_argument("--use_wrapper", action="store_true", help="Use wrapper (no AOTI package)")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())


# python -m qwen2_5_vl.run_qwen2_5_vl_prompt \
#   -m /Users/bruceli/Desktop/Git_sync/model/Qwen2.5-VL-3B \
#   --package_path /Users/bruceli/Desktop/Git_sync/model/pt2/qwen2.5_vl.pt2 \
#   --image /Users/bruceli/Desktop/Git_sync/AOTinfer/qwen2_5_vl/test.png \
#   --prompt "描述一下图片内容。" \
#   --device cpu --torch_dtype float32 \
#   --compare_hf
