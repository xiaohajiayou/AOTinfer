import argparse

import torch
from transformers import AutoTokenizer

from qwen.models.qwen2_adapter import Qwen2Adapter


@torch.no_grad()
def verify(args):
    adapter = Qwen2Adapter.from_pretrained(args.model_path, device=args.device, dtype=getattr(torch, args.torch_dtype))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    export_model = adapter.export_model
    hf_model = adapter.hf_model

    prompt = args.prompt
    tok = tokenizer(prompt, return_tensors="pt").to(args.device)
    batch_size, prompt_len = tok["input_ids"].shape

    key_cache = adapter.init_kv_cache(batch_size, 0, export_model.embed.weight.dtype, tok["input_ids"].device)
    value_cache = adapter.init_kv_cache(batch_size, 0, export_model.embed.weight.dtype, tok["input_ids"].device)
    cache_len = torch.tensor(0, dtype=torch.long, device=tok["input_ids"].device)

    hf_out = hf_model(**tok, use_cache=True, return_dict=True)
    logits_hf = hf_out.logits
    hf_past = hf_out.past_key_values

    logits_export, key_cache, value_cache, cache_len = export_model(
        tok["input_ids"], key_cache, value_cache, cache_len
    )

    compare_logits(logits_hf[:, -1, :], logits_export[:, -1, :], step="prefill")

    next_token = torch.argmax(logits_hf[:, -1, :], dim=-1, keepdim=True)
    for step in range(args.decode_steps):
        hf_out = hf_model(
            input_ids=next_token,
            use_cache=True,
            past_key_values=hf_past,
            return_dict=True,
        )
        logits_hf = hf_out.logits
        hf_past = hf_out.past_key_values

        logits_export, key_cache, value_cache, cache_len = export_model(
            next_token, key_cache, value_cache, cache_len
        )

        compare_logits(logits_hf[:, -1, :], logits_export[:, -1, :], step=f"decode-{step}")

        next_token = torch.argmax(logits_hf[:, -1, :], dim=-1, keepdim=True)


def compare_logits(logits_hf, logits_export, step: str):
    diff = (logits_hf.float() - logits_export.float()).abs().max().item()
    arg_hf = torch.argmax(logits_hf, dim=-1).item()
    arg_export = torch.argmax(logits_export, dim=-1).item()
    equal = arg_hf == arg_export
    print(f"[{step}] max_abs_diff={diff:.3e}, arg_hf={arg_hf}, arg_export={arg_export}, equal={equal}")


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Qwen2 export model prefll+decode")
    parser.add_argument("-m", "--model_path", type=str, default="/home/cdipc03/models/Qwen/Qwen2-0.5B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="float16")
    parser.add_argument("--prompt", type=str, default="你好，简单介绍一下你自己。")
    parser.add_argument("--decode_steps", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    verify(parse_args())
