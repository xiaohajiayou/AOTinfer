import os
import argparse

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenPrefillWrapper(nn.Module):
    """
    只做 full prefill、不带 KV cache 的包装。
    前向: (input_ids, position_ids) -> logits
    在内部构造 4D causal mask，并以 dict 形式传给 Qwen2。
    """

    def __init__(self, model, config, args):
        super().__init__()
        self.model = model
        self.config = config
        self.args = args

    def forward(self, input_ids, position_ids):
        B, N = input_ids.shape
        device = input_ids.device

        # 构造 4D causal mask: [B, 1, N, N]，下三角为 1
        causal = torch.tril(torch.ones(N, N, dtype=torch.bool, device=device))
        causal = causal.view(1, 1, N, N)
        attention_mask = {"full_attention": causal}

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        return outputs.logits


def export_qwen_prefill_pt2(args):
    device = args.device
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"[prefill] load model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).eval()
    config = model.config
    print("[prefill] config:", config)

    wrapper = QwenPrefillWrapper(model, config, args).to(device).eval()

    # example inputs: B=1, N0=16（示例长度）
    B = 1
    N0 = 16
    input_ids = torch.randint(
        0,
        config.vocab_size,
        (B, N0),
        dtype=torch.int64,
        device=device,
    )
    position_ids = torch.arange(0, N0, dtype=torch.int64, device=device).reshape(B, N0)

    example_inputs = (input_ids, position_ids)

    # 动态 N 维度: 允许 1..max_position_embeddings
    N = torch.export.Dim("N", min=1, max=config.max_position_embeddings)
    dynamic_shapes = (
        {1: N},  # input_ids: [B, N]
        {1: N},  # position_ids: [B, N]
    )

    print("[prefill] begin torch.export.export")
    exported = torch.export.export(
        wrapper,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    package_path = os.path.join(args.out_dir, "qwen2_prefill.pt2")
    print(f"[prefill] begin aoti_compile_and_package to {package_path}")
    out = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=package_path,
    )
    print("[prefill] AOTI PT2 package:", out)


def verify_qwen_prefill(args):
    device = args.device
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).eval()
    config = model.config

    pt2_path = os.path.join(args.out_dir, "qwen2_prefill.pt2")
    print(f"[prefill] load pt2 package from {pt2_path}")
    compiled = torch._inductor.aoti_load_package(pt2_path, device_index=0)
    loader = compiled.loader
    print("[prefill] call_spec:", loader.get_call_spec())

    # 构造一条真实 prompt 做单步 prefill 对比
    prompt = "你好，简单介绍一下你自己。"
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)  # [B, N]
    B, N = input_ids.shape
    assert B == 1

    position_ids = torch.arange(0, N, dtype=torch.long, device=device).view(B, N)

    with torch.no_grad():
        out_hf = model(
            input_ids=input_ids,
            # 不传 attention_mask，让 HF 走默认 mask 路径
            position_ids=position_ids,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    logits_hf = out_hf.logits  # [B, N, vocab]

    # PT2 按 wrapper 的签名，只传 (input_ids, position_ids)
    inputs = [input_ids, position_ids]
    with torch.no_grad():
        outputs_pt2 = loader.run(inputs)
    logits_pt2 = outputs_pt2[0]  # [B, N, vocab]

    # 对比最后一个位置 logits
    a = logits_hf[:, -1, :].float()
    b = logits_pt2[:, -1, :].float()
    diff = a - b
    max_abs = diff.abs().max().item()
    argmax_hf = torch.argmax(a, dim=-1).item()
    argmax_pt2 = torch.argmax(b, dim=-1).item()
    print(
        f"[prefill] HF vs PT2: max_abs_diff={max_abs:.3e}, "
        f"argmax_hf={argmax_hf}, argmax_pt2={argmax_pt2}, "
        f"equal={argmax_hf == argmax_pt2}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="export/verify qwen2 prefill PT2")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_qwen_prefill_pt2(args)
    verify_qwen_prefill(args)
