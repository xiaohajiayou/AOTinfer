import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def main(args):
    model_path = args.model_path
    device = args.device

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"[jit] load model from {model_path}")
    tokenizer = AutoModelForCausalLM.from_pretrained
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).eval()

    print("[jit] scripting model...")
    scripted = torch.jit.script(model).eval()

    prompts = [
        "你好，简单介绍一下你自己。",
        "你好，简单介绍一下你自己。再多说两句。",
    ]

    for i, prompt in enumerate(prompts):
        tok = tokenizer(prompt, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}

        with torch.no_grad():
            out_eager = model(**tok, use_cache=False, return_dict=True)
            out_script = scripted(**tok, use_cache=False, return_dict=True)

        a = out_eager.logits[:, -1, :].float()
        b = out_script.logits[:, -1, :].float()
        max_abs = (a - b).abs().max().item()
        argmax_eager = torch.argmax(a, dim=-1).item()
        argmax_script = torch.argmax(b, dim=-1).item()

        print(
            f"[case {i}] max_abs_diff={max_abs:.3e}, "
            f"argmax_eager={argmax_eager}, argmax_script={argmax_script}, "
            f"equal={argmax_eager == argmax_script}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="test qwen2 jit dynamic N")
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="/home/cdipc03/models/Qwen/Qwen2-0.5B",
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
    main(parse_args())

