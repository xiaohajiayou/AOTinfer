#!/usr/bin/env python3
"""Dump tensors required for C++ AOTI demo."""

import argparse
import os
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from minicpm.models.model import build_from_hf
from minicpm.run_minicpmo_hf_demo import WrapperRunner
from minicpm.preprocess.image_preprocess import preprocess_images


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def build_prompt(prompt: str, has_image: bool) -> str:
    system_block = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    )
    if has_image:
        user_block = f"<|im_start|>user\n(<image>./</image>)\n{prompt}<|im_end|>\n"
    else:
        user_block = f"<|im_start|>user\n{prompt}<|im_end|>\n"
    return system_block + user_block + "<|im_start|>assistant\n"


def ensure_tensor(data, device):
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return torch.as_tensor(data, device=device)


def main():
    parser = argparse.ArgumentParser(description="Dump MiniCPM tensors for AOTI C++ demo")
    parser.add_argument("--model-path", type=str, default="/home/liwenxiao/models/minicpm_o_2_6")
    parser.add_argument("--image", type=str, default="/home/liwenxiao/AOTinfer/qwen2_5_vl/test.png")
    parser.add_argument("--prompt", type=str, default="请描述图片里的内容。")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--out-dir", type=str, default="cpp/artifacts")
    parser.add_argument("--max-slices", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    image = load_image(args.image)
    prompt_text = build_prompt(args.prompt, has_image=image is not None)

    if image is not None:
        proc = processor(
            text=[prompt_text],
            images=[image],
            return_tensors="pt",
            max_slice_nums=args.max_slices,
            add_special_tokens=True,
            use_image_id=True,
            chunk_input=True,
        )
    else:
        proc = processor(text=[prompt_text], return_tensors="pt", add_special_tokens=True)

    input_ids = proc["input_ids"].to(device)
    attention_mask = proc["attention_mask"].to(device)
    image_bound = proc.get("image_bound")
    if image_bound is not None:
        if isinstance(image_bound, (list, tuple)):
            image_bound = torch.as_tensor(image_bound[0], dtype=torch.long, device=device).unsqueeze(0)
        else:
            image_bound = image_bound.to(device)

    pixel_values = tgt_sizes = None
    if image is not None:
        img_np = np.array(image.convert("RGB"))
        img_size = getattr(processor.image_processor, "scale_resolution", 448)
        patch_size = getattr(processor.image_processor, "patch_size", 14)
        max_slice = getattr(processor.image_processor, "max_slice_nums", args.max_slices)
        pixel_list, tgt_sizes_np, slice_counts = preprocess_images(
            [img_np],
            max_slice_nums=max_slice,
            scale_resolution=img_size,
            patch_size=patch_size,
        )
        num_slices = slice_counts[0]
        if num_slices > 0:
            pixel_values = torch.stack(
                [torch.from_numpy(pv).to(device=device, dtype=dtype) for pv in pixel_list[:num_slices]], dim=0
            )
            tgt_sizes = torch.from_numpy(tgt_sizes_np[:num_slices]).to(device=device, dtype=torch.int64)

    hf_model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
        init_vision=True,
        init_audio=False,
        init_tts=False,
        local_files_only=True,
    ).eval()
    custom_model = build_from_hf(hf_model).to(device)
    runner = WrapperRunner(custom_model, tokenizer)

    vision_tokens = runner.encode_image(pixel_values, tgt_sizes)
    inputs_embeds = runner.build_inputs(input_ids, vision_tokens, image_bound)

    num_layers = len(runner.model.decoder.layers)
    num_kv_heads = runner.model.decoder.num_kv_heads
    head_dim = runner.model.decoder.head_dim

    key_cache = [
        torch.zeros(input_ids.size(0), num_kv_heads, 0, head_dim, device=device, dtype=inputs_embeds.dtype)
        for _ in range(num_layers)
    ]
    value_cache = [torch.zeros_like(k) for k in key_cache]
    cache_len = torch.tensor(0, dtype=torch.long, device=device)

    logits, key_cache, value_cache, cache_len = runner.llm_step(
        inputs_embeds, key_cache, value_cache, cache_len, attention_mask
    )

    dump = {
        "pixel_values": pixel_values.cpu() if pixel_values is not None else torch.empty(0),
        "tgt_sizes": tgt_sizes.cpu() if tgt_sizes is not None else torch.empty(0, dtype=torch.long),
        "input_ids": input_ids.cpu(),
        "attention_mask": attention_mask.cpu(),
        "image_bound": image_bound.cpu() if image_bound is not None else torch.empty(0, dtype=torch.long),
        "vision_tokens": vision_tokens.cpu() if vision_tokens is not None else torch.empty(0),
        "inputs_embeds": inputs_embeds.cpu(),
        "key_cache": [k.cpu() for k in key_cache],
        "value_cache": [v.cpu() for v in value_cache],
        "cache_len": cache_len.cpu(),
        "logits": logits[:, -1, :].cpu(),
    }
    dump_path = Path(args.out_dir) / "minicpm_aoti_inputs.pt"
    torch.save(dump, dump_path)

    hidden_size = runner.model.decoder.lm_head.weight.shape[1]
    meta = {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "prompt": args.prompt,
        "image": args.image,
    }
    with open(Path(args.out_dir) / "minicpm_aoti_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[dump] tensors saved to {dump_path}")


if __name__ == "__main__":
    main()
