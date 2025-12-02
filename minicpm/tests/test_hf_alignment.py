"""
HF 对齐测试脚手架（占位）：
- 使用 HF MiniCPM 原模型与自研模型，给定相同输入（文本+图像预处理后的张量），对比 logits。
- 目前自研模型仍是占位实现，需替换为真实前向/权重后再跑。
"""

import argparse
import os
import sys

# add project root to path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import torch
from PIL import Image
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoProcessor

from minicpm.preprocess.image_preprocess import preprocess_images, pad_slices_to_tensor
from minicpm.models.model import build_from_hf
from minicpm.models.adapter import scatter_vision_tokens


def load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def parse_args():
    ap = argparse.ArgumentParser(description="HF vs custom MiniCPM alignment test")
    ap.add_argument("--model-dir", type=str, default="/home/liwenxiao/models/minicpm_o_2_6")
    ap.add_argument("--image", type=str, default="/home/liwenxiao/AOTinfer/qwen2_5_vl/test.png")
    ap.add_argument("--prompt", type=str, default="请描述图片")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-slices", type=int, default=2)
    ap.add_argument("--no-vision", action="store_true", help="仅测试文本对齐")
    ap.add_argument("--vision-only", action="store_true", help="仅对齐视觉塔（SigLIP）")
    ap.add_argument("--vision-resampler", action="store_true", help="对齐视觉+Resampler")
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model_dir = args.model_dir
    prompt = args.prompt

    hf_model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
        local_files_only=True,
        init_audio=False,
        init_tts=False,
        init_vision=not args.no_vision,
        attn_implementation="eager",
    ).eval()
    my_model = build_from_hf(hf_model).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if not args.no_vision:
        # 文本部分：简单占位符模板 (<image>./</image>) + 文本
        text_prompt = "(<image>./</image>)\n" + prompt
        enc = tokenizer(text_prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        # 视觉仍用自研预处理，保证与后续导出一致
        img_np = load_image(args.image)
        img_size = getattr(hf_model.config.vision_config, "image_size", 980)
        patch_size = getattr(hf_model.config.vision_config, "patch_size", 14)
        pixel_list, tgt_sizes_np, slice_counts = preprocess_images(
            [img_np],
            max_slice_nums=args.max_slices,
            scale_resolution=img_size,
            patch_size=patch_size,
        )
        pixel_values, tgt_sizes = pad_slices_to_tensor(
            pixel_list,
            tgt_sizes_np,
            s_max=args.max_slices,
            h=img_size,
            w=img_size,
            device=device,
        )
        pixel_values = pixel_values.to(device=device, dtype=dtype)
        num_slices = slice_counts[0]
        pixel_values_hf = pixel_values[:num_slices]
        tgt_sizes_hf = tgt_sizes[:num_slices]
        # image_bound 覆盖占位符区域：从 BOS 后第一个 token 开始
        vision_len = num_slices * 64
        image_bound = torch.tensor([[[1, 1 + vision_len]]], dtype=torch.long)
        # 如长度不足，pad input_ids/attention_mask
        need_len = 1 + vision_len
        if input_ids.shape[1] < need_len:
            pad_len = need_len - input_ids.shape[1]
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            pad = torch.full((1, pad_len), pad_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, pad], dim=1)
            att_pad = torch.ones((1, pad_len), dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, att_pad], dim=1)
        image_bound = image_bound.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    else:
        pixel_values_hf = tgt_sizes_hf = image_bound = None
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

    # 仅视觉塔对齐
    if args.vision_only:
        from minicpm.models.vision import SiglipVisionExport

        vpm = hf_model.vpm if hasattr(hf_model, "vpm") else hf_model.vision_model
        vpm = vpm.to(device=device, dtype=dtype)
        vpm_export = SiglipVisionExport.from_hf(vpm).to(device=device, dtype=dtype)
        with torch.inference_mode():
            hf_vis = vpm(pixel_values=pixel_values_hf, tgt_sizes=tgt_sizes_hf)
            hf_hidden = hf_vis.last_hidden_state if hasattr(hf_vis, "last_hidden_state") else hf_vis
            my_hidden = vpm_export(pixel_values_hf, tgt_sizes=tgt_sizes_hf)
        print("vision hf shape:", getattr(hf_hidden, "shape", None))
        print("vision my shape:", getattr(my_hidden, "shape", None))
        if hasattr(hf_hidden, "shape") and hasattr(my_hidden, "shape"):
            diff = (hf_hidden - my_hidden).abs().max().item()
            print("vision max diff:", diff)
        return

    # 视觉+Resampler 对齐
    if args.vision_resampler:
        from minicpm.models.vision import SiglipVisionExport
        from minicpm.models.resampler import ResamplerExport

        vpm = hf_model.vpm if hasattr(hf_model, "vpm") else hf_model.vision_model
        vpm = vpm.to(device=device, dtype=dtype)
        vpm_export = SiglipVisionExport.from_hf(vpm).to(device=device, dtype=dtype)
        res_export = ResamplerExport.from_hf(
            hf_model.resampler, num_heads=hf_model.config.vision_config.num_attention_heads
        ).to(device=device, dtype=dtype)
        hf_model.resampler = hf_model.resampler.to(device=device, dtype=dtype)
        with torch.inference_mode():
            hf_vis = vpm(pixel_values=pixel_values_hf, tgt_sizes=tgt_sizes_hf)
            hf_hidden = hf_vis.last_hidden_state if hasattr(hf_vis, "last_hidden_state") else hf_vis
            my_hidden = vpm_export(pixel_values_hf, tgt_sizes=tgt_sizes_hf)
            hf_res = hf_model.resampler(hf_hidden, tgt_sizes=tgt_sizes_hf)
            hf_res_hidden = hf_res.last_hidden_state if hasattr(hf_res, "last_hidden_state") else hf_res
            my_res_hidden = res_export(my_hidden, tgt_sizes=tgt_sizes_hf)
        print("resampler hf shape:", getattr(hf_res_hidden, "shape", None))
        print("resampler my shape:", getattr(my_res_hidden, "shape", None))
        if hasattr(hf_res_hidden, "shape") and hasattr(my_res_hidden, "shape"):
            diff_tensor = (hf_res_hidden - my_res_hidden).abs()
            max_abs = diff_tensor.max().item()
            max_val = hf_res_hidden.abs().max().item()
            rel = max_abs / max_val if max_val > 0 else float("nan")
            print("resampler max diff:", max_abs)
            print("resampler relative diff:", rel)
        return

    with torch.inference_mode():
        # 构造共享的 inputs_embeds，供 HF llm 与自研解码器对齐
        inputs_embeds = my_model.embed_tokens(input_ids) * my_model.scale_emb
        if pixel_values is not None:
            vision_hidden = my_model.vision(pixel_values_hf, tgt_sizes=tgt_sizes_hf)
            vision_tokens = my_model.resampler(vision_hidden, tgt_sizes=tgt_sizes_hf)
            inputs_embeds = scatter_vision_tokens(inputs_embeds, vision_tokens, image_bound)
        # HF llm 前向（避开 MiniCPMO 的 data 参数）
        position_ids = torch.arange(0, inputs_embeds.shape[1], device=device).unsqueeze(0)
        hf_out = hf_model.llm(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        my_logits, _ = my_model.decoder(inputs_embeds, past_key_values=None)

    print("HF logits shape:", hf_out.logits.shape)
    print("My logits shape:", my_logits.shape)
    diff = (hf_out.logits - my_logits).abs().max().item()
    max_val = hf_out.logits.abs().max().item()
    rel = diff / max_val if max_val > 0 else float("nan")
    print("max diff:", diff)
    print("relative diff:", rel)


if __name__ == "__main__":
    main()
