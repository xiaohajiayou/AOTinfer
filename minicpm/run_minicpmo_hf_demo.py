#!/usr/bin/env python3
"""
使用本地 HF 权重跑通 MiniCPM-o 2.6 的最小推理示例，便于后续对照和调试。

用法：
  python minicpm/run_minicpmo_hf_demo.py \
    --model-path /home/liwenxiao/models/minicpm-o-2.6 \
    --prompt "请简要描述这张图片" \
    --image /path/to/image.jpg
"""

import argparse
import os
import sys
from typing import Optional

import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# add project root to path to resolve 'minicpm' package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from minicpm.models.model import build_from_hf
from minicpm.models.adapter import scatter_vision_tokens
from minicpm.preprocess.image_preprocess import preprocess_images, pad_slices_to_tensor


def load_image(path: str) -> Image.Image:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MiniCPM-o 2.6 HF 推理 Demo")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/liwenxiao/models/minicpm_o_2_6",
        help="本地模型目录（含 config.json 等文件），建议使用无点/连字符的路径避免动态模块命名问题",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="请描述图片里的内容。",
        help="用户提问内容；纯文本模式时直接使用。",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="/home/liwenxiao/AOTinfer/qwen2_5_vl/test.png",
        help="图片路径，可选；提供后走图文多模态。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["auto", "cuda", "cpu"],
        help="推理设备，auto 优先使用 CUDA。",
    )
    parser.add_argument(
        "--attn",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="attention 实现方式。",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="生成最大新 token 数。",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="不初始化视觉塔（仅纯文本测试时可用）。",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="对比 HF 与自研前向的 logits（单步），不做生成。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 让 transformers 的动态模块缓存指向当前项目下的可写目录，避免 ~/.cache 缺文件/旧文件
    # hf_modules_cache = os.path.join(os.path.dirname(__file__), "..", ".hf_modules")
    # os.environ["HF_MODULES_CACHE"] = hf_modules_cache
    # print(f"[INFO] HF_MODULES_CACHE={hf_modules_cache}")

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"[INFO] 加载模型：{args.model_path}")
    print(f"[INFO] device={device}, dtype={dtype}, attn={args.attn}, init_vision={not args.no_vision}")

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        attn_implementation=args.attn,
        torch_dtype=dtype,
        init_vision=not args.no_vision,
        init_audio=False,
        init_tts=False,
        local_files_only=True,
    )
    model = model.eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    image: Optional[Image.Image] = load_image(args.image) if args.image else None

    if image is None and args.no_vision is False:
        print("[WARN] 未提供图片，但已初始化视觉塔；若仅做文本测试，可添加 --no-vision")

    msgs = [{"role": "user", "content": args.prompt}]
    if image is not None:
        msgs[0]["content"] = [image, args.prompt]

    if True:
        # 对比 HF 与自研前向 logits（单步）
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        # 文本模板：占位符 + 文本，交给 processor 生成 image_bound
        if image is not None:
            prompt_text = f"(<image>./</image>)\n{args.prompt}"
            proc = processor(
                text=[prompt_text],
                images=[image],
                return_tensors="pt",
                max_slice_nums=processor.image_processor.max_slice_nums if hasattr(processor, "image_processor") else 2,
                add_special_tokens=True,
                use_image_id=False,
                chunk_input=True,
            )
        else:
            proc = processor(
                text=[args.prompt],
                return_tensors="pt",
                add_special_tokens=True,
            )
        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)
        image_bound = proc.get("image_bound", None)
        if image_bound is not None:
            if isinstance(image_bound, (list, tuple)):
                # processor 返回 list[tensor]，取第一项并确保 batch 维度
                image_bound = torch.as_tensor(image_bound[0], dtype=torch.long, device=device).unsqueeze(0)
            else:
                image_bound = image_bound.to(device)

        # 视觉预处理改用自研，确保与导出一致
        pixel_values = tgt_sizes = None
        if image is not None:
            img_np = np.array(image.convert("RGB"))
            img_size = getattr(model.config.vision_config, "image_size", 980)
            patch_size = getattr(model.config.vision_config, "patch_size", 14)
            s_max = processor.image_processor.max_slice_nums if hasattr(processor, "image_processor") else 2
            pixel_list, tgt_sizes_np, slice_counts = preprocess_images(
                [img_np],
                max_slice_nums=s_max,
                scale_resolution=img_size,
                patch_size=patch_size,
            )
            pixel_values, tgt_sizes = pad_slices_to_tensor(
                pixel_list, tgt_sizes_np, s_max=s_max, h=img_size, w=img_size, device=device
            )
            pixel_values = pixel_values.to(device=device, dtype=dtype)
            num_slices = slice_counts[0]
            pixel_values = pixel_values[:num_slices]
            tgt_sizes = tgt_sizes[:num_slices].to(device)

        # 准备 HF 生成用的输入，确保在同一 device/dtype，并使用自研视觉张量以避免设备不一致
        proc_hf = {"input_ids": input_ids, "attention_mask": attention_mask}
        if image is not None:
            proc_hf["pixel_values"] = pixel_values
            proc_hf["tgt_sizes"] = tgt_sizes
            if image_bound is not None:
                proc_hf["image_bound"] = image_bound

        # 自研模型
        custom_model = build_from_hf(model).to(device)
        with torch.inference_mode():
            inputs_embeds = custom_model.embed_tokens(input_ids) * custom_model.scale_emb
            if image is not None and pixel_values is not None and image_bound is not None:
                vision_hidden = custom_model.vision(pixel_values, tgt_sizes=tgt_sizes)
                vision_tokens = custom_model.resampler(vision_hidden, tgt_sizes=tgt_sizes)
                inputs_embeds = scatter_vision_tokens(inputs_embeds, vision_tokens, image_bound)
            position_ids = torch.arange(0, inputs_embeds.shape[1], device=device).unsqueeze(0)
            hf_out = model.llm(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True,
            )
            my_logits, _ = custom_model.decoder(inputs_embeds, past_key_values=None)
        diff = (hf_out.logits - my_logits).abs()
        max_abs = diff.max().item()
        max_val = hf_out.logits.abs().max().item()
        rel = max_abs / max_val if max_val > 0 else float("nan")
        print("[COMPARE] logits max diff:", max_abs)
        print("[COMPARE] logits relative diff:", rel)

        # 简易生成对比（贪心），步数受 --max-new-tokens 限制；若失败则跳过
        print("[COMPARE] 开始贪心生成对比")
        try:
            gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False, return_dict_in_generate=False)
            position_ids = torch.arange(0, inputs_embeds.shape[1], device=device).unsqueeze(0)
            hf_gen_ids = model.llm.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **gen_kwargs,
            )
            hf_new_tokens = hf_gen_ids[0]
            hf_text = tokenizer.decode(hf_new_tokens, skip_special_tokens=True)

            # 自研生成（贪心）
            custom_kv = None
            cur_embeds = inputs_embeds
            gen_ids = []
            for _ in range(args.max_new_tokens):
                logits, custom_kv = custom_model.decoder(cur_embeds, past_key_values=custom_kv)
                next_id = logits[:, -1, :].argmax(dim=-1)
                gen_ids.append(next_id)
                cur_embeds = custom_model.embed_tokens(next_id.unsqueeze(1)) * custom_model.scale_emb
            my_ids = torch.cat(gen_ids, dim=0) if gen_ids else torch.tensor([], device=device, dtype=input_ids.dtype)
            my_text = tokenizer.decode(my_ids, skip_special_tokens=True)

            print("[COMPARE] HF 生成：", hf_text)
            print("[COMPARE] 自研生成：", my_text)
        except Exception as e:
            print(f"[COMPARE] 生成对比失败，跳过（{e}）")
        return

    print("[INFO] 开始推理...")
    with torch.inference_mode():
        answer = model.chat(
            image=image if image is not None else None,
            msgs=msgs,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            sampling=True,
            chunk_input=True,
            omni_input=False,
            use_tts_template=False,
            generate_audio=False,
            return_dict=False,
        )

    print("\n====== 模型输出 ======")
    print(answer)


if __name__ == "__main__":
    main()
