#!/usr/bin/env python3
"""
MiniCPM 推理/对比脚本（HF / 自研 Wrapper），runner 仅封装三段模型调用，prefill+decode 在外部统一。
 python minicpm/run_minicpmo_hf_demo.py   --model-path /home/liwenxiao/models/minicpm_o_2_6   --image /home/liwenxiao/AOTinfer/qwen2_5_vl/test.png      --device cuda    --use-wrapper --compare-hf
"""

import argparse
import os
import sys
import json
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from minicpm.models.model import build_from_hf
from minicpm.models.adapter import scatter_vision_tokens
from minicpm.preprocess.image_preprocess import preprocess_images


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


class BaseRunner:
    def encode_image(self, pixel_values, tgt_sizes):
        raise NotImplementedError

    def build_inputs(self, input_ids, vision_tokens, image_bound):
        raise NotImplementedError

    def llm_step(self, inputs_embeds, past_kv=None, attention_mask=None):
        raise NotImplementedError


class HFRunner(BaseRunner):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def encode_image(self, pixel_values, tgt_sizes):
        if pixel_values is None:
            return None
        vis = self.model.vpm(pixel_values=pixel_values, tgt_sizes=tgt_sizes)
        vis_h = vis.last_hidden_state if hasattr(vis, "last_hidden_state") else vis
        return self.model.resampler(vis_h, tgt_sizes=tgt_sizes)

    @torch.inference_mode()
    def build_inputs(self, input_ids, vision_tokens, image_bound):
        embeds = self.model.llm.model.embed_tokens(input_ids) * getattr(self.model.llm.config, "scale_emb", 1.0)
        if vision_tokens is not None and image_bound is not None:
            embeds = scatter_vision_tokens(embeds, vision_tokens.to(embeds.dtype), image_bound)
        return embeds

    @torch.inference_mode()
    def llm_step(self, inputs_embeds, past_kv=None, attention_mask=None):
        cache_len = past_kv[0][0].shape[2] if past_kv else 0
        position_ids = torch.arange(cache_len, cache_len + inputs_embeds.shape[1], device=inputs_embeds.device).view(1, -1)
        out = self.model.llm(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )
        return out.logits, out.past_key_values


class WrapperRunner(BaseRunner):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def encode_image(self, pixel_values, tgt_sizes):
        if pixel_values is None:
            return None
        vis_h = self.model.vision(pixel_values, tgt_sizes=tgt_sizes)
        return self.model.resampler(vis_h, tgt_sizes=tgt_sizes)

    @torch.inference_mode()
    def build_inputs(self, input_ids, vision_tokens, image_bound):
        embeds = self.model.embed_tokens(input_ids) * self.model.scale_emb
        if vision_tokens is not None and image_bound is not None:
            embeds = scatter_vision_tokens(embeds, vision_tokens.to(embeds.dtype), image_bound)
        return embeds

    @torch.inference_mode()
    def llm_step(self, inputs_embeds, past_kv=None, attention_mask=None):
        logits, new_kv = self.model.decoder(inputs_embeds, past_key_values=past_kv)
        return logits, new_kv


class AOTIRunner(BaseRunner):
    """
    简单的三段 AOTI runner，加载 vision/embed/llm 三个 pt2 包。
    假设各包的 run 接口与导出时的输入顺序一致。
    """

    def __init__(self, vision_path, embed_path, llm_path, device_index: int, dtype: torch.dtype):
        # 确保 codecache 存在
        try:
            import importlib

            spec = importlib.util.find_spec("torch._inductor.codecache")
            if spec is not None:
                import torch._inductor.codecache as _cc  # type: ignore

                torch._inductor.codecache = _cc  # type: ignore[attr-defined]
        except Exception:
            pass
        self.device_index = device_index
        self.dtype = dtype
        self.vision = torch._inductor.aoti_load_package(vision_path, device_index=device_index).loader
        self.embed = torch._inductor.aoti_load_package(embed_path, device_index=device_index).loader
        self.llm = torch._inductor.aoti_load_package(llm_path, device_index=device_index).loader

        # 读取导出时的元信息
        meta_path = os.path.join(os.path.dirname(llm_path), "minicpm_export_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta file not found: {meta_path}")
        with open(meta_path, "r") as f:
            meta = json.load(f)
        self.num_layers = meta["num_layers"]
        self.num_kv_heads = meta["num_kv_heads"]
        self.head_dim = meta["head_dim"]
        self.hidden_size = meta["hidden_size"]

    @torch.inference_mode()
    def encode_image(self, pixel_values, tgt_sizes):
        if pixel_values is None:
            return None
        # 输入 shape: (S,3,14,W_flat), (S,2)
        out = self.vision.run(pixel_values, tgt_sizes)
        return out

    @torch.inference_mode()
    def build_inputs(self, input_ids, vision_tokens, image_bound):
        if vision_tokens is None or image_bound is None:
            vision_tokens = torch.zeros(
                (0, 64, self.hidden_size), device=input_ids.device, dtype=self.dtype
            )
            image_bound = torch.zeros((input_ids.shape[0], 0, 2), device=input_ids.device, dtype=torch.long)
        out = self.embed.run(input_ids, vision_tokens, image_bound)
        return out

    @torch.inference_mode()
    def llm_step(self, inputs_embeds, past_kv=None, attention_mask=None):
        if past_kv is None:
            # 以空 KV 初始化，形状需与导出一致
            bsz = inputs_embeds.shape[0]
            kv = []
            for _ in range(self.num_layers):
                kv.append(
                    (
                        torch.zeros(bsz, self.num_kv_heads, 0, self.head_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                        torch.zeros(bsz, self.num_kv_heads, 0, self.head_dim, device=inputs_embeds.device, dtype=inputs_embeds.dtype),
                    )
                )
            past_kv = tuple(kv)
        out = self.llm.run(inputs_embeds, past_kv)
        logits, new_kv = out[0], out[1]
        return logits, new_kv


def greedy_generate(
    runner: BaseRunner,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vision_tokens: Optional[torch.Tensor],
    image_bound: Optional[torch.Tensor],
    max_new_tokens: int,
    tokenizer,
):
    # prefill
    inputs_embeds = runner.build_inputs(input_ids, vision_tokens, image_bound)
    logits, kv = runner.llm_step(inputs_embeds, past_kv=None, attention_mask=attention_mask)
    generated: List[torch.Tensor] = [logits[:, -1, :].argmax(dim=-1)]
    attn_mask = attention_mask

    for _ in range(max_new_tokens):
        cur_input_ids = generated[-1].unsqueeze(1)
        cur_emb = runner.build_inputs(cur_input_ids, None, None)
        if isinstance(runner, HFRunner) and attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.ones_like(attn_mask[:, :1])], dim=1)
        logits, kv = runner.llm_step(cur_emb, past_kv=kv, attention_mask=attn_mask if isinstance(runner, HFRunner) else None)
        next_id = logits[:, -1, :].argmax(dim=-1)
        generated.append(next_id)
        if tokenizer.eos_token_id is not None and (next_id == tokenizer.eos_token_id).all():
            break
    gen_ids = torch.cat(generated, dim=0)
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="/home/liwenxiao/models/minicpm_o_2_6")
    ap.add_argument("--prompt", type=str, default="请描述图片里的内容。")
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--torch-dtype", type=str, default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--use-wrapper", action="store_true", help="使用自研前向 runner；否则默认 HF")
    ap.add_argument("--compare-hf", action="store_true", help="同时跑 HF runner 对比输出（仅 use-wrapper 时生效）")
    ap.add_argument("--vision-pt", type=str, help="AOTI vision_resampler pt2 路径")
    ap.add_argument("--embed-pt", type=str, help="AOTI embed_scatter pt2 路径")
    ap.add_argument("--llm-pt", type=str, help="AOTI llm pt2 路径")
    ap.add_argument("--device-index", type=int, default=-1, help="AOTI 设备索引，GPU 用 0/1，CPU 用 -1")
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device
    dtype = getattr(torch, args.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    image = load_image(args.image) if args.image else None
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
        init_vision=not args.no_vision,
        init_audio=False,
        init_tts=False,
        local_files_only=True,
    ).eval()

    # 构造模板文本
    system_block = (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    )
    if image is not None:
        user_block = f"<|im_start|>user\n(<image>./</image>)\n{args.prompt}<|im_end|>\n"
    else:
        user_block = f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
    final_text = system_block + user_block + "<|im_start|>assistant\n"

    # 文本+占位符处理
    if image is not None:
        proc = processor(
            text=[final_text],
            images=[image],
            return_tensors="pt",
            max_slice_nums=processor.image_processor.max_slice_nums if hasattr(processor, "image_processor") else 2,
            add_special_tokens=True,
            use_image_id=True,
            chunk_input=True,
        )
    else:
        proc = processor(text=[final_text], return_tensors="pt", add_special_tokens=True)
    input_ids = proc["input_ids"].to(device)
    attention_mask = proc["attention_mask"].to(device)
    image_bound = proc.get("image_bound", None)
    if image_bound is not None:
        if isinstance(image_bound, (list, tuple)):
            image_bound = torch.as_tensor(image_bound[0], dtype=torch.long, device=device).unsqueeze(0)
        else:
            image_bound = image_bound.to(device)

    # 视觉预处理（自研）
    pixel_values = tgt_sizes = None
    if image is not None:
        img_np = np.array(image.convert("RGB"))
        img_size = processor.image_processor.scale_resolution if hasattr(processor, "image_processor") else 448
        patch_size = processor.image_processor.patch_size if hasattr(processor, "image_processor") else 14
        s_max = processor.image_processor.max_slice_nums if hasattr(processor, "image_processor") else 2
        pixel_list, tgt_sizes_np, slice_counts = preprocess_images(
            [img_np],
            max_slice_nums=s_max,
            scale_resolution=img_size,
            patch_size=patch_size,
        )
        num_slices = slice_counts[0]
        if num_slices > 0:
            pixel_values = torch.stack(
                [torch.from_numpy(pv).to(device=device, dtype=dtype) for pv in pixel_list[:num_slices]], dim=0
            )
            tgt_sizes = torch.from_numpy(tgt_sizes_np[:num_slices]).to(device)

    # runner 选择：优先 AOTI，其次 wrapper，默认 HF
    if args.vision_pt and args.embed_pt and args.llm_pt:
        runner = AOTIRunner(args.vision_pt, args.embed_pt, args.llm_pt, device_index=args.device_index, dtype=dtype)
    elif args.use_wrapper:
        custom_model = build_from_hf(model).to(device)
        runner = WrapperRunner(custom_model, tokenizer)
    else:
        runner = HFRunner(model, tokenizer)

    # 推理
    vision_tokens = runner.encode_image(pixel_values, tgt_sizes)
    text_out = greedy_generate(
        runner,
        input_ids,
        attention_mask,
        vision_tokens,
        image_bound,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
    )
    print("=== Runner 输出 ===")
    print(text_out)

    if args.compare_hf and (args.use_wrapper or args.vision_pt):
        hf_runner = HFRunner(model, tokenizer)
        vision_tokens_hf = hf_runner.encode_image(pixel_values, tgt_sizes)
        hf_text = greedy_generate(
            hf_runner,
            input_ids,
            attention_mask,
            vision_tokens_hf,
            image_bound,
            max_new_tokens=args.max_new_tokens,
            tokenizer=tokenizer,
        )
        print("=== HF Runner 输出 ===")
        print(hf_text)


if __name__ == "__main__":
    main()
