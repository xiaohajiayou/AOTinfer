#!/usr/bin/env python3
"""
MiniCPM 推理/对比脚本（HF / 自研 Wrapper），runner 仅封装三段模型调用，prefill+decode 在外部统一。
 python minicpm/run_minicpmo_hf_demo.py   --model-dir /home/liwenxiao/models/minicpm_o_2_6   --image /home/liwenxiao/AOTinfer/qwen2_5_vl/test.png      --device cuda    --use-wrapper --compare-hf
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

# add project root to sys.path so `minicpm` can be imported when running directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from minicpm.models.adapter import MiniCPMAdapter
from minicpm.export_minicpm_aoti import flatten_inputs, unflatten_outputs
# torch._inductor.config.cache_size_limit = 2 * 1024 * 1024 * 1024  # 2GB
# 禁用不必要的调试缓存
torch._inductor.config.debug = False
torch._inductor.config.trace.enabled = False
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/root/autodl-tmp/torch_inductor_cache"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# from minicpm.models.model import build_from_hf
from minicpm.preprocess.image_preprocess import preprocess_images


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def scatter_vision_tokens(
    text_embeds: torch.Tensor,
    vision_tokens: Optional[torch.Tensor],
    image_bound: Optional[torch.Tensor],
) -> torch.Tensor:
    """用视觉 token 替换文本 embedding 中的占位符。"""
    if (
        vision_tokens is None
        or image_bound is None
        or image_bound.numel() == 0
        or vision_tokens.numel() == 0
    ):
        return text_embeds

    new_embeds = text_embeds.clone()
    max_vis = vision_tokens.shape[1]
    seq_len = new_embeds.shape[1]

    for b in range(image_bound.size(0)):
        bounds = image_bound[b]
        for m in range(bounds.size(0)):
            start = bounds[m, 0].item()
            end = bounds[m, 1].item()
            length = max(end - start, 0)
            if length == 0:
                continue

            idx = torch.arange(max_vis, device=new_embeds.device)
            target_pos = start + idx
            valid = (idx < length) & (target_pos < seq_len)
            target_indices = target_pos[valid].long()

            src = vision_tokens[b, valid].to(new_embeds.dtype)
            if target_indices.numel() > 0:
                new_embeds[b].index_copy_(0, target_indices, src)

    return new_embeds


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
        embeds = (
            self.model.llm.model.embed_tokens(input_ids)
        )
        if vision_tokens is not None and image_bound is not None:
            embeds = scatter_vision_tokens(
                embeds, vision_tokens.to(embeds.dtype), image_bound
            )
        return embeds

    @torch.inference_mode()
    def llm_step(self, inputs_embeds, key_cache=None, value_cache=None, cache_len=None, attention_mask=None):
        cache_len_val = 0 if cache_len is None else int(cache_len.item()) if torch.is_tensor(cache_len) else int(cache_len)
        past_kv = None
        if key_cache is not None and value_cache is not None:
            past_kv = tuple(zip(key_cache, value_cache))
        position_ids = torch.arange(cache_len_val, cache_len_val + inputs_embeds.shape[1],
                                    device=inputs_embeds.device).view(1, -1)
        out = self.model.llm(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_kv,
            use_cache=True,
            return_dict=True,
        )
        new_key = tuple(k for k, v in out.past_key_values)
        new_value = tuple(v for k, v in out.past_key_values)
        new_cache_len = torch.tensor(cache_len_val + inputs_embeds.shape[1],
                                     device=inputs_embeds.device, dtype=torch.long)
        return out.logits, new_key, new_value, new_cache_len


class WrapperRunner(BaseRunner):
    def __init__(self, wrapper_adapter, vision_path, embed_path, llm_path, device_index: int):
        self.adapter = wrapper_adapter
        self.vision, self.embed, self.llm = self.adapter.get_export_modules()
        # self.vision = torch._inductor.aoti_load_package(vision_path, device_index=device_index)
        # self.embed = torch._inductor.aoti_load_package(embed_path, device_index=device_index)
        # self.llm = torch._inductor.aoti_load_package(llm_path, device_index=device_index)
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
        out = self.vision(pixel_values, tgt_sizes)
        return out

    @torch.inference_mode()
    def build_inputs(self, input_ids, vision_tokens, image_bound):
        input_ids = input_ids.to(torch.long)
        embeds = self.embed(input_ids)  # HF: model.llm.model.embed_tokens；AOTI: PT2 embed
        if vision_tokens is not None and image_bound is not None:
            embeds = scatter_vision_tokens(embeds, vision_tokens, image_bound)
        return embeds

    # @torch.inference_mode()
    # def llm_step(self, inputs_embeds, key_cache=None, value_cache=None, cache_len=None, attention_mask=None):
    #     if key_cache is None or value_cache is None:
    #         key_cache = []
    #         value_cache = []
    #         for _ in range(len(self.model.decoder.layers)):
    #             key_cache.append(torch.zeros(inputs_embeds.size(0), self.model.decoder.num_kv_heads, 0,
    #                                          self.model.decoder.head_dim, device=inputs_embeds.device,
    #                                          dtype=inputs_embeds.dtype))
    #             value_cache.append(torch.zeros_like(key_cache[-1]))
    #     if cache_len is None:
    #         cache_len = torch.tensor(0, device=inputs_embeds.device, dtype=torch.long)
    #     logits, key_cache, value_cache, cache_len = self.model.decoder(inputs_embeds, key_cache, value_cache, cache_len)
    #     return logits, key_cache, value_cache, cache_len

    @torch.inference_mode()
    def llm_step(self, inputs_embeds, key_cache=None, value_cache=None, cache_len=None, attention_mask=None):
        bsz = inputs_embeds.shape[0]
        if key_cache is None or value_cache is None:
            key_cache = [
                torch.zeros(
                    bsz,
                    self.num_kv_heads,
                    0,
                    self.head_dim,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
                for _ in range(self.num_layers)
            ]
            value_cache = [torch.zeros_like(k) for k in key_cache]
            cache_len_tensor = torch.tensor(0, dtype=torch.long, device=inputs_embeds.device)
        else:
            cache_len_tensor = cache_len if isinstance(cache_len, torch.Tensor) else torch.tensor(cache_len, device=inputs_embeds.device, dtype=torch.long)

        # cache_len_tensor = cache_len_tensor.to(torch.int32)
        out = self.llm(inputs_embeds, key_cache, value_cache, cache_len_tensor)
        logits, new_key, new_value, new_cache_len = out
        return logits, new_key, new_value, new_cache_len


class AOTIRunner(BaseRunner):
    """
    简单的三段 AOTI runner，加载 vision/embed/llm 三个 pt2 包。
    假设各包的 run 接口与导出时的输入顺序一致。
    """

    def __init__(self, vision_path, embed_path, llm_path, device_index: int, dtype: torch.dtype):
        # 确保 codecache 存在
        # try:
        #     import importlib

        #     spec = importlib.util.find_spec("torch._inductor.codecache")
        #     if spec is not None:
        #         import torch._inductor.codecache as _cc  # type: ignore

        #         torch._inductor.codecache = _cc  # type: ignore[attr-defined]
        # except Exception:
        #     pass
        self.vision = torch._inductor.aoti_load_package(vision_path, device_index=device_index)
        self.embed = torch._inductor.aoti_load_package(embed_path, device_index=device_index)
        self.llm = torch._inductor.aoti_load_package(llm_path, device_index=device_index)

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
        out = self.vision(pixel_values, tgt_sizes)
        return out

    @torch.inference_mode()
    def build_inputs(self, input_ids, vision_tokens, image_bound):
        input_ids = input_ids.to(torch.long)
        embeds = self.embed(input_ids)  # HF: model.llm.model.embed_tokens；AOTI: PT2 embed
        if vision_tokens is not None and image_bound is not None:
            embeds = scatter_vision_tokens(embeds, vision_tokens, image_bound)
        return embeds

    @torch.inference_mode()
    def llm_step(self, inputs_embeds, key_cache=None, value_cache=None, cache_len=None, attention_mask=None):
        bsz = inputs_embeds.shape[0]
        if key_cache is None or value_cache is None:
            key_cache = [
                torch.zeros(
                    bsz,
                    self.num_kv_heads,
                    0,
                    self.head_dim,
                    device=inputs_embeds.device,
                    dtype=inputs_embeds.dtype,
                )
                for _ in range(self.num_layers)
            ]
            value_cache = [torch.zeros_like(k) for k in key_cache]
            cache_len_tensor = torch.tensor(0, dtype=torch.long, device=inputs_embeds.device)
        else:
            cache_len_tensor = cache_len if isinstance(cache_len, torch.Tensor) else torch.tensor(cache_len, device=inputs_embeds.device, dtype=torch.long)

        # cache_len_tensor = cache_len_tensor.to(torch.int32)
        out = self.llm(inputs_embeds, key_cache, value_cache, cache_len_tensor)
        logits, new_key, new_value, new_cache_len = out
        return logits, new_key, new_value, new_cache_len
        # flat = flatten_inputs(inputs_embeds, key_cache, value_cache, cache_len)
        # outputs = self.llm.loader.run(flat)
        # return unflatten_outputs(outputs, self.num_layers)


def greedy_generate(
    runner: BaseRunner,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    vision_tokens: Optional[torch.Tensor],
    image_bound: Optional[torch.Tensor],
    max_new_tokens: int,
    tokenizer,
):
    # init cache
    key_cache = value_cache = None
    cache_len = torch.tensor(0, device=input_ids.device, dtype=torch.long)
    # prefill
    inputs_embeds = runner.build_inputs(input_ids, vision_tokens, image_bound)
    logits, key_cache, value_cache, cache_len = runner.llm_step(inputs_embeds, key_cache, value_cache, cache_len, attention_mask)
    generated = [logits[:, -1, :].argmax(dim=-1, keepdim=True)]
    attn_mask = attention_mask
    for _ in range(max_new_tokens):
        cur_emb = runner.build_inputs(generated[-1], None, None)
        if isinstance(runner, HFRunner) and attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.ones_like(attn_mask[:, :1])], dim=1)
        logits, key_cache, value_cache, cache_len = runner.llm_step(cur_emb, key_cache, value_cache, cache_len, attn_mask)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_id)
        if tokenizer.eos_token_id is not None and (next_id == tokenizer.eos_token_id).all():
            break
    gen_ids = torch.cat(generated, dim=1)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="请描述图片里的内容。")
    ap.add_argument("--model-dir", type=str, default="/home/liwenxiao/models/minicpm_o_2_6")
    ap.add_argument("--image", type=str, default="/home/liwenxiao/AOTinfer/qwen2_5_vl/test.png")
    ap.add_argument("--vision-pt", default="/home/liwenxiao/AOTinfer/minicpm/minicpm_vision_resampler.pt2", type=str, help="AOTI vision_resampler pt2 路径")
    ap.add_argument("--embed-pt", type=str, default="/home/liwenxiao/AOTinfer/minicpm/minicpm_embed.pt2", help="AOTI embed_scatter pt2 路径")
    ap.add_argument("--llm-pt", type=str, default="/home/liwenxiao/AOTinfer/minicpm/minicpm_llm.pt2", help="AOTI llm pt2 路径")

    # ap.add_argument("--model-dir", type=str, default="/root/autodl-tmp/models/MiniCPM_o_2_6")
    # ap.add_argument("--image", type=str, default="/root/autodl-tmp/AOTinfer/qwen2_5_vl/test.png")
    # ap.add_argument("--vision-pt", default="/root/autodl-tmp/AOTinfer/minicpm/minicpm_vision_resampler.pt2", type=str, help="AOTI vision_resampler pt2 路径")
    # ap.add_argument("--embed-pt", type=str, default="/root/autodl-tmp/AOTinfer/minicpm/minicpm_embed.pt2", help="AOTI embed_scatter pt2 路径")
    # ap.add_argument("--llm-pt", type=str, default="/root/autodl-tmp/AOTinfer/minicpm/minicpm_llm.pt2", help="AOTI llm pt2 路径")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--use-wrapper", action="store_true", help="使用自研前向 runner；否则默认 HF")
    ap.add_argument("--use-aoti", action="store_true", help="使用aotinductor前向 runner；否则默认 HF")
    ap.add_argument("--compare-hf", action="store_true", help="同时跑 HF runner 对比输出（仅 use-wrapper 时生效）")
    ap.add_argument("--device-index", type=int, default=-1, help="AOTI 设备索引，GPU 用 0/1，CPU 用 -1")
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device
    dtype = getattr(torch, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True)
    image = None
    if not args.no_vision:
        image = load_image(args.image) if args.image else None
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)



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
        # 无图时关闭 padding，避免自研路径把 pad token 写进 KV
        proc = processor(
            text=[final_text],
            return_tensors="pt",
            add_special_tokens=True,
            padding=False,
        )
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
            tgt_sizes = torch.from_numpy(tgt_sizes_np[:num_slices]).to(device=device, dtype=torch.int64)

    model = None
    runner = None

    # runner 选择：优先 AOTI，其次 wrapper，默认 HF
    if args.use_aoti:
    # if True:
        runner = AOTIRunner(args.vision_pt, args.embed_pt, args.llm_pt, device_index=args.device_index, dtype=dtype)
    elif args.use_wrapper:
        if model is None:
            wrapper_adapter = MiniCPMAdapter.from_pretrained(
                args.model_dir,
                device=device,
                dtype=dtype
            )
            model = wrapper_adapter.hf_model
            runner = WrapperRunner(wrapper_adapter,args.vision_pt, args.embed_pt, args.llm_pt, device_index=args.device_index)
    else:
        if model is None:
            model = AutoModel.from_pretrained(
                args.model_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=device,
                init_vision=not args.no_vision,
                init_audio=False,
                init_tts=False,
                local_files_only=True,
            ).eval()
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
    
    if args.compare_hf:
    # if True:
        if model is None:
            model = AutoModel.from_pretrained(
                args.model_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map=device,
                init_vision=not args.no_vision,
                init_audio=False,
                init_tts=False,
                local_files_only=True,
            ).eval()
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


# python /root/autodl-tmp/AOTinfer/minicpm/run_minicpmo_hf_demo.py \
#   --model-dir /root/autodl-tmp/models/MiniCPM_o_2_6 \
#   --image /root/autodl-tmp/AOTinfer/qwen2_5_vl/test.png \
#   --device cuda \
#   --dtype bfloat16 \
#   --use-aoti \
#   --compare-hf \
#   --vision-pt /root/autodl-tmp/AOTinfer/minicpm/minicpm_vision_resampler.pt2 \
#   --embed-pt /root/autodl-tmp/AOTinfer/minicpm/minicpm_embed.pt2 \
#   --llm-pt /root/autodl-tmp/AOTinfer/minicpm/minicpm_llm.pt2

# du -h -d1 /tmp /root/.cache /root /root/autodl-tmp | sort -h | tail -n 20
# du -ah . | sort -rh | head -n 10

# python /root/autodl-tmp/AOTinfer/minicpm/run_minicpmo_hf_demo.py \
#   --model-dir /root/autodl-tmp/models/MiniCPM_o_2_6 \
#   --device cuda \
#   --dtype bfloat16 \
#   --use-aoti \
#   --compare-hf \
#   --no-vision \
#   --embed-pt /root/autodl-tmp/AOTinfer/minicpm/minicpm_embed.pt2 \
#   --llm-pt /root/autodl-tmp/AOTinfer/minicpm/minicpm_llm.pt2