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
import types
import torch
from PIL import Image
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoProcessor

from minicpm.preprocess.image_preprocess import preprocess_images
# from minicpm.models.model import build_from_hf
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
    ap.add_argument("--export-resampler", action="store_true", help="自研 resampler 走 export_mode 路径")
    ap.add_argument("--use-export-for-hf", action="store_true", help="使用自研 ResamplerExport 代替 HF resampler，排查差异")
    ap.add_argument("--hf-resampler-math", action="store_true", help="强制 HF resampler 使用 math 路径（禁用 fast path）")
    ap.add_argument("--compare-embeds", action="store_true", help="对比 HF vs 自研 inputs_embeds（scatter 后）")
    ap.add_argument("--fp32-resampler", action="store_true", help="Resampler 前向强制 FP32 计算（用于精度对比）")
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model_dir = args.model_dir
    prompt = args.prompt

    # 强制关闭 flash / mem-efficient SDPA，确保与自研 math 路径对齐
    if device.startswith("cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        # 同时关闭 torch 的 MHA fast path，避免 _native_multi_head_attention
        torch.backends.cuda.enable_cuda_matmul_allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

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
    if args.hf_resampler_math:
        attn = hf_model.resampler.attn

        def slow_forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
            out, _ = self.multi_head_attention_forward(
                query,
                key,
                value,
                embed_dim_to_check=self.embed_dim,
                num_heads=self.num_heads,
                in_proj_weight=self.in_proj_weight,
                in_proj_bias=self.in_proj_bias,
                bias_k=self.bias_k,
                bias_v=self.bias_v,
                add_zero_attn=self.add_zero_attn,
                dropout_p=0.0,
                out_proj_weight=self.out_proj.weight,
                out_proj_bias=self.out_proj.bias,
                training=False,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=False,
                q_proj_weight=None,
                k_proj_weight=None,
                v_proj_weight=None,
                average_attn_weights=True,
                is_causal=False,
            )
            return out, None

        attn.forward = types.MethodType(slow_forward, attn)

    my_model = build_from_hf(hf_model).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    if not args.no_vision:
        # 文本部分：使用 processor 生成占位符/边界
        final_text = (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n(<image>./</image>)\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        proc = processor(
            text=[final_text],
            images=[Image.fromarray(load_image(args.image))],
            return_tensors="pt",
            max_slice_nums=args.max_slices,
            add_special_tokens=True,
            use_image_id=False,
            chunk_input=True,
        )
        input_ids = proc["input_ids"]
        attention_mask = proc["attention_mask"]
        image_bound = proc.get("image_bound", None)
        if image_bound is not None and isinstance(image_bound, (list, tuple)):
            image_bound = torch.as_tensor(image_bound[0], dtype=torch.long)
        if image_bound is not None:
            # 确保 batch 维存在
            if image_bound.dim() == 2:
                image_bound = image_bound.unsqueeze(0)
        # 视觉自研预处理
        img_np = load_image(args.image)
        img_size = processor.image_processor.scale_resolution if hasattr(processor, "image_processor") else 448
        patch_size = processor.image_processor.patch_size if hasattr(processor, "image_processor") else 14
        pixel_list, tgt_sizes_np, slice_counts = preprocess_images(
            [img_np],
            max_slice_nums=args.max_slices,
            scale_resolution=img_size,
            patch_size=patch_size,
        )
        num_slices = slice_counts[0]
        if num_slices > 0:
            pixel_values_hf = torch.stack(
                [torch.from_numpy(pv).to(device=device, dtype=dtype) for pv in pixel_list[:num_slices]], dim=0
            )
            tgt_sizes_hf = torch.from_numpy(tgt_sizes_np[:num_slices]).to(device)
        else:
            pixel_values_hf = torch.empty((0, 3, patch_size, 0), device=device, dtype=dtype)
            tgt_sizes_hf = None
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if image_bound is not None:
            image_bound = image_bound.to(device)
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
        )
        # 对齐 dtype/device
        res_export = res_export.to(device=device, dtype=res_export.queries.dtype)
        hf_model.resampler = hf_model.resampler.to(device=device, dtype=hf_model.resampler.kv_proj.weight.dtype)
        with torch.inference_mode():
            hf_vis = vpm(pixel_values=pixel_values_hf, tgt_sizes=tgt_sizes_hf)
            hf_hidden = hf_vis.last_hidden_state if hasattr(hf_vis, "last_hidden_state") else hf_vis
            my_hidden = vpm_export(pixel_values_hf, tgt_sizes=tgt_sizes_hf)
            # 按需升精度对齐
            if args.fp32_resampler:
                hf_hidden = hf_hidden.float()
                my_hidden = my_hidden.float()
                res_export = res_export.float()
                hf_model.resampler = hf_model.resampler.float()
            if args.use_export_for_hf:
                hf_resampler_export = ResamplerExport.from_hf(
                    hf_model.resampler, num_heads=hf_model.config.vision_config.num_attention_heads
                ).to(device=device, dtype=res_export.queries.dtype)
                hf_res_hidden = hf_resampler_export(
                    hf_hidden, tgt_sizes=tgt_sizes_hf, export_mode=args.export_resampler
                )
            else:
                hf_res = hf_model.resampler(hf_hidden, tgt_sizes=tgt_sizes_hf)
                hf_res_hidden = hf_res.last_hidden_state if hasattr(hf_res, "last_hidden_state") else hf_res

            # 自研 resampler 输入选择：默认用 hf_hidden，避免视觉误差放大；如需恢复可改为 my_hidden
            res_input = hf_hidden
            my_res_hidden = res_export(res_input, tgt_sizes=tgt_sizes_hf, export_mode=args.export_resampler)
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

    # inputs_embeds 对比（scatter 后）
    if args.compare_embeds:
        with torch.inference_mode():
            # HF 路径：用 HF get_vllm_embedding 得到 inputs_embeds
            hf_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if pixel_values_hf is not None:
                # get_vllm_embedding 期望 pixel_values/tgt_sizes/image_bound 为 batch list
                hf_inputs["pixel_values"] = [list(pixel_values_hf)]  # [B][num_slices x 3x14xW]
                if tgt_sizes_hf is not None:
                    hf_inputs["tgt_sizes"] = [tgt_sizes_hf]
                if image_bound is not None:
                    hf_inputs["image_bound"] = [image_bound[0] if image_bound.dim() == 3 else image_bound]
            hf_embeds, _ = hf_model.get_vllm_embedding(hf_inputs)

            # 自研路径
            inputs_embeds = my_model.embed_tokens(input_ids) * my_model.scale_emb
            if pixel_values_hf is not None and image_bound is not None:
                vis_tokens = my_model.resampler(my_model.vision(pixel_values_hf, tgt_sizes=tgt_sizes_hf), tgt_sizes=tgt_sizes_hf)
                vis_tokens = vis_tokens.to(inputs_embeds.dtype)
                inputs_embeds = scatter_vision_tokens(inputs_embeds, vis_tokens, image_bound)

        diff = (hf_embeds - inputs_embeds).abs()
        max_abs = diff.max().item()
        max_val = hf_embeds.abs().max().item()
        rel = max_abs / max_val if max_val > 0 else float("nan")
        print("inputs_embeds shape:", hf_embeds.shape)
        print("inputs_embeds max diff:", max_abs)
        print("inputs_embeds relative diff:", rel)
        return

    with torch.inference_mode():
        # 构造共享的 inputs_embeds，供 HF llm 与自研解码器对齐
        inputs_embeds = my_model.embed_tokens(input_ids) * my_model.scale_emb
        if pixel_values_hf is not None:
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
