"""
Quick probe to see whether HF MiniCPM vpm+resampler forward is torch.export friendly.
This does NOT solve export; it just reports success/failure with a dummy input.
"""

import torch
from transformers import AutoModel


class HFVisModule(torch.nn.Module):
    def __init__(self, vpm, resampler):
        super().__init__()
        self.vpm = vpm
        self.resampler = resampler

    def forward(self, pixel_values: torch.Tensor, tgt_sizes: torch.Tensor):
        vision_out = self.vpm(pixel_values, patch_attention_mask=None, tgt_sizes=tgt_sizes)
        hidden = vision_out.last_hidden_state
        tokens = self.resampler(hidden, tgt_sizes=tgt_sizes)
        return tokens


def main():
    # 使用无连字符/点的路径可避免部分工具误判 repo_id
    model_dir = "/home/liwenxiao/models/minicpm_o_2_6"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    hf_model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
        local_files_only=True,
        attn_implementation="eager",
        init_audio=False,
        init_tts=False,
        init_vision=True,
    ).eval()
    module = HFVisModule(hf_model.vpm, hf_model.resampler)

    # dummy inputs
    S_max = 2
    H = W = 980
    pixel_values = torch.zeros((S_max, 3, H, W), dtype=dtype, device=device)
    tgt_sizes = torch.tensor([[H // 14, W // 14]] * S_max, dtype=torch.int32, device=device)

    try:
        exported = torch.export.export(module, (pixel_values, tgt_sizes))
        print("[probe] export success, tokens shape:", exported.module()(pixel_values, tgt_sizes).shape)
    except Exception as e:
        print("[probe] export failed:", repr(e))


if __name__ == "__main__":
    main()
