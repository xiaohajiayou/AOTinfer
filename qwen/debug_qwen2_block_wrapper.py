import argparse

import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from export_layers import build_causal_mask
from qwen.models.qwen2_block import Qwen2ExportBlock


@torch.no_grad()
def compare(args):
    device = args.device
    dtype = getattr(torch, args.torch_dtype)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=device,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).eval()

    layer_idx = args.layer_idx
    hf_layer = hf_model.model.layers[layer_idx]
    block = Qwen2ExportBlock(hf_layer, hf_model.config).to(device).eval()

    vocab_size = hf_model.config.vocab_size
    hidden_size = hf_model.config.hidden_size
    num_kv_heads = hf_model.config.num_key_value_heads
    head_dim = hidden_size // hf_model.config.num_attention_heads

    B = 1
    seq = args.seq_len
    cache_len = torch.tensor(args.cache_len, dtype=torch.long, device=device)

    input_ids = torch.randint(0, vocab_size, (B, seq), device=device)
    hidden_states = hf_model.model.embed_tokens(input_ids)
    rotary = hf_model.model.rotary_emb

    total_len = cache_len.item() + seq
    position_ids = torch.arange(cache_len.item(), total_len, device=device).view(1, seq)
    cos, sin = rotary(hidden_states, position_ids)
    mask = build_causal_mask(
        batch_size=B,
        n_step=seq,
        past_key_len=cache_len.item(),
        dtype=hidden_states.dtype,
        device=device,
    )

    key_cache = torch.zeros(
        B, num_kv_heads, cache_len.item(), head_dim, dtype=hidden_states.dtype, device=device
    )
    value_cache = key_cache.clone()

    out_block = block(
        hidden_states.clone(),
        cos,
        sin,
        key_cache,
        value_cache,
        cache_len,
        mask,
    )
    hidden_block, k_block, v_block, cache_len_block = out_block

    cache = DynamicCache(config=hf_model.config)
    cache._seen_tokens = cache_len.item()
    cache_position = torch.arange(cache_len.item(), total_len, device=device)

    hidden_hf = hf_layer(
        hidden_states=hidden_states.clone(),
        attention_mask=mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
        position_embeddings=(cos, sin),
    )
    layer_cache = cache.layers[layer_idx]
    k_hf = layer_cache.keys
    v_hf = layer_cache.values
    k_new_block = k_block[:, :, -seq:, :]
    v_new_block = v_block[:, :, -seq:, :]
    k_new_hf = k_hf[:, :, -seq:, :]
    v_new_hf = v_hf[:, :, -seq:, :]
    cache_len_hf = torch.tensor(k_hf.shape[2], device=device, dtype=torch.long)

    report(hidden_block, hidden_hf, "hidden")
    report(k_new_block, k_new_hf, "key_new")
    report(v_new_block, v_new_hf, "value_new")
    print(f"cache_len: block={cache_len_block.item()}, hf={cache_len_hf.item()}")


def report(tensor_a, tensor_b, name):
    diff = (tensor_a - tensor_b).abs().max().item()
    print(f"[{name}] max_abs_diff={diff:.3e}")


def parse():
    parser = argparse.ArgumentParser(description="Debug single Qwen2 layer export block")
    # parser.add_argument("-m", "--model_path", type=str, default="/home/cdipc03/models/Qwen/Qwen2-0.5B")
    parser.add_argument("-m", "--model_path", type=str, default="/Users/bruceli/Desktop/Git_sync/model/Qwen2-0.5B-Instruct")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--torch_dtype", type=str, default="float32")
    parser.add_argument("--layer_idx", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--cache_len", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    compare(parse())
