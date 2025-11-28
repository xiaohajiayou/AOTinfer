import os
import logging
import argparse
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import DynamicCache


class QwenForCausalLMWrapper(nn.Module):
    def __init__(self, model, config, args):
        super().__init__()
        self.model = model
        self.config = config
        self.args = args
        self.layer_num = len(model.model.layers)

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        key_cache,
        value_cache,
        cache_position,
    ):

        use_cache = True
        output_attentions = False
        output_hidden_states = False
        return_dict = True
        num_logits_to_keep = 1

        past_key_values = DynamicCache()
        past_key_values.key_cache = key_cache
        past_key_values.value_cache = value_cache
        # Avoid data-dependent int() on symbolic tensors during torch.export.
        # Use a Python integer set externally when preparing the wrapper.
        past_key_values._seen_tokens = cache_position.item()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=None,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
        )

        logits = outputs.logits

        key_cache_out = [tensor for tensor in outputs.past_key_values.key_cache]
        value_cache_out = [tensor for tensor in outputs.past_key_values.value_cache]

        topk_indices = None
        if self.args.add_topk_warper > 0:
            logging.warning("add topk to glm model")
            if self.args.topk < 0:
                raise ValueError("topk {} is invalid")
            if self.args.topk > 1:
                values, topk_indices = torch.topk(logits, k=self.args.topk, dim=-1)
            else:
                topk_indices = torch.argmax(logits, dim=-1)

        topk_indices = [topk_indices] if topk_indices is not None else []
        outputs = [logits] + key_cache_out + value_cache_out + topk_indices
        return outputs


def export_qwen_to_single_pt2(model, config, dtype, args, model_name):
    qwen_model_wrapper = QwenForCausalLMWrapper(model, config, args)
    layer_num = len(model.model.layers)

    hidden_size = config.hidden_size
    head_num = config.num_attention_heads
    head_dim = hidden_size // head_num
    num_key_value_heads = config.num_key_value_heads

    device = args.device

    # ------------- 1. example inputs -------------
    # batch 先固定为 1，不标动态，避免不必要的约束
    B = 1
    N0 = 4          # 本步 token 数（示例），>1 比较稳
    cache_len0 = 0 # 历史 cache 长度（示例），>0 让导出能看到正常路径
    total_len0 = cache_len0 + N0

    # [B, N0]
    input_ids = torch.ones([B, N0], dtype=torch.int64, device=device)
    # [B, total_len0]
    attention_mask = torch.ones([B, total_len0], dtype=torch.int64, device=device)

    # [B, N0]，注意这里用真实的 token 位置，而不是全是 lastSum
    position_ids = torch.arange(
        cache_len0,
        cache_len0 + N0,
        dtype=torch.int64,
        device=device,
    ).reshape(B, N0)

    # cache_position 是“已经看到的 token 数”，标量即可
    cache_position = torch.tensor([cache_len0], dtype=torch.int64, device=device)

    # KV cache: [B, num_kv_heads, cache_len0, head_dim]
    kv_cache_shape = [B, num_key_value_heads, cache_len0, head_dim]

    key_cache = [
        torch.randn(kv_cache_shape, dtype=dtype, device=device)
        for _ in range(layer_num)
    ]
    value_cache = [
        torch.randn(kv_cache_shape, dtype=dtype, device=device)
        for _ in range(layer_num)
    ]



    example_inputs = (
        input_ids,
        attention_mask,
        position_ids,
        key_cache,
        value_cache,
        cache_position,
    )

    # ------------- 2. 动态维度声明 -------------
    # 官方推荐：用 Dim("名字") 来表达语义，重复使用同一个 Dim 表示“同一类维度”
    N = torch.export.Dim("N", min=1)          # 本步 token 数（prefill / decode 都用这个）
    total = torch.export.Dim("total", min=1)  # 当前 total 序列长度（= cache_len + N）
    cache_len = torch.export.Dim("cache_len", min=0)  # 历史 KV cache 长度

    dynamic_shapes = {
        # [B, N]
        "input_ids": {1: N},
        # [B, total]
        "attention_mask": {1: total},
        # [B, N]
        "position_ids": {1: N},
        # key_cache/value_cache 是 list[Tensor]，为每一层的 seq 维声明动态 cache_len
        "key_cache": [{} for _ in range(layer_num)],
        "value_cache": [{} for _ in range(layer_num)],
        "cache_position": {}
    }

    exported = torch.export.export(
        qwen_model_wrapper,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
    )

    # ------------- 3. AOTInductor AOTI 编译+打包 -------------
    os.makedirs(args.out_dir, exist_ok=True)
    package_path = os.path.join(args.out_dir, f"{model_name}.pt2")

    out = torch._inductor.aoti_compile_and_package(
        exported,
        package_path=package_path,
    )

    print("AOTI PT2 package:", out)



def export_qwen(args):
    device = args.device
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    print(f"begin load model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map=device, trust_remote_code=True, torch_dtype=dtype).eval()

    model.model.layers = model.model.layers[:1]  # debug

    print(f"finish load model from {args.model_path}")
    config = model.config
    print("config:", config)

    print(f"begin export qwen")
    export_qwen_to_single_pt2(model, config, dtype, args, "qwen2")


def verify_qwen(args):
    device = args.device
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    pt2_path = os.path.join(args.out_dir, "qwen2.pt2")

    # print(f"begin load pt model from {args.out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    compiled_model = torch._inductor.aoti_load_package(pt2_path, device_index=0)
    loader = compiled_model.loader
    # print("call_spec:", loader.get_call_spec())
    
    # KV cache：形状 [B, num_kv_heads, cache_len, head_dim]
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = hidden_size // num_heads
    layer_num = config.num_hidden_layers
    
    prompt = "你好，简单介绍一下你自己。"
    tok = tokenizer(prompt, return_tensors="pt")
    input_ids = tok["input_ids"].to(device)        # [B, N]
    B, L0 = input_ids.shape
    assert B == 1
    max_new_tokens = 1
    
    # ===== HF 整句生成（greedy） =====
    with torch.no_grad():
        hf_out = hf_model.generate(
            **{k: v.to(device) for k, v in tok.items()},
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    hf_ids = hf_out[0]  # [L0 + K]
    hf_text = tokenizer.decode(hf_ids, skip_special_tokens=True)
    print("HF generate:")
    print(hf_text)    
    
    # =====  prefill：一次性喂完整 prompt =====
    cache_len = 0
    N = L0
    total_len = cache_len + N
    
    attention_mask = torch.ones(B, total_len, dtype=torch.long, device = device)
    position_ids = torch.arange(0, total_len, dtype = torch.long, device = device).view(B, N)
    cache_position = torch.tensor([cache_len], dtype = torch.long, device = device)
    
    kv_shape = (B, num_kv_heads, cache_len, head_dim)
    # 用空的（长度 0）cache 作为“没有历史”的表示
    key_cache = [
        torch.empty(kv_shape, dtype=dtype, device=device)
        for _ in range(layer_num)
    ]
    value_cache = [
        torch.empty(kv_shape, dtype=dtype, device=device)
        for _ in range(layer_num)
    ]   
    inputs = [input_ids, attention_mask, position_ids]
    inputs += list(key_cache)
    inputs += list(value_cache)
    inputs.append(cache_position)
    
    with torch.no_grad():
        outputs = loader.run(inputs)
    logits = outputs[0]      # [B, N, vocab_size]
    key_cache = outputs[1:1 + layer_num]
    value_cache = outputs[1 + layer_num:1 + 2 * layer_num]
    cache_len += N
    
    generated = input_ids                        # 当前生成序列
    
    # =====  decode loop：一步一步生成 =====
    for step in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim = -1)
        generated = torch.cat([generated, next_token[:, None]], dim = 1)
        N = 1
        total_len = cache_len + N
        
        new_input_ids = next_token.view(B, N) # [B, 1]
        attention_mask = torch.ones(B, total_len, dtype=torch.long, device = device)
        position_ids = torch.arange(cache_len, cache_len + N, dtype = torch.long, device = device).view(B, N)
        cache_position = torch.tensor([cache_len], dtype = torch.long, device = device)
        inputs = [next_token, attention_mask, position_ids]
        inputs += list(key_cache)
        inputs += list(value_cache)
        inputs.append(cache_position)
        with torch.no_grad():
            outputs = loader.run(inputs)
        logits = outputs[0]      # [B, N, vocab_size]
        key_cache = outputs[1:1 + layer_num]
        value_cache = outputs[1 + layer_num:1 + 2 * layer_num]
        cache_len += N
        
        if tokenizer.eos_token_id is not None and int(next_token[0]) == tokenizer.eos_token_id:     
            break
    aoti_ids = generated[0]
    aoti_text = tokenizer.decode(aoti_ids, skip_special_tokens=True)
    print("AOTI generate:")
    print(aoti_text) 
    
    # ===== token 序列对比 =====
    # 截到相同长度比较（HF 可能提前 eos）
    min_len = min(hf_ids.shape[0], aoti_ids.shape[0])
    eq_all = torch.equal(hf_ids[:min_len].cpu(), aoti_ids[:min_len].cpu())
    print(f"token sequence equal (up to min length)? {eq_all}")

    if not eq_all:
        # 找第一个不一致位置
        diff = (hf_ids[:min_len] != aoti_ids[:min_len]).nonzero(as_tuple=False)
        first_diff = int(diff[0].item()) if diff.numel() > 0 else None
        if first_diff is not None:
            print(
                f"first diff at pos {first_diff}: "
                f"HF={int(hf_ids[first_diff])}, AOTI={int(aoti_ids[first_diff])}"
            ) 
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='export qwen',
    )
    parser.add_argument('-m', '--model_path', required=False, type=str, default="/home/cdipc03/models/Qwen/Qwen2-0.5B")
    parser.add_argument('-o', '--out_dir', required=False, type=str, default="/home/cdipc03/models/pt2/Qwen2-0.5B")
    parser.add_argument('-d', '--device', required=False, type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument('-p', '--dtype', required=False, type=str,
                        choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument('--add_topk_warper', action='store_true')
    parser.add_argument('--topk', required=False, type=int, default=4)
    args = parser.parse_args()


    export_qwen(args)
    
    verify_qwen(args)
