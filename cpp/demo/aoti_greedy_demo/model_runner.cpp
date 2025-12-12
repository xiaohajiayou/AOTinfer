#include "model_runner.h"

#include <iostream>
#include <stdexcept>

ModelRunner::ModelRunner(const std::string& llm_path,
                         int device_index,
                         int64_t num_layers,
                         int64_t num_kv_heads,
                         int64_t head_dim)
    : llm_(llm_path, "model", false, 1, device_index),
      num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim),
      cache_initialized_(false) {
  std::cout << "Loaded AOTI LLM package from: " << llm_path << std::endl;
}

void ModelRunner::ResetCache() {
  cache_initialized_ = false;
  key_cache_.clear();
  value_cache_.clear();
  cache_len_tensor_ = torch::Tensor();
}

void ModelRunner::InitializeCacheIfNeeded(const torch::Tensor& inputs_embeds) {
  if (cache_initialized_) {
    return;
  }

  key_cache_.clear();
  value_cache_.clear();

  const int64_t batch_size = inputs_embeds.size(0);
  const auto options = inputs_embeds.options();

  key_cache_.reserve(num_layers_);
  value_cache_.reserve(num_layers_);

  for (int64_t layer = 0; layer < num_layers_; ++layer) {
    // Shape: [B, num_kv_heads, 0, head_dim].
    torch::Tensor key =
        torch::zeros({batch_size, num_kv_heads_, 0, head_dim_}, options);
    torch::Tensor value = torch::zeros_like(key);
    key_cache_.push_back(key);
    value_cache_.push_back(value);
  }

  cache_len_tensor_ = torch::zeros(
      {}, torch::TensorOptions()
              .dtype(torch::kLong)
              .device(inputs_embeds.device()));

  cache_initialized_ = true;
}

torch::Tensor ModelRunner::LlmStep(const torch::Tensor& inputs_embeds) {
  if (!inputs_embeds.defined()) {
    throw std::invalid_argument("inputs_embeds is undefined in LlmStep.");
  }

  // Ensure cache is initialized for this batch.
  InitializeCacheIfNeeded(inputs_embeds);

  // Build input list for the AOTI model.
  //
  // ⚠️ 非常重要：这里假定导出 LLM pt2 的 forward 签名为：
  //
  //   forward(
  //     inputs_embeds,
  //     key_cache_layer0, ..., key_cache_layerN-1,
  //     value_cache_layer0, ..., value_cache_layerN-1,
  //     cache_len
  //   )
  //
  // 输出假定为：
  //   outputs[0]              = logits
  //   outputs[1 .. L]         = new_key (每层一个)
  //   outputs[1+L .. 1+2L-1]  = new_value
  //   outputs[1+2L]           = new_cache_len
  //
  // 如果你导出时顺序不同，请对应修改这里的拼接 & 解析逻辑。
  std::vector<torch::Tensor> inputs;
  inputs.reserve(1 + 2 * num_layers_ + 1);

  // 1) inputs_embeds
  inputs.push_back(inputs_embeds);

  // 2) key cache
  for (const auto& key : key_cache_) {
    inputs.push_back(key);
  }

  // 3) value cache
  for (const auto& value : value_cache_) {
    inputs.push_back(value);
  }

  // 4) cache_len
  inputs.push_back(cache_len_tensor_);

  // Run AOTI model.
  std::vector<torch::Tensor> outputs;
  try {
    outputs = llm_.run(inputs);
  } catch (const std::exception& e) {
    std::cerr << "Error running AOTI LLM: " << e.what() << std::endl;
    throw;
  }

  const int64_t expected_outputs =
      1 + 2 * num_layers_ + 1;  // logits + new_kv + new_cache_len

  if (static_cast<int64_t>(outputs.size()) != expected_outputs) {
    std::cerr << "Unexpected number of outputs from LLM. Got "
              << outputs.size() << ", expected " << expected_outputs
              << std::endl;
    throw std::runtime_error("Unexpected number of outputs from LLM.");
  }

  // 解析输出
  torch::Tensor logits = outputs[0];

  std::vector<torch::Tensor> new_key;
  std::vector<torch::Tensor> new_value;
  new_key.reserve(num_layers_);
  new_value.reserve(num_layers_);

  for (int64_t layer = 0; layer < num_layers_; ++layer) {
    new_key.push_back(outputs[1 + layer]);
  }
  for (int64_t layer = 0; layer < num_layers_; ++layer) {
    new_value.push_back(outputs[1 + num_layers_ + layer]);
  }
  torch::Tensor new_cache_len = outputs[1 + 2 * num_layers_];

  key_cache_ = std::move(new_key);
  value_cache_ = std::move(new_value);
  cache_len_tensor_ = new_cache_len;

  return logits;
}
