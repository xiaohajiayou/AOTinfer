#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

// A simple runner wrapper around an AOTI LLM package.
// Keeps the design similar to your Python AOTIRunner.llm_step.
class ModelRunner {
 public:
  ModelRunner(const std::string& llm_path,
              int device_index,
              int64_t num_layers,
              int64_t num_kv_heads,
              int64_t head_dim);

  ModelRunner(const ModelRunner&) = delete;
  ModelRunner& operator=(const ModelRunner&) = delete;
  ~ModelRunner() = default;

  // One LLM step. If key_cache / value_cache are empty, this function will
  // initialize them (equivalent to your Python code's "if key_cache is None").
  //
  // Returns logits tensor. key_cache, value_cache and cache_len_tensor
  // are updated in-place.
  torch::Tensor LlmStep(const torch::Tensor& inputs_embeds,
                        std::vector<torch::Tensor>* key_cache,
                        std::vector<torch::Tensor>* value_cache,
                        torch::Tensor* cache_len_tensor);

  int64_t num_layers() const { return num_layers_; }
  int64_t num_kv_heads() const { return num_kv_heads_; }
  int64_t head_dim() const { return head_dim_; }

 private:
  void InitializeCache(const torch::Tensor& inputs_embeds,
                       std::vector<torch::Tensor>* key_cache,
                       std::vector<torch::Tensor>* value_cache,
                       torch::Tensor* cache_len_tensor) const;

  torch::inductor::AOTIModelPackageLoader llm_;
  int64_t num_layers_;
  int64_t num_kv_heads_;
  int64_t head_dim_;
};
