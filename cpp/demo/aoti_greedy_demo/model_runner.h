#ifndef MODEL_RUNNER_H_
#define MODEL_RUNNER_H_

#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

// A simple LLM runner wrapper around an AOTI model package.
// - Holds the AOTIModelPackageLoader.
// - Manages KV cache internally.
// - Exposes LlmStep() similar to your Python AOTIRunner.llm_step.
class ModelRunner {
 public:
  // Constructor.
  // llm_path: path to the compiled pt2 (AOTI) LLM package.
  // device_index: CUDA device index used when loading the package.
  // num_layers / num_kv_heads / head_dim: must match your model meta.
  ModelRunner(const std::string& llm_path,
              int device_index,
              int64_t num_layers,
              int64_t num_kv_heads,
              int64_t head_dim);

  ModelRunner(const ModelRunner&) = delete;
  ModelRunner& operator=(const ModelRunner&) = delete;
  ~ModelRunner() = default;

  // One LLM step, equivalent to Python:
  //
  //   logits, new_key, new_value, new_cache_len = llm(
  //       inputs_embeds, key_cache, value_cache, cache_len)
  //
  // In this C++ version:
  // - KV cache and cache_len are *internal* to ModelRunner.
  // - On the first call, they will be initialized automatically.
  // - On subsequent calls, they are reused and updated in-place.
  //
  // Inputs:
  //   inputs_embeds: [B, T, D], embedding sequence for current step
  //
  // Return:
  //   logits: [B, T, vocab_size] of this step.
  //
  // Note:
  //   If you只想验证pre-fill一致性，可以只调用一次 LlmStep。
  torch::Tensor LlmStep(const torch::Tensor& inputs_embeds);

  // Reset KV cache and cache_len. After calling this, next LlmStep()
  // will behave like a fresh prefill.
  void ResetCache();

  int64_t num_layers() const { return num_layers_; }
  int64_t num_kv_heads() const { return num_kv_heads_; }
  int64_t head_dim() const { return head_dim_; }

 private:
  // Initialize KV cache and cache_len for a given batch size.
  // Called automatically on first LlmStep() if cache is not initialized.
  void InitializeCacheIfNeeded(const torch::Tensor& inputs_embeds);

  torch::inductor::AOTIModelPackageLoader llm_;
  int64_t num_layers_;
  int64_t num_kv_heads_;
  int64_t head_dim_;

  bool cache_initialized_;
  std::vector<torch::Tensor> key_cache_;
  std::vector<torch::Tensor> value_cache_;
  torch::Tensor cache_len_tensor_;
};

#endif  // MODEL_RUNNER_H_
