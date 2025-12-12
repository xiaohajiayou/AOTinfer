
#include <iostream>
#include <string>

#include <torch/torch.h>
#include "model_runner.h"
#include <torch/serialize/input-archive.h>

#include <fstream>
#include <vector>
// #include <nlohmann/json.hpp>
void Profile(const std::string& tag,
  std::chrono::steady_clock::time_point start_tp) {
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
  std::chrono::steady_clock::now() - start_tp)
  .count();
  uint64_t length = tag.size();
  uint64_t padding = (40 - length)/2;
  std::string padding_str(padding, '*');
  std::cout << padding_str << tag << " took " << ms << " ms" << padding_str << std::endl;
};

torch::Tensor LoadBinaryTensor(const std::string& bin_path,
                               const std::string& meta_path,
                               std::vector<int64_t>& shape,
                               c10::Device device = torch::kCUDA,
                               torch::Dtype dtype = torch::kBFloat16) {

  auto t0 = std::chrono::steady_clock::now();
  // 1. 读 bin（float32）
  std::ifstream bin_stream(bin_path, std::ios::binary);
  if (!bin_stream) {
    throw std::runtime_error("failed to open bin file: " + bin_path);
  }
  bin_stream.seekg(0, std::ios::end);
  size_t num_bytes = bin_stream.tellg();
  bin_stream.seekg(0, std::ios::beg);
  size_t num_elems = num_bytes / sizeof(float);

  std::vector<float> buffer(num_elems);
  bin_stream.read(reinterpret_cast<char*>(buffer.data()), num_bytes);
  Profile("Read bin file", t0);
  auto t1 = std::chrono::steady_clock::now();
  // 2. from_blob -> clone() -> 搬到 GPU
  auto tensor = torch::from_blob(buffer.data(), shape).clone().to(device, dtype);

  Profile("from_blob and clone", t1);

  std::cout << "Loaded tensor from " << bin_path << "|| shape: " << tensor.sizes() << " || dtype: " << tensor.dtype() << std::endl;
  return tensor;
}

double MaxAbsDiff(const torch::Tensor& a, const torch::Tensor& b) {
  torch::Tensor diff = (a - b).abs();
  return diff.max().item<double>();
}

int main(int argc, char** argv) {

  // Usage:
  //   ./llm_verify <llm_pt2_path> <inputs_embeds.pt> [ref_logits.pt]
  // if (argc < 3) {
  //   std::cerr << "Usage: " << argv[0]
  //             << " <llm_pt2_path> <inputs_embeds.pt> [ref_logits.pt]"
  //             << std::endl;
  //   return 1;
  // }

  const std::string llm_path = argv[1];
  const std::string inputs_path = argv[2];
  const bool has_ref_logits = (argc >= 4);
  const std::string ref_logits_path = has_ref_logits ? argv[3] : "";

  std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;

  const int kDeviceIndex = 0;
  if (!torch::cuda::is_available()) {
    std::cerr << "CUDA is not available." << std::endl;
    return 1;
  }

  std::cout << "CUDA device count: "
            << static_cast<long long>(torch::cuda::device_count())
            << std::endl;
  std::cout << "Using device: cuda:" << kDeviceIndex << std::endl;

  // torch::Device device(torch::kCUDA, kDeviceIndex);
  // const std::string inputs_path = "/home/liwenxiao/AOTinfer/artifacts/minicpm/minicpm_prefill_embed.pt";
  // torch::Tensor inputs_embeds;
  // torch::pickle_load(inputs_embeds, inputs_path);
  // torch::load(inputs_embeds, "/home/liwenxiao/AOTinfer/artifacts/minicpm/minicpm_prefill_embed.pt");
  // torch::Tensor inputs_embeds;
  // torch::serialize::InputArchive archive;
  // archive.load_from("/home/liwenxiao/AOTinfer/artifacts/minicpm/minicpm_prefill_embed.pt");
  // archive.read("tensor", inputs_embeds);            // 键名与 Python 保存时一致

  // torch::serialize::InputArchive archive;
  // archive.load_from(inputs_path, torch::Device(torch::kCPU));
  // torch::Tensor t = torch::empty({1,105,3584}, torch::kFloat32);
  // archive.read("tensor", t);

  std::vector<int64_t> input_embed_shape = {1, 105, 3584};
  auto inputs_embeds =
  LoadBinaryTensor("/home/liwenxiao/AOTinfer/artifacts/minicpm/prefill_embed.bin",
                      "/home/liwenxiao/AOTinfer/artifacts/minicpm/prefill_embed.json",
                      input_embed_shape,
                      torch::Device(torch::kCUDA, 0),
                      torch::kBFloat16);  


  // // TODO：把 28 / 4 / 128 替换成你 meta 里的真实 num_layers / num_kv_heads / head_dim。
  const int64_t kNumLayers = 28;
  const int64_t kNumKvHeads = 4;
  const int64_t kHeadDim = 128;
  auto t_runner = std::chrono::steady_clock::now();
  ModelRunner runner(llm_path, kDeviceIndex,
                     kNumLayers, kNumKvHeads, kHeadDim);
  Profile("Create ModelRunner", t_runner);


  // // 只跑一次 LlmStep，对应 Python 的 prefill。
  auto t_llm = std::chrono::steady_clock::now();
  torch::Tensor logits = runner.LlmStep(inputs_embeds);
  Profile("LlmStep", t_llm);
  std::cout << "C++ logits shape: " << logits.sizes() << std::endl;

// 对比 logits 与 ref_logits
  std::vector<int64_t> output_logits_shape = {1, 105, 151700};
  auto output_logits =
  LoadBinaryTensor("/home/liwenxiao/AOTinfer/artifacts/minicpm/prefill_logits.bin",
                      "/home/liwenxiao/AOTinfer/artifacts/minicpm/prefill_logits.json",
                      output_logits_shape,
                      torch::Device(torch::kCUDA, 0),
                      torch::kBFloat16);



  if (!logits.sizes().equals(output_logits.sizes())) {
    std::cerr << "Shape mismatch between logits and output_logits."
              << std::endl;
  } else {
    const double max_diff = MaxAbsDiff(logits, output_logits);
    std::cout << "Max abs diff between C++ logits and output_logits: "
              << max_diff << std::endl;
  }
  auto probs = logits.index({0, logits.size(1) - 1}).softmax(-1);
  auto result = probs.max(-1);
  auto max_value = std::get<0>(result);   // Tensor
  auto max_index = std::get<1>(result);   // Tensor
  std::cout << "Greedy token id: " << max_index.item<int64_t>()
            << ", prob = " << max_value.item<float>() << std::endl;
  



  return 0;
}
