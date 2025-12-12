#include <torch/torch.h>
#include <iostream>

int main() {
    std::cout << "LibTorch version: " << TORCH_VERSION_MAJOR << "." << TORCH_VERSION_MINOR << std::endl;

    if (!torch::cuda::is_available()) {
        std::cerr << "CUDA is not available. Please check your environment." << std::endl;
        return 1;
    } else {
        std::cout<<"CUDA is ready! Hello World!"<<std::endl;
    }

    torch::Device device(torch::kCUDA, 0);
    std::cout << "CUDA device count: " << static_cast<int>(torch::cuda::device_count()) << std::endl;
    std::cout << "Using device: " << static_cast<int>(device.index()) << std::endl;

    auto a = torch::rand({512, 512}, torch::device(device).dtype(torch::kFloat32));
    auto b = torch::rand({512, 512}, torch::device(device).dtype(torch::kFloat32));
    auto c = torch::mm(a, b);

    std::cout << "Result tensor shape: " << c.sizes() << std::endl;
    std::cout << "Mean value: " << c.mean().item<float>() << std::endl;

    return 0;
}
