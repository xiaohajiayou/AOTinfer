import os
import torch
import torch.nn as nn

# ===== 1. 定义模型（改成你自己的） =====
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

device = "cuda"  # 如果要导 CPU 模型改成 "cpu"
model = MyModel().to(device).eval()

# 如果有自己的 checkpoint，在这里加载
# model.load_state_dict(torch.load("your_ckpt.pth", map_location=device))

# ===== 2. 准备 example_inputs + dynamic_shapes =====
batch_dim = torch.export.Dim("batch", min=1, max=64)
# 输入名字 "x" 要和 forward 的参数名一致
example_inputs = (torch.randn(4, 10, device=device),)

exported = torch.export.export(
    model,
    example_inputs,
    dynamic_shapes={"x": {0: batch_dim}},
)

# ===== 3. AOTI 编译并打包 =====
package_path = os.path.join(os.getcwd(), "model.pt2")  # 生成的包目录

out = torch._inductor.aoti_compile_and_package(
    exported,
    package_path=package_path,
)

print("AOTI package:", out)


