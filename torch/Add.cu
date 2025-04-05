#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>



// CUDA 核函数
template <typename scalar_t>
__global__ void elementwise_add_kernel(
    const scalar_t* a, const scalar_t* b, scalar_t* c, int64_t n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 包装函数（调用 CUDA 核）
at::Tensor elementwise_add(const at::Tensor& a, const at::Tensor& b) {
    // 检查输入合法性
    TORCH_CHECK(a.sizes() == b.sizes(), "Input shapes must match");
    TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda(), "Inputs must be CUDA tensors");

    // 分配输出 Tensor
    auto c = at::empty_like(a);

    // 获取数据指针和元素数量
    int64_t n = a.numel();
    const dim3 block(256);
    const dim3 grid((n + block.x - 1) / block.x);

    // 根据数据类型分发核函数
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "elementwise_add_cuda", [&] {
        elementwise_add_kernel<scalar_t><<<grid, block>>>(
            a.data_ptr<scalar_t>(),
            b.data_ptr<scalar_t>(),
            c.data_ptr<scalar_t>(),
            n
        );
    });

    return c;
}

// 注册算子到 PyTorch
TORCH_LIBRARY(my_ops, m) {
    m.def("elementwise_add(Tensor a, Tensor b) -> Tensor", elementwise_add);
}
