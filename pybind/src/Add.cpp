#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "Add_kernel.cuh"

template <typename T>
at::Tensor Add(const at::Tensor& a, const at::Tensor& b) {
    assert(a.numel() == b.numel());
    
    int size = a.numel();
    at::Tensor c = at::empty_like(a);
    
    launch_add_kernel(
        at::cuda::getCurrentCUDAStream(), 
        size,
        a.contiguous().data_ptr<T>(), b.contiguous().data_ptr<T>(), c.contiguous().data_ptr<T>()
        );
    
    return c;
}

template at::Tensor Add<float>(const at::Tensor& a, const at::Tensor& b);