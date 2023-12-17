#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>


int div_up(int a, int b) {
    return (a + b - 1) / b;
}

template <typename T>
__global__ void add_kernel(const int size, const T* a, const T* b, T* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        c[i] = a[i] + b[i];
    }
}


template <typename T>
void launch_add_kernel(cudaStream_t stream, const int size, T* A, T* B, T* C) {
    constexpr int threadsPerBlock = 256;
    int numBlocks = div_up(size, threadsPerBlock);
    add_kernel<<<numBlocks, threadsPerBlock,0,stream>>>(size, A, B, C);
}

// instantiate the template
template void launch_add_kernel<float>(cudaStream_t stream, const int size, float* A, float* B, float* C);



