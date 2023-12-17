# Write a cuda extention for pytorch using pybind11

* Write a cuda kernel in `src/Add_kernel.cu`
```CUDA
    template <typename T>
    __global__ void add_kernel(const int size, const T* a, const T* b, T* c) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < size) {
            c[i] = a[i] + b[i];
        }
    }
```
* Write a function to launch the kernel in `src/Add_kernel.cu`
```CUDA
    template <typename T>
    void launch_add_kernel(cudaStream_t stream, const int size, T* A, T* B, T* C) {
        constexpr int threadsPerBlock = 256;
        int numBlocks = div_up(size, threadsPerBlock);
        add_kernel<<<numBlocks, threadsPerBlock,0,stream>>>(size, A, B, C);
    }
```
* Write a function that accepts pytorch tensors and return tensor in `src/Add.cpp`
```C++
    template <typename T>
    at::Tensor Add(const at::Tensor& a, const at::Tensor& b) {
        assert(a.numel() == b.numel());
        a.
        int size = a.numel();
        at::Tensor c = at::empty_like(a);
        
        launch_add_kernel(
            at::cuda::getCurrentCUDAStream(), 
            size,
            a.contiguous().data_ptr<T>(), b.contiguous().data_ptr<T>(), c.contiguous().data_ptr<T>()
            );
        
        return c;
    }
```

* instantiate the template functions in `src/Add_kernel.cu` and `src/Add.cpp` for float.    
    If this is not done, the linker will complain about undefined reference to the functions.
```C++
    template void launch_add_kernel<float>(cudaStream_t stream, const int size, float* A, float* B, float* C);
    template at::Tensor Add<float>(const at::Tensor& a, const at::Tensor& b);
```

* Use pybind11 to wrap the function in `src/bind.cpp`
```C++
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("add", &Add<float>, "Add two tensors");
    }
```
* Write `setup.py` to build the extention
```python
    from setuptools import setup
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    import os

    cxx_compiler_flags = []

    setup(
        name="Add",
        ext_modules=[
            CUDAExtension(
                name="Add._C",
                sources=[
                    "src/Add_kernel.cu",
                    "src/Add.cpp",
                    "src/bind.cpp"
                ],
                extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
            ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
```
* Build the extention
```bash
    python setup.py build
```


