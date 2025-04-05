# Write a cuda extention for pytorch using TORCH_LIBRARY


## Implement CUDA kernel
```cpp
template <typename scalar_t>
__global__ void elementwise_add_kernel(
    const scalar_t* a, const scalar_t* b, scalar_t* c, int64_t n
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```
## Warp the kernel
```cpp
// wrap the kernel in a function
at::Tensor elementwise_add(const at::Tensor& a, const at::Tensor& b) {

    TORCH_CHECK(a.sizes() == b.sizes(), "Input shapes must match");
    TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda(), "Inputs must be CUDA tensors");

    auto c = at::empty_like(a);

    int64_t n = a.numel();
    const dim3 block(256);
    const dim3 grid((n + block.x - 1) / block.x);

    // dispatch the kernel based on the scalar type
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
```
## Register the extension
```cpp
TORCH_LIBRARY(my_ops, m) {
    m.def("elementwise_add(Tensor a, Tensor b) -> Tensor", elementwise_add);
}
```

## Build the extension
```bash
# build the extension
python setup.py build
```
```bash
# install the extension
# this will install the extension in the current environment
python setup.py install
```
```bash

python setup.py develop
```
## Invoke the function

```python
import torch

library_path = './build/lib.linux-x86_64-cpython-312/custom_ops.cpython-312-x86_64-linux-gnu.so'
torch.ops.load_library(library_path)

x = torch.ones(3).cuda()
y = torch.ones(3).cuda()
z = torch.ops.my_ops.elementwise_add(x, y)
print(z)
```
## References
* [PyTorch Custom Operators](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html#custom-ops-landing-page)
* [torch/library.h](https://github.com/pytorch/pytorch/blob/main/torch/library.h)