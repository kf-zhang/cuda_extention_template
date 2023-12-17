#include <torch/extension.h>
#include "Add.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &Add<float>, "Add two tensors");
}