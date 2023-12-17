#pragma once
#include <torch/extension.h>

template <typename T>
at::Tensor Add(const at::Tensor& a, const at::Tensor& b);