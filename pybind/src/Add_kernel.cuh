#pragma once
template <typename T>
void launch_add_kernel(cudaStream_t stream, const int size, T* A, T* B, T* C);

