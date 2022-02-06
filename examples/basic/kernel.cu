#include "kernel.h"

#include <iostream>

__global__ void Kernel() { printf("cuda kernel called!\n"); }

void ReportIfError(cudaError_t error) {
  if (error != cudaSuccess)
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
}

void Launch() {
  Kernel<<<1, 1>>>();
  ReportIfError(cudaGetLastError());
  ReportIfError(cudaDeviceSynchronize());
}
