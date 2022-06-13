#include "kernel.h"

__global__ void kernel() {
  printf("cuda enabled\n");
}

void launch() {
  kernel<<<1, 1>>>();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}
