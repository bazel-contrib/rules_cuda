#include "kernel_b.h"
#include "common.h"

__global__ void kernel_b() {
#ifdef FEATURE_1
  printf("kernel b - feature_1 enabled\n");
#else
  printf("kernel b - feature_1 disabled\n");
#endif
}

void launch_b() {
  kernel_b<<<1, 1>>>();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

