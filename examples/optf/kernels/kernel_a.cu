#include "kernel_a.h"
#include "common.h"

__global__ void kernel_a() {
#ifdef MODE_A
  printf("kernel a - mode a\n");
#else
  printf("kernel a - mode b\n");
#endif
#ifdef FEATURE_1
  printf("kernel a - feature_1 enabled\n");
#else
  printf("kernel a - feature_1 disabled\n");
#endif
}

void launch_a() {
  kernel_a<<<1, 1>>>();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

