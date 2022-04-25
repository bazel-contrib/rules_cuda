#include "b.cuh"
#include <stdio.h>

#define CUDA_CHECK(expr)                                                \
  do {                                                                  \
    cudaError_t err = (expr);                                           \
    if (err != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA Error Code  : %d\n     Error String: %s\n", \
              err, cudaGetErrorString(err));                            \
      exit(err);                                                        \
    }                                                                   \
  } while (0)

__global__ void foo() {
  __shared__ int a[N];
  a[threadIdx.x] = threadIdx.x;
  __syncthreads();

  g[threadIdx.x] = a[blockDim.x - threadIdx.x - 1];
  bar();
}

int main(void) {
  unsigned int i;
  int *dg, hg[N];
  int sum = 0;

  foo<<<1, N>>>();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaGetSymbolAddress((void**)&dg, g));
  CUDA_CHECK(cudaMemcpy(hg, dg, N * sizeof(int), cudaMemcpyDeviceToHost));

  for (i = 0; i < N; i++) {
    sum += hg[i];
  }
  if (sum == 36) {
    printf("PASSED\n");
  } else {
    printf("FAILED (%d)\n", sum);
  }

  return 0;
}
