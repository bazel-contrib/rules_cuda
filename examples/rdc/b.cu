#include "b.cuh"

__device__ int g[N];

__device__ void bar() {
  g[threadIdx.x]++;
}
