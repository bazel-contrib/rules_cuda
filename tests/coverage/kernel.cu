#include "kernel.cuh"

__global__ void kernel(int* out) { *out = 42; }

void launch_kernel(int* out) { kernel<<<1, 1>>>(out); }
