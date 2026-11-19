#include <cuda_runtime.h>

__global__ void read_value(int *input, int *output) {
#ifdef INVALID_ACCESS
    output[0] = input[1];
#else
    output[0] = input[0];
#endif
}

int main() {
    int *input, *output;
    if (cudaMalloc(&input, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&output, sizeof(int)) != cudaSuccess) {
        return 2;
    }
#ifndef UNINITIALIZED
    if (cudaMemset(input, 0, sizeof(int)) != cudaSuccess) {
        return 2;
    }
#endif
    read_value<<<1, 1>>>(input, output);
    cudaError_t status = cudaDeviceSynchronize();
    cudaFree(input);
    cudaFree(output);
    // Let the sanitizer determine failure for the negative fixtures, even when
    // the application itself would have returned success.
#if defined(INVALID_ACCESS) || defined(UNINITIALIZED)
    return 0;
#else
    return status == cudaSuccess ? 0 : 2;
#endif
}
