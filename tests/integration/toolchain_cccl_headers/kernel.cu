// `<thrust/...>` has to resolve through the header-only targets. CUDA 13 moved the CCCL
// headers into `include/cccl/`, so an `includes` pointing at `include/` resolves
// `<cccl/thrust/...>` and this stops compiling.
#include <thrust/complex.h>

__global__ void probe() {}
