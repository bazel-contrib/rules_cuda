#if defined(CUDA_ENABLED)
#include "kernel.h"
#endif

#include <cstdio>

void do_something_else() {
  fprintf(stderr, "cuda disabled\n");
}

int main() {
#if defined(CUDA_ENABLED)
  launch();
  return 0;
#elif defined(CUDA_DISABLED)
  do_something_else();
  return -1;
#else
#error either CUDA_ENABLED or CUDA_NOT_ENABLED must be defined
#endif
}
