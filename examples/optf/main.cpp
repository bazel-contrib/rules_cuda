#include "kernels/kernel_a.h"
#include "kernels/kernel_b.h"

#include <cstdio>

int main() {
  launch_a();
  launch_b();
  return 0;
}

