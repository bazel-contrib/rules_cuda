#include <cstdio>

#if defined(REQUIRE_NVJITLINK) || __has_include(<nvJitLink.h>)
#include <nvJitLink.h>
int main() {
  // This fixture deliberately makes nvJitLink unavailable.
  return 1;
}
#else
int main() {
  std::puts("nvJitLink is unavailable; optional fallback works");
  return 0;
}
#endif
