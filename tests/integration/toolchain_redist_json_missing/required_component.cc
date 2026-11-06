// Keep this include separate from the optional __has_include probe, which can
// suppress missing-header errors in GCC 9 (https://gcc.gnu.org/PR80753).
#include <nvJitLink.h>

nvJitLinkHandle required_component() { return nullptr; }
