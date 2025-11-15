#include "lib.h"
#include <iostream>

namespace {
int helper_function(OptionB method) {
  if (method == OptionB::B) {
    return 1;
  }

  return 1;
}
} // namespace

template <class T> bool process(OptionA format, OptionB resizeMethod) {
  if (format == OptionA::A) {
    // The implementation doesn't matter, just the call to the function
    // in the anonymous namespace.
    helper_function(resizeMethod);
    return true;
  }

  return false;
}

template bool process<unsigned char>(OptionA format,
                                    OptionB resizeMethod);
