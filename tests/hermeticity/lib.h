#ifndef GENERIC_LIB_H
#define GENERIC_LIB_H

enum class OptionA {
  A,
};

enum class OptionB {
  B,
};

template <class T> bool process(OptionA format, OptionB method);

#endif // GENERIC_LIB_H
