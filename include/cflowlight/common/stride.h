#pragma once

#include <algorithm>
#include <iostream>
#include <vector>

#include "cflowlight/common/shape.h"

namespace fl {

class Stride : public std::vector<int> {
 public:
  static Stride from_shape(Shape const& shape) {
    Stride stride(shape.ndim());
    std::exclusive_scan(shape.rbegin(), shape.rend(), stride.rbegin(), 1,
                        std::multiplies{});
    return stride;
  }

  using std::vector<int>::vector;
  size_t ndim() const noexcept { return size(); }

 private:
};

inline std::ostream& operator<<(std::ostream& os, Stride const& stride) {
  os << "Stride(";
  for (size_t i = 0; i < stride.ndim(); ++i) {
    os << stride[i];
    if (i != stride.ndim() - 1) {
      os << ", ";
    }
  }
  os << ")";
  return os;
}

}  // namespace fl