#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

namespace fl {

class Shape : public std::vector<int> {
 public:
  using std::vector<int>::vector;
  size_t ndim() const noexcept { return size(); }

  int64_t elements() const noexcept {
    return std::reduce(begin(), end(), 1ll, std::multiplies{});
  }

  /**
   * @brief Check if the shape can be broadcasted to another shape.
   * we can broadcast a shape to another shape in two cases:
   * 1. ndims are equal and for dim i, either dims_[i] == rhs.dims_[i] or
   * dims_[i] == 1
   * 2. ndims of the current shape is less than the ndims of the rhs shape and
   * the tailing of rhs.shape is same as the current shape.
   * example:
   * [3, 1, 5] can be broadcasted to [3, x, 5] for x > 0, according to rule 1.
   * [3, 1, 5] can be broadcasted to [..., 3, 1, 5] according to rule 2.
   * [3, 1, 5] can be broadcasted to [..., 3, x, 5] according to rule 1 and 2.
   */
  bool can_broadcast_to(Shape const &rhs) const noexcept {
    if (ndim() > rhs.ndim()) {
      return false;
    }
    return std::equal(rbegin(), rend(), rhs.rbegin(),
                      [](int a, int b) { return a == b || a == 1; });
  }

 private:
  // void check_or_throw(const size_t dim) const {
  //   if (dim > ndim() - 1) {
  //     throw std::invalid_argument(
  //         fmt::format("Shape index {} out of bounds for shape with {} "
  //                     "dimensions.",
  //                     dim, ndim()));
  //   }
  // }
};

inline std::ostream &operator<<(std::ostream &os, Shape const &shape) {
  os << "Shape(";
  for (size_t i = 0; i < shape.ndim(); ++i) {
    os << shape[i];
    if (i != shape.ndim() - 1) {
      os << ", ";
    }
  }
  os << ")";
  return os;
}

}  // namespace fl
