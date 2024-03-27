#pragma once

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "cflowlight/common/defines.h"

namespace fl {

class Dtype {
 public:
  static Dtype f32() { return Dtype(type::f32); }
  static Dtype i32() { return Dtype(type::i32); }

  static Dtype common_dtype(Dtype const &lhs, Dtype const &rhs) {
    if (lhs == rhs) {
      return lhs;
    } else {
      return Dtype::f32();
    }
  }
  // iternel use
  static Dtype none() { return Dtype(type::none); }

  static Dtype from_scalar(scalar_concept auto scalar) {
    if constexpr (std::is_same_v<decltype(scalar), int32_t>) {
      return Dtype::i32();
    } else if constexpr (std::is_same_v<decltype(scalar), float>) {
      return Dtype::f32();
    }
  }

  Dtype() : dtype_(type::none) {}
  Dtype(Dtype const &) noexcept = default;
  Dtype(Dtype &&) noexcept = default;
  Dtype &operator=(Dtype const &) noexcept = default;
  Dtype &operator=(Dtype &&) noexcept = default;
  Dtype(std::string_view dtype) {
    if (dtype == "f32") {
      dtype_ = type::f32;
    } else if (dtype == "i32") {
      dtype_ = type::i32;
    } else {
      // TODO: use fmt::format
      throw std::invalid_argument("Invalid dtype: " + std::string(dtype));
    }
  }
  bool operator==(Dtype const &rhs) const noexcept {
    return dtype_ == rhs.dtype_;
  }
  bool operator!=(Dtype const &rhs) const noexcept { return !(*this == rhs); }
  size_t bytes() const noexcept { return sizeof(float); }
  friend std::ostream &operator<<(std::ostream &os, Dtype const &dt);

 private:
  enum class type : uint8_t {
    f32,
    i32,
    none,
  };
  Dtype(type dtype) noexcept : dtype_(dtype) {}
  type dtype_;
};

inline std::ostream &operator<<(std::ostream &os, Dtype const &dt) {
  if (dt == Dtype::type::f32) {
    os << "nd.f32";
  } else if (dt == Dtype::type::i32) {
    os << "nd.i32";
  } else {
    os << "nd.none";
  }
  return os;
}

}  // namespace fl