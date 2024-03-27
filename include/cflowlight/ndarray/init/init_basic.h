#pragma once
#include <random>

#include "cflowlight/common/dtype.h"
#include "cflowlight/ndarray/ndarray.h"
#include "xoshiro/xoshiro.h"

namespace fl {

[[maybe_unused]] static xso::xoshiro_4x32_plus &getGlobalGen() {
  static xso::xoshiro_4x32_plus gen;
  return gen;
}

void manual_seed(unsigned int seed);

NDArray ones(Shape const &shape, Dtype dtype = Dtype::f32());

NDArray zeros(Shape const &shape, Dtype dtype = Dtype::f32());

template <scalar_concept T>
NDArray arange(T start, T end, T step = 1) {
  Shape shape = {static_cast<int>((end - start) / step)};
  NDArray res(shape, Dtype::from_scalar(T{}));
  if (res.dtype() == Dtype::i32()) {
    std::iota(res.data()->ibegin(), res.data()->iend(), start);
  } else {
    std::iota(res.data()->fbegin(), res.data()->fend(), start);
  }
  return res;
}

NDArray randn(Shape const &shape);

NDArray rand(Shape const &shape);

NDArray randint(int low, int high, Shape const &shape);

}  // namespace fl