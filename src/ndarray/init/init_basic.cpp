#include "cflowlight/ndarray/init/init_basic.h"

#include "absl/random/gaussian_distribution.h"
#include "cflowlight/ndarray/ndarray.h"

namespace fl {

void manual_seed(unsigned int seed) { getGlobalGen().seed(seed); }

NDArray full(Shape const &shape, Scalar const &value, Dtype dtype) {
  NDArray res(shape, dtype);
  std::fill(res.data()->begin(), res.data()->end(), value);
  return res;
};

NDArray ones(Shape const &shape, Dtype dtype) {
  if (dtype == Dtype::i32()) {
    return full(shape, Scalar{1}, dtype);
  } else {
    return full(shape, Scalar{1.0f}, dtype);
  }
};

NDArray zeros(Shape const &shape, Dtype dtype) {
  if (dtype == Dtype::i32()) {
    return full(shape, Scalar{0}, dtype);
  } else {
    return full(shape, Scalar{0.0f}, dtype);
  }
};

#include <chrono>
class [[nodiscard]] Timer {
 public:
  Timer() noexcept : start(std::chrono::high_resolution_clock::now()) {}
  [[nodiscard]] auto elapsed() const noexcept -> std::chrono::duration<double> {
    const auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - start);
  }
  void reset() noexcept { start = std::chrono::high_resolution_clock::now(); }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

NDArray randn(Shape const &shape) {
  NDArray res(shape, Dtype::f32());
  // static std::normal_distribution<float> dist(0.0f, 1.0f);
  static absl::gaussian_distribution<float> dist(0.0f, 1.0f);
  std::generate(res.data()->fbegin(), res.data()->fend(),
                [] { return dist(getGlobalGen()); });
  return res;
}

NDArray rand(Shape const &shape) {
  NDArray res(shape, Dtype::f32());
  static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::generate(res.data()->fbegin(), res.data()->fend(),
                [] { return dist(getGlobalGen()); });
  return res;
}

NDArray randint(int low, int high, Shape const &shape) {
  NDArray res(shape, Dtype::i32());
  std::generate(res.data()->ibegin(), res.data()->iend(), [low, high] {
    std::uniform_int_distribution<int> dist(low, high);
    return dist(getGlobalGen());
  });
  return res;
}
}  // namespace fl
