#include <chrono>
#include <iostream>

#include "cflowlight/ndarray/ndarray.hpp"

// static std::mt19937& getGlobalGen() {
//   static std::mt19937 gen(std::random_device{}());
//   return gen;
// }

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

int main() {
  // fl::manual_seed(0);
  // auto a = fl::arange(0, 10).reshape({2, 1, 5});
  // //   auto b = nd::ones({2, 3}, nd::Dtype::i32());
  // std::cout << a << '\n';
  // std::cout << a.is_compact() << '\n';
  // auto b = a.broadcast_to({2, 3, 5});
  // std::cout << b << '\n';
  Timer timer;
  auto a = fl::randn({6400, 12800});
  auto b = fl::randn({12800, 6400});
  std::cout << "init time: " << timer.elapsed().count() << " s\n";
  timer.reset();
  // volatile float x = a[{0, 0}].fdata;
  // std::cout << x << '\n';

  // std::vector<int> v(12800 * 6400);
  // std::vector<int> w(6400 * 12800);
  // timer.reset();
  // static xso::xoshiro_4x32_plus gen;
  // std::generate(v.begin(), v.end(), gen);
  // std::generate(w.begin(), w.end(), gen);
  // std::cout << "raw init time: " << timer.elapsed().count() << " s\n";
  // std::cout << "w[0]" << w[0] << '\n';
  // timer.reset();
  // auto c = a.matmul(b);
  // std::cout << "matmul: " << timer.elapsed().count() << " s\n";

  // for (int i = 0; i < 10; i++) {
  //   for (int j = 0; j < 10; j++) {
  //     std::cout << c[{i, j}].fdata << ' ';
  //   }
  //   std::cout << '\n';
  // }

  return 0;
}