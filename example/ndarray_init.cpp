#include <iostream>
#include <string_view>
#include <vector>

#include "cflowlight/ndarray/ndarray.hpp"
using namespace std::string_view_literals;

int main() {
  // auto a = fl::ones({10}, fl::Dtype::f32());
  // auto b = fl::ones({2, 3}, fl::Dtype::i32());
  // auto c = fl::zeros({2, 3}, fl::Dtype::f32());
  // auto d = fl::arange(0.f, 10.f, 1.f);
  // auto e = a + d;
  // auto g = a + 32.23f;
  // std::cout << "d.dtype() = " << d.dtype() << '\n';  // "d.dtype() = f32"
  // std::cout << "e.dtype() = " << e.dtype() << '\n';  // "e.dtype() = f32
  // std::cout << a << '\n';
  // std::cout << b << '\n';
  // std::cout << d << '\n';
  // std::cout << g << '\n';
  // auto h = fl::arange(0, 20).reshape({4, 5});
  // std::cout << h << '\n';
  // std::cout << h.shape() << '\n';
  // std::cout << h.stride() << '\n';
  // h = h.permute({0, 1});
  // std::cout << h.shape() << '\n';
  // std::cout << h.stride() << '\n';
  // std::cout << h << '\n';
  // fl::manual_seed(123);
  auto h = fl::randn({1, 5, 2, 3});
  auto a = fl::rand({1});
  a = a + 1;
  std::cout << h << '\n';
  h = h.reshape({5, 6});
  std::cout << a << '\n';
  std::cout << "h, after reshape:\n" << h << '\n';
  h = h.permute({1, 0});
  std::cout << "h, after permute:\n" << h << '\n';
  a = fl::ones({5, 3, 1}, "f32"sv);
  std::cout << a << '\n';

  a = fl::from_iterable(std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  std::cout << a.shape() << '\n';
  std::cout << a << '\n';

  std::vector<std::vector<int>> vec = {{1, 3}, {4, 6}, {7, 9}};
  auto b = fl::from_iterable(vec);
  std::cout << b << '\n';
  std::cout << b.dtype() << '\n';
  std::cout << b.shape() << '\n';

  try {
    auto c = std::vector<int>();
    auto d = fl::from_iterable(c);
  } catch (std::exception const &e) {
    std::cout << e.what() << '\n';
  }

  try {
    // inconsistent shape
    auto c =
        fl::from_iterable(std::vector<std::vector<int>>{{1, 2}, {3, 4}, {1}});
  } catch (std::exception const &e) {
    std::cout << e.what() << '\n';
  }

  try {
    // nested shape
    auto c =
        fl::from_iterable(std::vector<std::vector<int>>{{}, {1, 2}, {3, 4}});
  } catch (std::exception const &e) {
    std::cout << e.what() << '\n';
  }

  return 0;
}