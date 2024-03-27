#include <iostream>

#include "cflowlight/ndarray/ndarray.hpp"

int main() {
  auto a = fl::ndarray({2, 3}, fl::Dtype::f32());

  auto b = fl::ndarray({2, 3}, fl::Dtype::i32());
  auto c = a + b;
  std::cout << c.dtype() << '\n';
  auto d = a + 1;
  std::cout << d.dtype() << '\n';
  auto e = a + 1.0f;
  std::cout << e.dtype() << '\n';
  auto f = b + 1.0f;
  std::cout << f.dtype() << '\n';
  auto g = b + 1;
  std::cout << g.dtype() << '\n';

  auto x = -a;
  auto y = -b;
  auto z = x - y;
  std::cout << z.dtype() << '\n';

  a = fl::ones({2, 3}, fl::Dtype::i32());
  b = fl::ones({2, 3}, fl::Dtype::f32());
  float* ptr1 = a.data()->fbegin();
  std::cout << *ptr1 << '\n';
  float* ptr2 = b.data()->fbegin();
  std::cout << *ptr2 << '\n';

  c = a + b;

  std::cout << c.dtype() << '\n';

  float* ptr = c.data()->fbegin();

  std::cout << *ptr << '\n';

  return 0;
}