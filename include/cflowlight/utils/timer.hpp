#pragma once

#include <chrono>

namespace fl {

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

// template <>
// struct fmt::formatter<tbs::Timer> : formatter<double> {
//   template <typename FormatContext>
//   auto format(const tbs::Timer &timer,
//               FormatContext &ctx) const -> decltype(ctx.out()) {
//     return fmt::formatter<double>::format(timer.elapsed().count(), ctx);
//   }
// };

}  // namespace fl