#pragma once

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace fl {

template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

template <typename T>
concept i32_concept = std::is_same<T, int32_t>::value;

template <typename T>
concept f32_concept = std::is_same<T, float>::value;

template <typename T>
concept scalar_concept = i32_concept<T> || f32_concept<T>;

template <typename T>
concept Iterable = requires(T t) {
  { t.begin() } -> std::same_as<decltype(t.begin())>;
  { t.end() } -> std::same_as<decltype(t.end())>;
  { t.size() } -> std::convertible_to<size_t>;
};

}  // namespace fl
