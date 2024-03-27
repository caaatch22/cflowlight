#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <vector>

#include "cflowlight/common/defines.h"
#include "cflowlight/common/dtype.h"
#include "cflowlight/common/shape.h"
#include "cflowlight/common/stride.h"
#include "cflowlight/ndarray/storage.h"

namespace fl {

class [[nodiscard]] NDArray {
 public:
  NDArray() = default;
  NDArray(Shape const &shape, Dtype dtype);
  NDArray(std::shared_ptr<Storage> const &data, Shape shape, Stride stride,
          Dtype dtype);

  NDArray(NDArray const &other) = default;
  NDArray(NDArray &&other) noexcept = default;
  NDArray &operator=(NDArray const &other) = default;
  NDArray &operator=(NDArray &&other) noexcept = default;
  ~NDArray() = default;

  // TODO: using deducing this to overload over [] when c++23 is available.
  // template <typename Self>
  // auto &operator[](this Self &&self, Shape const &Shape) {}
  Scalar &operator[](Shape const &shape);
  Scalar const &operator[](Shape const &shape) const;

  Shape const &shape() const { return shape_; }
  Stride const &stride() const { return stride_; }
  Dtype dtype() const { return dtype_; }
  size_t ndim() const noexcept { return shape_.ndim(); }
  size_t elements() const noexcept { return shape_.elements(); }
  /**
   * @brief Return the physical bytes in the array.
   * @note Maybe different from elements() * dtype.bytes() if the ndarray is not
   * compact.
   */
  size_t nbytes() const noexcept { return data_->nbytes(); }
  Storage *data() noexcept { return data_.get(); }
  Storage const *data() const noexcept { return data_.get(); }

  /**
   * @brief deep copy the ndarray.
   * @return a compact ndarray.
   */
  NDArray dcopy() const;
  /**
   * @brief may or may not deep copy the ndarray.
   * if dtype is the same with the current dtype, return a shallow copy.
   * else return a deep copy.
   * @note this is not a inplace operation.
   */
  [[nodiscard]] NDArray astype(Dtype const &dtype) const;

  int use_count() const noexcept { return data_.use_count(); }

  bool is_compact() const noexcept;
  /**
   * @brief Return a compact ndarray **may or may not** be a deep copy.
   */
  NDArray compact() const;

  /**************** operators ***************/
  NDArray negate() const;
  NDArray reshape(Shape const &shape) const;
  NDArray broadcast_to(Shape const &shape) const;
  NDArray permute(std::vector<size_t> const &dim) const;
  NDArray sum(std::optional<size_t> dim = std::nullopt,
              bool keepdim = false) const;
  NDArray max(std::optional<size_t> dim = std::nullopt,
              bool keepdim = false) const;
  NDArray matmul(NDArray const &rhs) const;

 private:
  std::shared_ptr<Storage> data_;
  Shape shape_;
  Stride stride_;
  Dtype dtype_;

  // helpers
  /**
   * @brief deep copy the uncompact ndarray.
   * @param dst the destination ndarray which should have the same shape with
   * this
   * @note Users should call dcopy() instead of this function.
   */
  void incompact_dcopy(NDArray &dst) const;

  NDArray incompact_dcopy() const {
    auto ret = NDArray(shape_, dtype_);
    incompact_dcopy(ret);
    return ret;
  }

  std::pair<NDArray, NDArray> reduce_view_out(std::optional<size_t> dim,
                                              bool keepdim = false) const;
};

NDArray operator+(NDArray const &lhs, NDArray const &rhs);
NDArray operator+(NDArray const &lhs, scalar_concept auto rhs);
NDArray operator+(scalar_concept auto lhs, NDArray const &rhs);

NDArray operator-(NDArray const &x);
NDArray operator-(NDArray const &lhs, NDArray const &rhs);
NDArray operator-(NDArray const &lhs, scalar_concept auto rhs);
NDArray operator-(scalar_concept auto lhs, NDArray const &rhs);

NDArray operator*(NDArray const &lhs, NDArray const &rhs);
NDArray operator*(NDArray const &lhs, scalar_concept auto rhs);
NDArray operator*(scalar_concept auto lhs, NDArray const &rhs);

NDArray operator/(NDArray const &lhs, NDArray const &rhs);
NDArray operator/(NDArray const &lhs, scalar_concept auto rhs);
NDArray operator/(scalar_concept auto lhs, NDArray const &rhs);

NDArray operator^(NDArray const &lhs, NDArray const &rhs);
NDArray operator^(NDArray const &lhs, scalar_concept auto rhs);
NDArray operator^(scalar_concept auto lhs, NDArray const &rhs);

NDArray log(NDArray const &x);
NDArray exp(NDArray const &x);

NDArray maximum(NDArray const &lhs, NDArray const &rhs);
NDArray minimum(NDArray const &lsh, NDArray const &rhs);
NDArray reshape(NDArray const &x, Shape const &shape);
NDArray permute(NDArray const &x, std::vector<size_t> const &dim);
NDArray sum(NDArray const &x, std::optional<size_t> dim = std::nullopt,
            bool keepdim = false);
NDArray max(NDArray const &x, std::optional<size_t> dim = std::nullopt,
            bool keepdim = false);
NDArray broadcast_to(NDArray const &x, Shape const &shape);
NDArray matmul(NDArray const &lhs, NDArray const &rhs);

/**********************  templates    *****************************/

namespace detail {
NDArray scalar_i32_op(NDArray const &lhs, scalar_concept auto rhs, auto &&op) {
  auto ret = lhs.dcopy().astype(Dtype::i32());
  std::transform(ret.data()->ibegin(), ret.data()->iend(), ret.data()->ibegin(),
                 [rhs, &op](int32_t const &v) { return op(v, rhs); });
  return ret;
}

NDArray scalar_f32_op(NDArray const &lhs, scalar_concept auto rhs, auto &&op) {
  auto ret = lhs.dcopy().astype(Dtype::f32());
  std::transform(ret.data()->fbegin(), ret.data()->fend(), ret.data()->fbegin(),
                 [rhs, &op](float const &v) { return op(v, rhs); });
  return ret;
}
}  // namespace detail

NDArray operator+(NDArray const &lhs, scalar_concept auto rhs) {
  auto ret_type = Dtype::common_dtype(lhs.dtype(), Dtype::from_scalar(rhs));
  if (ret_type == Dtype::f32()) {
    return detail::scalar_f32_op(lhs, rhs, std::plus<float>{});
  } else {
    return detail::scalar_i32_op(lhs, rhs, std::plus<int32_t>{});
  }
}

NDArray operator+(scalar_concept auto lhs, NDArray const &rhs) {
  return rhs + lhs;
}

NDArray operator-(NDArray const &lhs, scalar_concept auto rhs) {
  return lhs + (-rhs);
}

NDArray operator-(scalar_concept auto lhs, NDArray const &rhs) {
  // directly return lhs + (-rhs) will cause a extra deep copy.
  // 1. the (-rhs) which calls rhs.negate() will cause a deep copy.
  // 2. operator+(NDArray const&, scalar_concept auto) will cause a deep copy.
  // we can reuse the result of (-rhs) to avoid the extra deep copy.
  auto ret_type = Dtype::common_dtype(Dtype::from_scalar(lhs), rhs.dtype());
  auto ret = rhs.negate().astype(ret_type);
  if (ret_type == Dtype::f32()) {
    std::transform(ret.data()->fbegin(), ret.data()->fend(),
                   ret.data()->fbegin(),
                   [lhs](float const &v) { return lhs - v; });
  } else {
    std::transform(ret.data()->ibegin(), ret.data()->iend(),
                   ret.data()->ibegin(),
                   [lhs](int32_t const &v) { return lhs - v; });
  }
  return ret;
}

NDArray operator*(NDArray const &lhs, scalar_concept auto rhs) {
  auto ret_type = Dtype::common_dtype(lhs.dtype(), Dtype::from_scalar(rhs));
  if (ret_type == Dtype::f32()) {
    return detail::scalar_f32_op(lhs, rhs, std::multiplies<float>{});
  } else {
    return detail::scalar_i32_op(lhs, rhs, std::multiplies<int32_t>{});
  }
}

NDArray operator*(scalar_concept auto lhs, NDArray const &rhs) {
  return rhs * lhs;
}

NDArray operator/(NDArray const &lhs, scalar_concept auto rhs) {
  auto ret_type = Dtype::common_dtype(lhs.dtype(), Dtype::from_scalar(rhs));
  if (ret_type == Dtype::f32()) {
    return detail::scalar_f32_op(lhs, rhs, std::divides<float>{});
  } else {
    return detail::scalar_i32_op(lhs, rhs, std::divides<int32_t>{});
  }
}

NDArray operator/(scalar_concept auto lhs, NDArray const &rhs) {
  auto ret_type = Dtype::common_dtype(Dtype::from_scalar(lhs), rhs.dtype());
  auto ret = rhs.dcopy().astype(ret_type);
  if (ret_type == Dtype::f32()) {
    std::transform(ret.data()->fbegin(), ret.data()->fend(),
                   ret.data()->fbegin(),
                   [lhs](float const &v) { return lhs / v; });
  } else {
    std::transform(ret.data()->ibegin(), ret.data()->iend(),
                   ret.data()->ibegin(),
                   [lhs](int32_t const &v) { return lhs / v; });
  }
  return ret;
}

NDArray operator^(NDArray const &lhs, scalar_concept auto rhs) {
  auto ret_type = Dtype::common_dtype(lhs.dtype(), Dtype::from_scalar(rhs));
  if (ret_type == Dtype::f32()) {
    return detail::scalar_f32_op(lhs, rhs, std::pow);
  } else {
    return detail::scalar_i32_op(lhs, rhs, std::pow);
  }
}

NDArray operator^(scalar_concept auto lhs, NDArray const &rhs) {
  auto ret_type = Dtype::common_dtype(Dtype::from_scalar(lhs), rhs.dtype());
  auto ret = rhs.dcopy().astype(ret_type);
  if (ret_type == Dtype::f32()) {
    std::transform(ret.data()->fbegin(), ret.data()->fend(),
                   ret.data()->fbegin(),
                   [lhs](float const &v) { return std::pow(lhs, v); });
  } else {
    std::transform(ret.data()->ibegin(), ret.data()->iend(),
                   ret.data()->ibegin(), [lhs](int32_t const &v) {
                     return static_cast<int32_t>(std::pow(lhs, v));
                   });
  }
  return ret;
}

/************** from iterable **********************/
template <Iterable T>
NDArray from_iterable(T const &container) {
  Dtype dtype = Dtype::none();
  check_shape_dtype(container, dtype);
  Shape shape;
  extract_shape(container, shape);

  NDArray res(shape, dtype);
  auto iter = res.data()->begin();
  extract_elements(container, iter);
  return res;
}

void check_dtype(auto v, Dtype &dtype) {
  if constexpr (!std::is_arithmetic_v<decltype(v)>) {
    throw std::runtime_error("Unsupported scalar type");
  }
  if (dtype == Dtype::none()) {
    if constexpr (std::floating_point<decltype(v)>) {
      dtype = Dtype::f32();
    } else if constexpr (std::integral<decltype(v)>) {
      dtype = Dtype::i32();
    }
  } else {
    if constexpr (std::floating_point<decltype(v)>) {
      if (dtype != Dtype::f32()) {
        throw std::runtime_error("Inconsistent dtype from iterable");
      }
    } else if constexpr (std::integral<decltype(v)>) {
      if (dtype != Dtype::i32()) {
        throw std::runtime_error("Inconsistent dtype from iterable");
      }
    }
  }
}

template <typename T>
int64_t check_shape_dtype(T const &t, Dtype &dtype) {
  if constexpr (Iterable<T>) {
    std::vector<int64_t> layer_dim;
    for (auto const &v : t) {
      layer_dim.push_back(check_shape_dtype(v, dtype));
    }
    if (layer_dim.empty()) {
      throw std::runtime_error("Nested iterable empty");
    }
    if (std::ranges::any_of(layer_dim,
                            [&](int64_t v) { return v != layer_dim[0]; })) {
      throw std::runtime_error("Inconsistent shape from iterable");
    }
    return t.size();
  } else {
    check_dtype(t, dtype);
    return 0;
  }
}

template <typename T>
void extract_shape(T const &t, Shape &shape) {
  if constexpr (Iterable<T>) {
    shape.push_back(t.size());
    extract_shape(*t.begin(), shape);
  }
}

template <typename T>
void extract_elements(T const &t, Scalar *&out) {
  if constexpr (Iterable<T>) {
    for (auto const &v : t) {
      extract_elements(v, out);
    }
  } else {
    // we have checked the dtype to be consistent in iterable_shape_dtype
    if constexpr (std::floating_point<T>) {
      out->fdata = static_cast<float>(t);
      out++;
    } else if constexpr (std::integral<T>) {
      out->idata = static_cast<int32_t>(t);
      out++;
    }
  }
}

// print
std::ostream &operator<<(std::ostream &os, NDArray const &arr);

}  // namespace fl