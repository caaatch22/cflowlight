#include "cflowlight/ndarray/ndarray.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <ranges>
#include <set>

namespace fl {
/************** helpers *******************/
namespace detail {
NDArray ewise_f32_op(NDArray const &lhs, NDArray const &rhs, auto &&op) {
  static_assert(
      std::is_same_v<std::invoke_result_t<decltype(op), float, float>, float>,
      "Invalid operator.");

  auto l = lhs.compact();
  auto r = rhs.compact();
  auto ret = NDArray(l.shape(), Dtype::f32());
  std::transform(l.data()->fbegin(), l.data()->fend(), r.data()->fbegin(),
                 ret.data()->fbegin(), op);
  return ret;
}

NDArray ewise_i32_op(NDArray const &lhs, NDArray const &rhs, auto &&op) {
  static_assert(
      std::is_same_v<std::invoke_result_t<decltype(op), int32_t, int32_t>,
                     int32_t>,
      "Invalid operator.");

  auto l = lhs.compact();
  auto r = rhs.compact();
  auto ret = NDArray(l.shape(), Dtype::i32());
  std::transform(l.data()->ibegin(), l.data()->iend(), r.data()->ibegin(),
                 ret.data()->ibegin(), op);
  return ret;
}

}  // namespace detail
// ctors

NDArray::NDArray(Shape const &shape, Dtype dtype)
    : data_(std::make_shared<Storage>(shape.elements())),
      shape_(shape),
      stride_(Stride::from_shape(shape)),
      dtype_(dtype) {}

NDArray::NDArray(std::shared_ptr<Storage> const &data, Shape shape,
                 Stride stride, Dtype dtype)
    : data_(data),
      shape_(std::move(shape)),
      stride_(std::move(stride)),
      dtype_(dtype) {}

Scalar &NDArray::operator[](Shape const &shape) {
  return const_cast<Scalar &>(
      static_cast<NDArray const &>(*this).operator[](shape));
}

Scalar const &NDArray::operator[](Shape const &shape) const {
  if (shape.ndim() != ndim()) {
    throw std::invalid_argument("Invalid number of dimensions.");
  }
  auto all_less = [](Shape const &a, Shape const &b) {
    return std::ranges::equal(a, b, std::less{});
  };
  if (!all_less(shape, shape_) ||
      std::ranges::any_of(shape, [](auto i) { return i < 0; })) {
    throw std::invalid_argument("Invalid shape.");
  }
  size_t const idx = std::inner_product(shape.begin(), shape.end(),
                                        stride_.begin(), size_t{0});
  return data_->operator[](idx);
}

NDArray NDArray::dcopy() const {
  NDArray ret(shape_, dtype_);
  if (is_compact()) {
    std::memcpy(ret.data()->begin(), data()->begin(), nbytes());
  } else {
    incompact_dcopy(ret);
  }
  return ret;
}

void NDArray::incompact_dcopy(NDArray &dst) const {
  Shape cur(this->stride_.ndim(), 0);
  auto increase_cur = [&] {
    for (int64_t idx = cur.ndim() - 1; idx >= 0; idx--) {
      cur[idx] += 1;
      if (cur[idx] < this->shape_[idx]) {
        return;
      }
      cur[idx] = 0;
    }
  };
  for (size_t idx = 0; idx < this->elements(); idx++) {
    dst.data_->operator[](idx) = this->operator[](cur);
    increase_cur();
  }
}

NDArray NDArray::astype(Dtype const &dtype) const {
  if (dtype == dtype_) {
    return NDArray(*this);
  }
  NDArray ret = this->dcopy();
  if (dtype == Dtype::f32()) {
    std::transform(ret.data()->ibegin(), ret.data()->iend(),
                   ret.data()->fbegin(),
                   [](int32_t const &x) { return static_cast<float>(x); });
  } else {
    std::transform(ret.data()->fbegin(), ret.data()->fend(),
                   ret.data()->ibegin(),
                   [](float const &x) { return static_cast<int32_t>(x); });
  }
  return ret;
}

bool NDArray::is_compact() const noexcept {
  return stride_ == Stride::from_shape(shape_) &&
         this->elements() * dtype_.bytes() == this->nbytes();
}

NDArray NDArray::compact() const {
  if (is_compact()) {
    // calls the copy constructor, return a shallow copy.
    return NDArray(*this);
  }
  // return a deep copy of this object.
  return incompact_dcopy();
}

/*************** operators *****************/
NDArray NDArray::negate() const {
  auto ret = this->dcopy();
  if (dtype_ == Dtype::f32()) {
    std::transform(ret.data()->fbegin(), ret.data()->fend(),
                   ret.data()->fbegin(), std::negate<float>{});
  } else {
    std::transform(ret.data()->ibegin(), ret.data()->iend(),
                   ret.data()->ibegin(), std::negate<int32_t>{});
  }
  return ret;
}

NDArray NDArray::reshape(Shape const &shape) const {
  // TODO(lmj): support -1 in shape.
  // not sure the logic of is_compact here.
  if (!is_compact()) {
    NDArray ret = this->incompact_dcopy();
    return ret.reshape(shape);
  }
  if (shape.elements() != shape_.elements()) {
    throw std::invalid_argument(
        "Cannot reshape array to shape with different number of elements.");
  }
  return NDArray(data_, shape, Stride::from_shape(shape), dtype_);
}

NDArray NDArray::broadcast_to(Shape const &shape) const {
  if (!shape_.can_broadcast_to(shape)) {
    // TODO: err meg: Cannont broadcast shape: {} to target shape: {}.
    throw std::invalid_argument("Cannot broadcast shape to target shape.");
  }
  Stride s(shape.ndim());
  for (size_t i = 0; i < shape_.ndim(); i++) {
    if (shape_[i] != shape[i + shape.ndim() - shape_.ndim()]) {
      s[i] = 0;
    } else {
      s[i] = stride_[i];
    }
  }

  return NDArray(data_, shape, s, dtype_);
}

NDArray NDArray::permute(std::vector<size_t> const &dim) const {
  auto pm = std::vector<size_t>(this->ndim());
  std::iota(pm.begin(), pm.end(), 0ull);
  if (!std::ranges::is_permutation(dim, pm)) {
    throw std::invalid_argument("Invalid permutation.");
  }

  Shape new_shape(this->ndim());
  Stride new_stride(this->ndim());
  for (size_t i = 0; i < ndim(); i++) {
    new_shape[i] = shape_[dim[i]];
    new_stride[i] = stride_[dim[i]];
  }
  return NDArray(data_, new_shape, new_stride, dtype_);
}

std::pair<NDArray, NDArray> NDArray::reduce_view_out(std::optional<size_t> dim,
                                                     bool keepdim) const {
  if (dim.has_value() && dim.value() >= ndim()) {
    throw std::invalid_argument("Invalid dimension.");
  }
  // reduce all dimensions.
  if (dim == std::nullopt) {
    Shape view_shape(this->ndim() - 1, 1);
    view_shape.push_back(this->elements());
    Shape out_shape(1, 1 * (keepdim ? this->ndim() : 1));
    NDArray view = this->compact().reshape(view_shape);
    NDArray out = NDArray(out_shape, dtype_);
    return {view, out};
  } else {
    // Reduce a specific dimension
    size_t reduce_dim = dim.value();
    std::vector<size_t> view_permute(this->ndim());
    std::iota(view_permute.begin(), view_permute.end(), 0);
    std::swap(view_permute[reduce_dim], view_permute.back());
    Shape out_shape = this->shape();

    // Adjust shapes for view and out based on keepdim
    if (keepdim) {
      out_shape[reduce_dim] = 1;
    } else {
      out_shape.erase(out_shape.begin() + reduce_dim);
    }

    // Move the reduction dimension to the end for view
    NDArray view = this->permute(view_permute);
    NDArray out = NDArray(out_shape, dtype_);

    return {view, out};
  }
}

// TODO: combine sum and max to one function: reduce
NDArray NDArray::sum(std::optional<size_t> dim, bool keepdim) const {
  auto [view, out] = reduce_view_out(dim, keepdim);
  view = view.compact();
  // out is compact, returned by reduce_view_out
  size_t const reduce_size = view.shape().back();

  if (dtype_ == Dtype::f32()) {
    for (size_t i = 0; i < out.elements(); i++) {
      out.data_->fbegin()[i] =
          std::accumulate(view.data_->fbegin() + i * reduce_size,
                          view.data_->fbegin() + (i + 1) * reduce_size, 0.0f);
    }
  } else {
    for (size_t i = 0; i < out.elements(); i++) {
      out.data_->ibegin()[i] =
          std::accumulate(view.data_->ibegin() + i * reduce_size,
                          view.data_->ibegin() + (i + 1) * reduce_size, 0);
    }
  }
  return out;
}

NDArray NDArray::max(std::optional<size_t> dim, bool keepdim) const {
  auto [view, out] = reduce_view_out(dim, keepdim);
  view = view.compact();
  // out is compact, returned by reduce_view_out
  size_t const reduce_size = view.shape().back();
  if (dtype_ == Dtype::f32()) {
    for (size_t i = 0; i < out.elements(); i++) {
      out.data_->fbegin()[i] =
          *std::max_element(view.data_->fbegin() + i * reduce_size,
                            view.data_->fbegin() + (i + 1) * reduce_size);
    }
  } else {
    for (size_t i = 0; i < out.elements(); i++) {
      out.data_->ibegin()[i] =
          *std::max_element(view.data_->ibegin() + i * reduce_size,
                            view.data_->ibegin() + (i + 1) * reduce_size);
    }
  }
  return out;
}

/************** binary operators *******************/
NDArray operator+(NDArray const &lhs, NDArray const &rhs) {
  if (lhs.shape() != rhs.shape()) {
    throw std::invalid_argument("Cannot add arrays with different shapes.");
  }
  auto const ret_type = Dtype::common_dtype(lhs.dtype(), rhs.dtype());
  if (auto d = lhs.dtype(); d == rhs.dtype() && d == Dtype::i32()) {
    return detail::ewise_i32_op(lhs, rhs, std::plus<int32_t>{});
  } else if (d == rhs.dtype() && d == Dtype::f32()) {
    return detail::ewise_f32_op(lhs, rhs, std::plus<float>{});
  } else {
    // lhs.dtype != rhs.dtype, need astype first
    // TODO: can reduce a dcopy here
    return detail::ewise_f32_op(lhs.astype(ret_type), rhs.astype(ret_type),
                                std::plus<float>{});
  }
}

NDArray operator-(NDArray const &x) { return x.negate(); }

NDArray operator-(NDArray const &lhs, NDArray const &rhs) {
  return lhs + (-rhs);
}

NDArray operator*(NDArray const &lhs, NDArray const &rhs) {
  if (lhs.shape() != rhs.shape()) {
    throw std::invalid_argument(
        "Cannot multiply arrays with different shapes.");
  }
  auto const ret_type = Dtype::common_dtype(lhs.dtype(), rhs.dtype());
  if (auto d = lhs.dtype(); d == rhs.dtype() && d == Dtype::i32()) {
    return detail::ewise_i32_op(lhs, rhs, std::multiplies<int32_t>{});
  } else if (d == rhs.dtype() && d == Dtype::f32()) {
    return detail::ewise_f32_op(lhs, rhs, std::multiplies<float>{});
  } else {
    return detail::ewise_f32_op(lhs.astype(ret_type), rhs.astype(ret_type),
                                std::multiplies<float>{});
  }
}

NDArray operator/(NDArray const &lhs, NDArray const &rhs) {
  if (lhs.shape() != rhs.shape()) {
    throw std::invalid_argument("Cannot divide arrays with different shapes.");
  }
  auto const ret_type = Dtype::common_dtype(lhs.dtype(), rhs.dtype());
  if (auto d = lhs.dtype(); d == rhs.dtype() && d == Dtype::i32()) {
    return detail::ewise_i32_op(lhs, rhs, std::divides<int32_t>{});
  } else if (d == rhs.dtype() && d == Dtype::f32()) {
    return detail::ewise_f32_op(lhs, rhs, std::divides<float>{});
  } else {
    return detail::ewise_f32_op(lhs.astype(ret_type), rhs.astype(ret_type),
                                std::divides<float>{});
  }
}

NDArray operator^(NDArray const &lhs, NDArray const &rhs) {
  if (lhs.shape() != rhs.shape()) {
    throw std::invalid_argument("Cannot power arrays with different shapes.");
  }
  auto const ret_type = Dtype::common_dtype(lhs.dtype(), rhs.dtype());
  if (auto d = lhs.dtype(); d == rhs.dtype() && d == Dtype::i32()) {
    return detail::ewise_i32_op(lhs, rhs, [](int32_t a, int32_t b) {
      return static_cast<int32_t>(std::pow(a, b));
    });
  } else if (d == rhs.dtype() && d == Dtype::f32()) {
    return detail::ewise_f32_op(
        lhs, rhs, [](float a, float b) { return std::pow(a, b); });
  } else {
    return detail::ewise_f32_op(
        lhs.astype(ret_type), rhs.astype(ret_type),
        [](float a, float b) { return std::pow(a, b); });
  }
}

NDArray log(NDArray const &x) {
  auto ret = x.dcopy().astype(Dtype::f32());
  std::transform(ret.data()->fbegin(), ret.data()->fend(), ret.data()->fbegin(),
                 [](float const &v) { return std::log(v); });
  return ret;
}

NDArray exp(NDArray const &x) {
  auto ret = x.dcopy().astype(Dtype::f32());
  std::transform(ret.data()->fbegin(), ret.data()->fend(), ret.data()->fbegin(),
                 [](float const &v) { return std::exp(v); });
  return ret;
}

NDArray maximum(NDArray const &lhs, NDArray const &rhs) {
  if (lhs.shape() != rhs.shape()) {
    throw std::invalid_argument("Cannot compare arrays with different shapes.");
  }
  auto const ret_type = Dtype::common_dtype(lhs.dtype(), rhs.dtype());
  if (auto d = lhs.dtype(); d == rhs.dtype() && d == Dtype::i32()) {
    return detail::ewise_i32_op(
        lhs, rhs, [](int32_t a, int32_t b) { return std::max(a, b); });
  } else if (d == rhs.dtype() && d == Dtype::f32()) {
    return detail::ewise_f32_op(
        lhs, rhs, [](float a, float b) { return std::max(a, b); });
  } else {
    return detail::ewise_f32_op(
        lhs.astype(ret_type), rhs.astype(ret_type),
        [](float a, float b) { return std::max(a, b); });
  }
}

NDArray minimum(NDArray const &lhs, NDArray const &rhs) {
  if (lhs.shape() != rhs.shape()) {
    throw std::invalid_argument("Cannot compare arrays with different shapes.");
  }
  auto const ret_type = Dtype::common_dtype(lhs.dtype(), rhs.dtype());
  if (auto d = lhs.dtype(); d == rhs.dtype() && d == Dtype::i32()) {
    return detail::ewise_i32_op(
        lhs, rhs, [](int32_t a, int32_t b) { return std::min(a, b); });
  } else if (d == rhs.dtype() && d == Dtype::f32()) {
    return detail::ewise_f32_op(
        lhs, rhs, [](float a, float b) { return std::min(a, b); });
  } else {
    return detail::ewise_f32_op(
        lhs.astype(ret_type), rhs.astype(ret_type),
        [](float a, float b) { return std::min(a, b); });
  }
}

NDArray reshape(NDArray const &arr, Shape const &shape) {
  return arr.reshape(shape);
}

NDArray permute(NDArray const &arr, std::vector<size_t> const &dim) {
  return arr.permute(dim);
}

NDArray broadcast_to(NDArray const &arr, Shape const &shape) {
  return arr.broadcast_to(shape);
}

NDArray sum(NDArray const &arr, std::optional<size_t> dim, bool keepdim) {
  return arr.sum(dim, keepdim);
}

NDArray max(NDArray const &arr, std::optional<size_t> dim, bool keepdim) {
  return arr.max(dim, keepdim);
}

NDArray matmul(NDArray const &lhs, NDArray const &rhs) {
  return lhs.matmul(rhs);
}

/*********** print, need to be improved *********************/

void increase_cur(NDArray const &arr, Shape &idx) {
  for (int i = arr.ndim() - 1; i >= 0; --i) {
    if (idx[i] == arr.shape()[i] - 1) {
      idx[i] = 0;
    } else {
      idx[i] += 1;
      break;
    }
  }
}

std::string print_1d(NDArray const &arr, Shape &idx) {
  std::string s = "[";
  for (int i = 0; i < arr.shape().back(); ++i) {
    if (arr.dtype() == Dtype::f32()) {
      s += std::to_string(arr[idx].fdata);
    } else {
      s += std::to_string(arr[idx].idata);
    }
    increase_cur(arr, idx);
    if (i != arr.shape().back() - 1) {
      s += ", ";
    }
  }
  s += "]";
  return s;
}

std::ostream &operator<<(std::ostream &os, NDArray const &arr) {
  Shape idx(arr.ndim(), 0);
  auto shape = arr.shape();

  int oned_cnt = std::accumulate(shape.begin(), shape.end() - 1, 1,
                                 std::multiplies<int>());
  std::vector<std::string> lines;
  for (int i = 0; i < oned_cnt; ++i) {
    lines.push_back(print_1d(arr, idx));
  }

  if (arr.ndim() == 1) {
    os << lines[0];
    return os;
  }
  for (int t = 1, i = shape.ndim() - 2; i >= 0; i--) {
    t *= shape[i];
    for (size_t j = 0; j < lines.size(); j += t) {
      lines[j] = "[" + lines[j];
      lines[j + t - 1] = lines[j + t - 1] + "]";
    }
  }
  for (int leading_pad = (lines[0].find_first_not_of('['));
       auto &line : lines) {
    int cur_pad = line.find_first_not_of('[');
    line = std::string(leading_pad - cur_pad, ' ') + line;
    os << line << '\n';
  }
  return os;
}
}  // namespace fl