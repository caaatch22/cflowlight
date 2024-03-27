#pragma once

#include <cstdint>
#include <memory>

namespace fl {

union Scalar {
  static constexpr size_t BYTES = 4;
  Scalar() = default;
  explicit Scalar(float f) : fdata(f) {}
  explicit Scalar(int32_t i) : idata(i) {}
  float fdata;
  int32_t idata;
};

class Storage {
 public:
  using value_type = Scalar;
  using pointer = Scalar *;
  using reference = Scalar &;
  using const_reference = Scalar const &;
  using iterator = Scalar *;
  using const_iterator = Scalar const *;

  Storage() = default;
  explicit Storage(size_t size)
      : data_(new(std::align_val_t(64)) Scalar[size]), size_(size) {}
  ~Storage() { delete[] data_; }

  Storage(Storage const &other) = delete;
  Storage(Storage &&other) noexcept = delete;
  Storage &operator=(Storage const &other) = delete;
  Storage &operator=(Storage &&other) noexcept = delete;

  iterator begin() noexcept { return data_; }
  const_iterator begin() const noexcept { return data_; }
  iterator end() noexcept { return data_ + size_; }
  const_iterator end() const noexcept { return data_ + size_; }
  // TODO: use deducing this to overload over [] when c++23 is available.
  float *fbegin() noexcept { return reinterpret_cast<float *>(data_); }
  float const *fbegin() const noexcept {
    return reinterpret_cast<float *>(data_);
  }
  float const *fend() const noexcept {
    return reinterpret_cast<float *>(data_ + size_);
  }
  float *fend() noexcept { return reinterpret_cast<float *>(data_ + size_); }

  int32_t *ibegin() noexcept { return reinterpret_cast<int32_t *>(data_); }
  int32_t const *ibegin() const noexcept {
    return reinterpret_cast<int32_t *>(data_);
  }
  int32_t const *iend() const noexcept {
    return reinterpret_cast<int32_t *>(data_ + size_);
  }
  int32_t *iend() noexcept {
    return reinterpret_cast<int32_t *>(data_ + size_);
  }

  Scalar *data() noexcept { return data_; }
  Scalar const *data() const noexcept { return data_; }
  Scalar &operator[](size_t idx) noexcept { return data_[idx]; }
  Scalar const &operator[](size_t idx) const noexcept { return data_[idx]; }
  size_t size() const noexcept { return size_; }
  size_t nbytes() const noexcept { return size_ * Scalar::BYTES; }

 private:
  Scalar *data_{nullptr};
  size_t size_{0};
};

}  // namespace fl