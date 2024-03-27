#include <omp.h>

#include <experimental/simd>

#include "cflowlight/ndarray/ndarray.h"

namespace fl {

namespace detail {

static constexpr size_t SIMD_SIZE =
    std::experimental::native_simd<float>::size();
static constexpr size_t BLOCK_SIZE = SIMD_SIZE * 8;

template <scalar_concept T>
struct AlignedMatrix {
  AlignedMatrix(size_t m, size_t n)
      : data(new(std::align_val_t(BLOCK_SIZE)) T[m * n]), row(m), col(n) {}
  ~AlignedMatrix() { delete[] data; }
  T& operator()(size_t i, size_t j) { return data[i * col + j]; }
  T const& operator()(size_t i, size_t j) const { return data[i * col + j]; }
  T* data;
  size_t row;
  size_t col;
};

template <scalar_concept T>
AlignedMatrix<T> make_aligned(T const* src, size_t block_size, size_t row,
                              size_t col) {
  size_t const new_row = ((row + block_size - 1) / block_size) * block_size;
  size_t const new_col = ((col + block_size - 1) / block_size) * block_size;
  AlignedMatrix<T> aligned(new_row, new_col);
  for (size_t i = 0; i < row; i++) {
    for (size_t j = 0; j < col; j++) {
      aligned(i, j) = src[i * col + j];
    }
  }
  return aligned;
}

template <scalar_concept T>
AlignedMatrix<T> make_compact_aligned(NDArray const& x, size_t block_size,
                                      size_t row, size_t col) {
  if (x.is_compact()) {
    if constexpr (std::is_same_v<T, float>) {
      return make_aligned(x.data()->fbegin(), block_size, row, col);
    } else {
      return make_aligned(x.data()->ibegin(), block_size, row, col);
    }
  }
  size_t const new_row = ((row + block_size - 1) / block_size) * block_size;
  size_t const new_col = ((col + block_size - 1) / block_size) * block_size;
  AlignedMatrix<T> aligned(new_row, new_col);
  for (int i = 0; i < (int)row; i++) {
    for (int j = 0; j < (int)col; j++) {
      if constexpr (std::is_same_v<T, float>) {
        aligned(i, j) = x[{i, j}].fdata;
      } else {
        aligned(i, j) = x[{i, j}].idata;
      }
    }
  }
  return aligned;
}

template <scalar_concept T>
void gemm(T* C, AlignedMatrix<T> const& aligned_a,
          AlignedMatrix<T> const& aligned_b, size_t m, size_t n) {
  namespace stdx = std::experimental;
  alignas(BLOCK_SIZE) static thread_local T local_a[BLOCK_SIZE][BLOCK_SIZE];
  alignas(BLOCK_SIZE) static thread_local T local_b[BLOCK_SIZE][BLOCK_SIZE];
  alignas(BLOCK_SIZE) static thread_local T local_c[BLOCK_SIZE][BLOCK_SIZE];
  // Ideally, we should use omp threadprivate instead of thread_local.
  // This seems to be a compiler bug
  // https://stackoverflow.com/questions/40976380/openmp-threadprivate-variable-template
  // #pragma omp threadprivate(local_a, local_b, local_c)
  auto aligned_c = AlignedMatrix<T>(aligned_a.row, aligned_b.col);

  size_t const ai_block_num = aligned_a.row / BLOCK_SIZE;
  size_t const aj_block_num = aligned_b.col / BLOCK_SIZE;
  size_t const bk_block_num = aligned_b.row / BLOCK_SIZE;

#pragma omp parallel for
  for (size_t bi = 0; bi < ai_block_num; bi++) {
    for (size_t bj = 0; bj < aj_block_num; bj++) {
      // Clear localC.
      for (size_t i = 0; i < BLOCK_SIZE; i++) {
        for (size_t j = 0; j < BLOCK_SIZE; j += SIMD_SIZE) {
          static const stdx::native_simd<T> zero = 0;
          zero.copy_to(&local_c[i][j], stdx::element_aligned);
        }
      }
      for (size_t bk = 0; bk < bk_block_num; bk++) {
        // Copy local block.
        for (size_t i = 0; i < BLOCK_SIZE; i++) {
          for (size_t j = 0; j < BLOCK_SIZE; j += SIMD_SIZE) {
            size_t const ax = bi * BLOCK_SIZE + i;
            size_t const ay = bk * BLOCK_SIZE + j;
            size_t const bx = bk * BLOCK_SIZE + i;
            size_t const by = bj * BLOCK_SIZE + j;
            stdx::native_simd<T> a;
            stdx::native_simd<T> b;
            b.copy_from(&aligned_b(bx, by), stdx::element_aligned);
            a.copy_from(&aligned_a(ax, ay), stdx::element_aligned);
            a.copy_to(&local_a[i][j], stdx::element_aligned);
            b.copy_to(&local_b[i][j], stdx::element_aligned);
          }
        }
        // BLOCK_GEMM
        for (size_t i = 0; i < BLOCK_SIZE; i++) {
          for (size_t k = 0; k < BLOCK_SIZE; k++) {
            stdx::native_simd<T> a = local_a[i][k];
            for (size_t j = 0; j < BLOCK_SIZE; j += SIMD_SIZE) {
              stdx::native_simd<T> b;
              stdx::native_simd<T> c;
              b.copy_from(&local_b[k][j], stdx::element_aligned);
              c.copy_from(&local_c[i][j], stdx::element_aligned);
              c += a * b;
              c.copy_to(&local_c[i][j], stdx::element_aligned);
            }
          }
        }
      }
      for (size_t i = 0; i < BLOCK_SIZE; i++) {
        std::array<stdx::native_simd<T>, BLOCK_SIZE / SIMD_SIZE> c;
        for (size_t j = 0; j < BLOCK_SIZE; j += SIMD_SIZE) {
          c[i].copy_from(&local_c[i][j], stdx::element_aligned);
          c[i].copy_to(&aligned_c(bi * BLOCK_SIZE + i, bj * BLOCK_SIZE + j),
                       stdx::element_aligned);
        }
      }
    }
  }
  // copy aligned_c to C
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      C[i * n + j] = aligned_c(i, j);
    }
  }
}

}  // namespace detail

NDArray NDArray::matmul(NDArray const& b) const {
  auto const& a = *this;
  if (a.ndim() != 2 || b.ndim() != 2) {
    throw std::runtime_error("matmul: inputs must be 2D arrays");
  }
  // a.compact_aligned, b.compact_aligned

  if (a.dtype() == b.dtype() && a.dtype() == Dtype::i32()) {
    NDArray c(Shape{a.shape()[0], b.shape()[1]}, Dtype::i32());
    auto aligned_a = detail::make_compact_aligned<int>(
        a, detail::BLOCK_SIZE, a.shape()[0], a.shape()[1]);
    auto aligned_b = detail::make_compact_aligned<int>(
        b, detail::BLOCK_SIZE, b.shape()[0], b.shape()[1]);

    detail::gemm(c.data()->ibegin(), aligned_a, aligned_b, a.shape()[0],
                 b.shape()[1]);
    return c;

  } else if (a.dtype() == b.dtype() && a.dtype() == Dtype::f32()) {
    NDArray c(Shape{a.shape()[0], b.shape()[1]}, Dtype::f32());
    auto aligned_a = detail::make_compact_aligned<float>(
        a, detail::BLOCK_SIZE, a.shape()[0], a.shape()[1]);
    auto aligned_b = detail::make_compact_aligned<float>(
        b, detail::BLOCK_SIZE, b.shape()[0], b.shape()[1]);
    detail::gemm(c.data()->fbegin(), aligned_a, aligned_b, a.shape()[0],
                 b.shape()[1]);
    return c;
  } else {
    throw std::runtime_error("matmul: inputs must have the same dtype");
  }
}

}  // namespace fl
