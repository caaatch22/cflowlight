#pragma once
#include "cflowlight/ndarray/ndarray.h"

namespace fl {

class Tensor {
 public:
  using GradFunc = std::function<void(std::vector<Tensor>& inputs,
                                      Tensor const& grad_output)>;

  using GradHook = std::function<void(Tensor& grad)>;

  Tensor() : grad_data_(std::make_shared<GD>()) {}
  Tensor(NDArray data, bool requires_grad = false);
  Tensor(NDArray data, std::vector<Tensor> inputs, GradFunc grad_fn);

  NDArray const& ndarray() const { return data_; }
  NDArray& ndarray() { return data_; }
  // Accessors
  bool requires_grad() const noexcept { return grad_data_->requires_grad; }
  Shape shape() const { return data_.shape(); }
  Dtype dtype() const { return data_.dtype(); }
  int ndim() const { return data_.ndim(); }
  size_t nbytes() const { return data_.nbytes(); }

  // autograd functions
  void zero_grad();
  void add_grad(Tensor const& child_grad);
  void register_gradhook(GradHook const& hook);
  void backward(Tensor const& grad_output, bool retain_graph = false);
  void backward(bool retain_graph = false);

 private:
  using DAG = std::vector<Tensor>;

  void calc_grad_inputs(bool retain_graph = false);

  void apply_gradhook();

  std::vector<Tensor>& inputs() const;

  DAG build_graph() const;
  // Gradient Data
  struct GD {
    bool requires_grad{false};
    std::vector<Tensor> inputs{};
    std::unique_ptr<Tensor> grad{nullptr};
    GradFunc grad_fn{nullptr};
    GradHook grad_hook{nullptr};
  };
  std::shared_ptr<GD> grad_data_;
  NDArray data_;
};
}  // namespace fl