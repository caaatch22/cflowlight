#include "cflowlight/autograd/tensor.h"

#include <ranges>
#include <unordered_set>

namespace fl {

Tensor::Tensor(NDArray data, bool requires_grad) {
  data = std::move(data);
  grad_data_->requires_grad = requires_grad;
}

Tensor::Tensor(NDArray data, std::vector<Tensor> inputs, GradFunc grad_fn) {
  data = std::move(data);
  if (std::ranges::any_of(
          inputs, [](Tensor const& input) { return input.requires_grad(); })) {
    grad_data_->requires_grad = true;
    grad_data_->inputs = std::move(inputs);
    grad_data_->grad_fn = std::move(grad_fn);
  }
}

void Tensor::add_grad(Tensor const& child_grad) {
  // TODO: check dtype and shape
  if (grad_data_->requires_grad) {
    if (grad_data_->grad == nullptr) {
      // shared underlying data
      grad_data_->grad = std::make_unique<Tensor>(child_grad);
    } else {
      grad_data_->grad = std::make_unique<Tensor>(
          grad_data_->grad->data_ + child_grad.data_, false);
    }
  }
}

void Tensor::zero_grad() { grad_data_->grad.reset(); }

void Tensor::calc_grad_inputs(bool retain_graph) {
  if (grad_data_->grad_fn) {
    if (!grad_data_->grad) {
      throw std::logic_error("gradient was not propagated to this Tensor");
    }

    grad_data_->grad_fn(grad_data_->inputs, *grad_data_->grad);
  }
  if (!retain_graph) {
    grad_data_->inputs.clear();
  }
}

void Tensor::register_gradhook(GradHook const& hook) {
  grad_data_->grad_hook = hook;
}

void Tensor::apply_gradhook() {
  if (grad_data_->grad_hook) {
    // assert(grad_data_->grad);
    grad_data_->grad_hook(*grad_data_->grad);
  }
}

void Tensor::backward(Tensor const& grad_output, bool retain_graph) {
  add_grad(grad_output);
  auto dag = build_graph();
  for (auto& node : dag | std::views::reverse) {
    node.calc_grad_inputs(retain_graph);
    node.apply_gradhook();
    if (!retain_graph) {
      node = Tensor();
    }
  }
}

void Tensor::backward(bool retain_graph) {
  //   auto all_one = ;
  backward({}, retain_graph);
}

std::vector<Tensor>& Tensor::inputs() const { return grad_data_->inputs; }

Tensor::DAG Tensor::build_graph() const {
  std::unordered_set<GD*> cache;
  DAG dag;
  std::function<void(const Tensor&)> dfs = [&](Tensor const& node) {
    auto id = node.grad_data_.get();
    if (cache.contains(id)) {
      return;
    }
    for (auto const& input : node.inputs()) {
      dfs(input);
    }
    cache.insert(id);
    dag.push_back(node);
  };
  return dag;
}

}  // namespace fl