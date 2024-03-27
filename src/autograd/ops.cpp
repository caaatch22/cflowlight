#include "cflowlight/autograd/ops.h"

namespace fl {

Tensor operator+(Tensor const& lhs, Tensor const& rhs) {
  //   FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = lhs.ndarray() + rhs.ndarray();
  auto grad_fn = [](std::vector<Tensor>& inputs, Tensor const& grad_output) {
    inputs[0].add_grad(Tensor(grad_output.ndarray(), false));
    inputs[1].add_grad(Tensor(grad_output.ndarray(), false));
  };
  return Tensor(result, {lhs, rhs}, grad_fn);
}

Tensor operator-(Tensor const& lhs, Tensor const& rhs) {
  //   FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = lhs.ndarray() - rhs.ndarray();
  auto grad_fn = [](std::vector<Tensor>& inputs, Tensor const& grad_output) {
    inputs[0].add_grad(Tensor(grad_output.ndarray(), false));
    inputs[1].add_grad(Tensor(negate(grad_output).ndarray(), false));
  };
  return Tensor(result, {lhs, rhs}, grad_fn);
}

Tensor operator*(Tensor const& lhs, Tensor const& rhs) {
  //   FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = lhs.ndarray() * rhs.ndarray();
  auto grad_fn = [](std::vector<Tensor>& inputs, Tensor const& grad_output) {
    if (inputs[0].requires_grad()) {
      inputs[0].add_grad(
          Tensor(grad_output.ndarray() * inputs[1].ndarray(), false));
    }
    if (inputs[1].requires_grad()) {
      inputs[1].add_grad(
          Tensor(grad_output.ndarray() * inputs[0].ndarray(), false));
    }
  };
  return Tensor(result, {lhs, rhs}, grad_fn);
}

Tensor operator/(Tensor const& lhs, Tensor const& rhs) {
  //   FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = lhs.ndarray() / rhs.ndarray();
  auto gradFunc = [](std::vector<Tensor>& inputs, Tensor const& grad_output) {
    auto inputs1rec = reciprocal(inputs[1]);
    auto gradInput0 = grad_output * inputs1rec;
    if (inputs[0].requires_grad()) {
      inputs[0].add_grad(Tensor(gradInput0.ndarray(), false));
    }
    if (inputs[1].requires_grad()) {
      inputs[1].add_grad(Tensor(
          (gradInput0 * negate(inputs[0]) * inputs1rec).ndarray(), false));
    }
  };
  return Tensor(result, {lhs, rhs}, gradFunc);
}

}  // namespace fl