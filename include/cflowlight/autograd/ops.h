#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cflowlight/autograd/tensor.h"
#include "cflowlight/common/defines.h"

namespace fl {

Tensor operator+(Tensor const& lhs, Tensor const& rhs);
Tensor operator+(scalar_concept auto lhs, Tensor const& rhs);
Tensor operator+(Tensor const& lhs, scalar_concept auto rhs);
Tensor operator*(Tensor const& lhs, Tensor const& rhs);
Tensor operator*(scalar_concept auto lhs, Tensor const& rhs);
Tensor operator*(Tensor const& lhs, scalar_concept auto rhs);
Tensor operator-(Tensor const& lhs, Tensor const& rhs);
Tensor operator-(scalar_concept auto lhs, Tensor const& rhs);
Tensor operator-(Tensor const& lhs, scalar_concept auto rhs);
Tensor operator/(Tensor const& lhs, Tensor const& rhs);
Tensor operator/(scalar_concept auto lhs, Tensor const& rhs);
Tensor operator/(Tensor const& lhs, scalar_concept auto rhs);
Tensor operator>(Tensor const& lhs, Tensor const& rhs);
Tensor operator>(scalar_concept auto lhs, Tensor const& rhs);
Tensor operator>(Tensor const& lhs, scalar_concept auto rhs);
Tensor operator<(Tensor const& lhs, Tensor const& rhs);
Tensor operator<(scalar_concept auto lhs, Tensor const& rhs);
Tensor operator<(Tensor const& lhs, scalar_concept auto rhs);
Tensor operator>=(Tensor const& lhs, Tensor const& rhs);
Tensor operator>=(scalar_concept auto lhs, Tensor const& rhs);
Tensor operator>=(Tensor const& lhs, scalar_concept auto rhs);
Tensor operator<=(Tensor const& lhs, Tensor const& rhs);
Tensor operator<=(scalar_concept auto lhs, Tensor const& rhs);
Tensor operator<=(Tensor const& lhs, scalar_concept auto rhs);
Tensor operator&&(Tensor const& lhs, Tensor const& rhs);
Tensor operator!(Tensor const& input);
Tensor negate(Tensor const& input);
Tensor reciprocal(Tensor const& input);
Tensor exp(Tensor const& input);
Tensor log(Tensor const& input);
Tensor pow(Tensor const& input, double p);
Tensor log1p(Tensor const& input);
Tensor sin(Tensor const& input);
Tensor cos(Tensor const& input);
Tensor sqrt(Tensor const& input);
Tensor tanh(Tensor const& input);
Tensor clamp(Tensor const& input, const double min, const double max);
Tensor sigmoid(Tensor const& input);
Tensor swish(Tensor const& input, double beta);
Tensor erf(Tensor const& input);
Tensor max(Tensor const& lhs, Tensor const& rhs);
Tensor max(Tensor const& lhs, scalar_concept auto rhs);
Tensor max(scalar_concept auto lhs, Tensor const& rhs);
Tensor min(Tensor const& lhs, Tensor const& rhs);
Tensor min(Tensor const& lhs, scalar_concept auto rhs);
Tensor min(scalar_concept auto lhs, Tensor const& rhs);
Tensor transpose(Tensor const& input, const Shape& dims = {});
Tensor tileAs(Tensor const& input, Tensor const& reference);
Tensor tileAs(Tensor const& input, const Shape& rdims);
Tensor sumAs(Tensor const& input, Tensor const& reference);
Tensor sumAs(Tensor const& input, const Shape& rdims);
Tensor concatenate(const std::vector<Tensor>& concatInputs, int dim);
std::vector<Tensor> split(Tensor const& input, long splitSize, int dim);
std::vector<Tensor> split(Tensor const& input,
                          const std::vector<long>& splitSizes, int dim);
Tensor tile(Tensor const& input, const Shape& dims);
Tensor sum(Tensor const& input, const std::vector<int>& axes,
           bool keepDims = false);
Tensor mean(Tensor const& input, const std::vector<int>& axes,
            bool keepDims = false);
Tensor norm(Tensor const& input, const std::vector<int>& axes, double p = 2,
            bool keepDims = false);
Tensor normalize(Tensor const& input, const std::vector<int>& axes,
                 double p = 2, double eps = 1e-12);
Tensor var(Tensor const& input, const std::vector<int>& axes,
           const bool isbiased = false, bool keepDims = false);
Tensor matmul(Tensor const& lhs, Tensor const& rhs);

Tensor matmulTN(Tensor const& lhs, Tensor const& rhs);
Tensor matmulNT(Tensor const& lhs, Tensor const& rhs);
Tensor abs(Tensor const& input);
Tensor flat(Tensor const& input);
Tensor moddims(Tensor const& input, const Shape& dims);
Tensor reorder(Tensor const& input, const Shape& shape);
Tensor linear(Tensor const& input, Tensor const& weight);
Tensor linear(Tensor const& input, Tensor const& weight, Tensor const& bias);
Tensor softmax(Tensor const& input, const int dim);
Tensor logSoftmax(Tensor const& input, const int dim);
Tensor binaryCrossEntropy(Tensor const& inputs, Tensor const& targets);
// Tensor categoricalCrossEntropy(Tensor const& input, Tensor const& targets,
//                                ReduceMode reduction = ReduceMode::MEAN,
//                                int ignoreIndex = -1);
// Tensor weightedCategoricalCrossEntropy(Tensor const& input,
//                                        Tensor const& targets,
//                                        Tensor const& weight, int
//                                        ignoreIndex);

// Tensor gatedlinearunit(Tensor const& input, const int dim);
// std::tuple<Tensor, Tensor, Tensor> rnn(Tensor const& input,
//                                        Tensor const& hiddenState,
//                                        Tensor const& cellState,
//                                        Tensor const& weights, int hiddenSize,
//                                        int numLayers, RnnMode mode,
//                                        bool bidirectional, float dropout);

// Tensor embedding(Tensor const& input, Tensor const& embeddings);
// Tensor batchnorm(Tensor const& input, Tensor const& weight, Tensor const&
// bias,
//                  Tensor& runningMean, Tensor& runningVar,
//                  const std::vector<int>& axes, bool train, double momentum,
//                  double epsilon);
// Tensor padding(Tensor const& input, std::vector<std::pair<int, int>> pad,
//                double val);
// Tensor dropout(Tensor const& input, double p);
// Tensor relu(Tensor const& input);
// Tensor gelu(Tensor const& input);
// Tensor relativePositionalEmbeddingRotate(Tensor const& input);
// Tensor multiheadAttention(Tensor const& query, Tensor const& key,
//                           Tensor const& value, Tensor const& posEmb,
//                           Tensor const& mask, Tensor const& padMask,
//                           const int32_t nHeads, const double pDropout,
//                           const int32_t offset = 0);

/************** templates ops impl ***********************/

}  // namespace fl
