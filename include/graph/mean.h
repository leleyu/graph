//
// Created by leleyu on 2019-01-07.
//

#ifndef TEST_MEAN_H
#define TEST_MEAN_H

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>
#include <graph/dataset.h>

namespace graph {
namespace nn {

struct MeanOptions {
  MeanOptions(int in, int out);

  TORCH_ARG(int, in);

  TORCH_ARG(int, out);
};
/// Mean aggregator for neibours
class MeanImpl : public torch::nn::Cloneable<MeanImpl> {
public:
  MeanImpl(int in, int out): MeanImpl(MeanOptions(in, out)) {}
  explicit MeanImpl(MeanOptions options);

  /// Aggregate and calculate the mean of  features of neibours.
  /// Then, multiplying the mean with `weight` and a `relu` activation
  torch::Tensor forward(const torch::Tensor& nodes,
    const torch::Tensor& features,
    const std::unordered_map<int, int>& node_to_index,
    const graph::dataset::AdjList& adj);

  torch::Tensor aggregate(const torch::Tensor& nodes,
    const torch::Tensor& features,
    const std::unordered_map<int, int>& node_to_index,
    const graph::dataset::AdjList& adj);

  void reset() override;

  /// The learned weight
  torch::Tensor weight;

  MeanOptions options;
};

TORCH_MODULE(Mean);

} // namespace nn
} // namespace graph

#endif //TEST_MEAN_H
