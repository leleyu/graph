//
// Created by leleyu on 2019-01-07.
//

#ifndef GRAPH_LAYER_H
#define GRAPH_LAYER_H

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <graph/dataset.h>
#include <graph/sampler.h>

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
  MeanImpl(int in, int out) : MeanImpl(MeanOptions(in, out)) {}

  explicit MeanImpl(MeanOptions options);

  /// Aggregate and calculate the mean of  features of neibours.
  /// Then, multiplying the mean with `weight` and a `relu` activation
  torch::Tensor forward(const torch::Tensor &nodes,
                        const torch::Tensor &features,
                        const std::unordered_map<int, int> &node_to_index,
                        const dataset::AdjList &adj);

  /// Aggregate the features of neibours without sampling
  torch::Tensor aggregate(const torch::Tensor &nodes,
                          const torch::Tensor &features,
                          const std::unordered_map<int, int> &node_to_index,
                          const dataset::AdjList &adj);

  void reset() override;

  /// The learned weight with dim [in, out]
  torch::Tensor weight;

  MeanOptions options;
};

class Mean0Impl : public torch::nn::Cloneable<Mean0Impl> {
 public:
  Mean0Impl(int in, int out) : Mean0Impl(MeanOptions(in, out)) {}

  explicit Mean0Impl(MeanOptions options);


  torch::Tensor Forward(const NodeArray &nodes,
                        const NodeArray &neighbors,
                        const SparseNodeEmbedding &embedding,
                        const size_t num_sample);

  void reset() override;

  /// The learned weight with dim [in, out]
  torch::Tensor weight;

  MeanOptions options;

};

TORCH_MODULE(Mean);

TORCH_MODULE(Mean0);

} // namespace nn
} // namespace graph

#endif //GRAPH_LAYER_H
