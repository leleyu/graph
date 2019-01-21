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

namespace th = torch;
namespace gd = graph::dataset;

class GraphLayer {

  /// forward with nodes and its neibours.
  /// nodes dimension [n_node], neibours dimension [n_node, n_nb]
  /// We assume that each node has the same number of neibors.
  /// If more, we do sampling among its neibours; if less, we set it as -1
  virtual th::Tensor forward(const th::Tensor& nodes, const th::Tensor& neibours,
                 const th::Tensor& features,
                 const std::unordered_map<int, int>& node_to_index);


};

/// Mean aggregator for neibours
class MeanImpl : public th::nn::Cloneable<MeanImpl> {
public:
  MeanImpl(int in, int out): MeanImpl(MeanOptions(in, out)) {}
  explicit MeanImpl(MeanOptions options);

  /// Aggregate and calculate the mean of  features of neibours.
  /// Then, multiplying the mean with `weight` and a `relu` activation
  th::Tensor forward(const th::Tensor& nodes,
    const th::Tensor& features,
    const std::unordered_map<int, int>& node_to_index,
    const gd::AdjList& adj);

  /// Aggregate the features of neibours without sampling
  th::Tensor aggregate(const th::Tensor& nodes,
    const th::Tensor& features,
    const std::unordered_map<int, int>& node_to_index,
    const gd::AdjList& adj);

  void reset() override;

  /// The learned weight with dim [in, out]
  torch::Tensor weight;

  MeanOptions options;
};

class Mean0Impl : public th::nn::Cloneable<Mean0Impl> {
public:
  Mean0Impl(int in, int out): Mean0Impl(MeanOptions(in, out)) {}
  explicit Mean0Impl(MeanOptions options);


  /// forward with nodes and its neibours.
  /// nodes dimension [n_node], neibours dimension [n_node, n_nb]
  /// We assume that each node has the same number of neibors.
  /// If more, we do sampling among its neibours; if less, we set it as -1
  virtual th::Tensor forward(const th::Tensor& nodes, const th::Tensor& neibours,
                         const th::Tensor& features,
                         const std::unordered_map<int, int>& node_to_index);

  void reset() override;

  /// The learned weight with dim [in, out]
  th::Tensor weight;

  MeanOptions options;

};

TORCH_MODULE(Mean);

TORCH_MODULE(Mean0);

} // namespace nn
} // namespace graph

#endif //GRAPH_LAYER_H
