//
// Created by leleyu on 2019-01-07.
//

#ifndef GRAPH_MEAN_H
#define GRAPH_MEAN_H

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

using namespace torch;
using namespace torch::nn;
using namespace graph::dataset;
using namespace graph::sampler;

class GraphLayer {



  /// forward with nodes and its neibours.
  /// nodes dimension [n_node], neibours dimension [n_node, n_nb]
  /// We assume that each node has the same number of neibors.
  /// If more, we do sampling among its neibours; if less, we set it as -1
  virtual Tensor forward(const Tensor& nodes, const Tensor& neibours,
                 const Tensor& features,
                 const std::unordered_map<int, int>& node_to_index);


};

/// Mean aggregator for neibours
class MeanImpl : public Cloneable<MeanImpl> {
public:
  MeanImpl(int in, int out): MeanImpl(MeanOptions(in, out)) {}
  explicit MeanImpl(MeanOptions options);

  /// Aggregate and calculate the mean of  features of neibours.
  /// Then, multiplying the mean with `weight` and a `relu` activation
  Tensor forward(const Tensor& nodes,
    const Tensor& features,
    const std::unordered_map<int, int>& node_to_index,
    const AdjList& adj);

  /// Aggregate the features of neibours without sampling
  Tensor aggregate(const Tensor& nodes,
    const Tensor& features,
    const std::unordered_map<int, int>& node_to_index,
    const AdjList& adj);

  void reset() override;

  /// The learned weight with dim [in, out]
  torch::Tensor weight;

  MeanOptions options;
};

class Mean0Impl : public Cloneable<Mean0Impl> {
public:
  Mean0Impl(int in, int out): Mean0Impl(MeanOptions(in, out)) {}
  explicit Mean0Impl(MeanOptions options);


  /// forward with nodes and its neibours.
  /// nodes dimension [n_node], neibours dimension [n_node, n_nb]
  /// We assume that each node has the same number of neibors.
  /// If more, we do sampling among its neibours; if less, we set it as -1
  virtual Tensor forward(const Tensor& nodes, const Tensor& neibours,
                         const Tensor& features,
                         const std::unordered_map<int, int>& node_to_index);

  void reset() override;

  /// The learned weight with dim [in, out]
  Tensor weight;

  MeanOptions options;

};

TORCH_MODULE(Mean);

} // namespace nn
} // namespace graph

#endif //GRAPH_MEAN_H
