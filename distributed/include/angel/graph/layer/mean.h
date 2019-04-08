//
// Created by leleyu on 2019-04-04.
//

#ifndef GRAPH_ANGEL_MEAN_H
#define GRAPH_ANGEL_MEAN_H

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include <graph/graph.h>
#include <angel/graph/data/subgraph.h>

namespace angel {
namespace graph {

struct MeanOptions {
  MeanOptions(int in, int out);

  TORCH_ARG(int, in);

  TORCH_ARG(int, out);
};

class MeanImpl : public torch::nn::Cloneable<MeanImpl> {
 public:
  MeanImpl(int in, int out) : MeanImpl(MeanOptions(in, out)) {}

  explicit MeanImpl(MeanOptions options);

  /**
   * Forward in the last layer, which directly use the embeddings for
   * neibors from ``neibor_embeddings``. The ``neibor_embeddings`` is
   * aggregated from parameter server.
   * @param nodes, nodes in this batch
   * @param self_embeddings, embeddings for this batch
   * @param neibor_embeddings, neibors' aggregate embeddings for this batch
   * @return
   */
  torch::Tensor Forward(const torch::Tensor& nodes,
    const torch::Tensor &self_embeddings,
    const torch::Tensor &neibor_embeddings);


  /**
   * Forward in the [0,last) layer, which requires to manually calculate
   * the aggregate embeddings for neibors
   * @param nodes, nodes in this batch
   * @param sub_graph, sub graph structure for this batch
   * @param first, the first-order nodes for this batch
   * @param embeddings, the embeddings for the first-order nodes
   * @return
   */
  torch::Tensor Forward(const torch::Tensor& nodes,
    const SubGraph& sub_graph,
    const torch::Tensor &first,
    const torch::Tensor &embeddings);

  /**
   * Combine the embeddings of self and neibors after aggregating.
   * @param self
   * @param neibors
   * @return
   */
  torch::Tensor Combine(const torch::Tensor &self,
    const torch::Tensor &neibors);

  void reset() override;

  /// The learned weight with dim [in * 2, out]
  torch::Tensor weight;

  MeanOptions options;

};

TORCH_MODULE(Mean);

} // namespace graph
} // namespace angel

#endif //GRAPH_ANGEL_MEAN_H
