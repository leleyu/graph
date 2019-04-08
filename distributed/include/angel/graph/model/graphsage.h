//
// Created by leleyu on 2019-04-03.
//

#ifndef GRAPH_ANGEL_GRAPHSAGE_H
#define GRAPH_ANGEL_GRAPHSAGE_H

#include <angel/graph/data/subgraph.h>
#include <angel/graph/layer/mean.h>

#include <torch/torch.h>

namespace angel {
namespace graph {

class SupervisedGraphSage : public torch::nn::Module {
 public:
  SupervisedGraphSage(int input_dim,
    int num_class,
    const std::vector<int32_t>& output_dims);

  /**
   * Forward method for SupervisedGraphSage
   * @param nodes
   * @param sub_graph
   * @param self_embeddings
   * @param neibor_embeddings
   * @return
   */
  virtual torch::Tensor Forward(const torch::Tensor& nodes,
    const SubGraph& sub_graph,
    const torch::Tensor& self_embeddings,
    const torch::Tensor& neibor_embeddings);

  /**
   * Compute the output of layer ``layer``, this method will recursively compute
   * the embedding output of layer ``layer`` for nodes ``nodes``.
   * @param nodes, the set of nodes need to compute embedding output
   * @param layer, the layer
   * @param sub_graph, graph structure for this batch of nodes
   * @param self_embeddings, embedding vectors for each node in this sub-graph
   * @param neibor_embeddings, raw neibor embeddings (after aggregate) for each node in this sub-graph
   * @return the embedding outputs for ``nodes``
   */
  torch::Tensor ComputeOutput(const torch::Tensor& nodes,
    int layer,
    const SubGraph& sub_graph,
    const torch::Tensor& self_embeddings,
    const torch::Tensor& neibor_embeddings);

  virtual std::map<std::string, torch::Tensor> Backward(const torch::Tensor &nodes,
    const SubGraph& sub_graph,
    const torch::Tensor &self_embeddings,
    const torch::Tensor &neibor_embeddings,
    const torch::Tensor &targets);

  int32_t GetDim();

  std::vector<Mean> layers;
  torch::Tensor weight_;
  int num_class_;
};

} // namespace angel
} // namespace angel

#endif //GRAPH_ANGEL_GRAPHSAGE_H
