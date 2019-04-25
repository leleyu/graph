//
// Created by leleyu on 2019-04-04.
//

#ifndef GRAPH_ANGEL_SUBGRAPH_H
#define GRAPH_ANGEL_SUBGRAPH_H

#include <torch/torch.h>

namespace angel {
namespace graph {

/**
 * For a sub-graph, we re-index the node id start from 0
 */
struct SubGraph {
  const std::vector<int32_t> &nodes; // num_nodes = nodes.size(0)
  const std::vector<int32_t> &neighbors;
  int64_t max_neighbor;

  SubGraph(const std::vector<int32_t> &nodes,
           const std::vector<int32_t> &neighbors,
           const int64_t max_neighbor) : nodes(nodes), neighbors(neighbors), max_neighbor(max_neighbor) {}

  /*
   * retrieve the first-order (including neighbors and its self) for a batch of nodes
   */
  torch::Tensor GetFirstOrder(const torch::Tensor &batch) const;

  /*
   * Construct the neighbor index for batch nodes ``batch``,
   * the orders of nodes is given in tensor ``order``.
   */
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  GetNeighborIndexValue(const torch::Tensor &batch, const torch::Tensor &orders) const;

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  GetNeighborIndexValue(const torch::Tensor &batch) const;
};

} // namespace graph
} // namespace angel

#endif //GRAPH_ANGEL_SUBGRAPH_H
