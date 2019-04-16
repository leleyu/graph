//
// Created by leleyu on 2019-04-04.
//

#ifndef GRAPH_ANGEL_SUBGRAPH_H
#define GRAPH_ANGEL_SUBGRAPH_H

#include <torch/torch.h>

namespace angel {
namespace graph {

/**
 * For a sub-graph, we re-index the node id start from 1, since the 0-dim
 * is reserved for tomb (which means that we use ``0`` to represent that a
 * node has no neibor)
 */
struct SubGraph {
  const std::vector<int32_t>& nodes; // num_nodes = nodes.size(0)
  const std::vector<int32_t>& neibors;
  int64_t max_neibor;

  SubGraph(const std::vector<int32_t>& nodes,
    const std::vector<int32_t>& neibors,
    const int64_t max_neibor): nodes(nodes), neibors(neibors), max_neibor(max_neibor) {}

  /*
   * retrieve the first-order (including neibors and its self) for a batch of nodes
   */
  torch::Tensor FirstOrder(const torch::Tensor& batch) const {
    assert(batch.dim() == 1);

    // set for removing duplicated nodes
    std::set<int64_t> set;
    auto f1 = batch.accessor<int64_t, 1>();
    for (int64_t i = 0; i < f1.size(0); i++) {
      int64_t node = f1[i];
      set.insert(node);
      for (int64_t j = nodes[node]; j < nodes[node + 1]; j++)
        set.insert(neibors[j]);
    }

    auto size = static_cast<int64_t>(set.size());
    auto option = torch::TensorOptions().dtype(torch::kInt64)
      .requires_grad(false);
    auto first = torch::zeros({size}, option);
    auto f2 = first.accessor<int64_t, 1>();
    size_t idx = 0;
    for (auto node: set)
      f2[idx++] = node;

    return first;
  }

  /*
   * Construct the neibor index for batch nodes ``batch``,
   * the orders of nodes is given in tensor ``order``.
   */
  std::tuple<torch::Tensor, torch::Tensor>
  NeiborIndex(const torch::Tensor& batch,
    const torch::Tensor& orders) const {

    // map store the order/index
    std::unordered_map<int64_t, int64_t> position;
    auto f1 = orders.accessor<int64_t, 1>();
    for (size_t i = 0; i < f1.size(0); i++)
      position.insert(std::make_pair(f1[i], i));

    // init a index tensor with dim [batch_size, max_neibor]
    auto options = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
    auto index = torch::zeros({batch.size(0), max_neibor}, options);

    auto f2 = batch.accessor<int64_t, 1>();
    auto f3 = index.accessor<int64_t, 2>();

    auto length = torch::zeros({batch.size(0)}, options);
    auto f4 = length.accessor<int64_t, 1>();

    for (size_t i = 0; i < f2.size(0); i++) {
      int64_t node = f2[i];
      assert((nodes[node + 1] - nodes[node]) <= max_neibor);
      f4[i] = nodes[node + 1] - nodes[node];
      size_t idx = 0;
      // neibors for node
      for (size_t j = nodes[node]; j < nodes[node + 1]; j++)
        // set position, we add the position with 1 since we reserve the 0-dim as a tomb
        f3[i][idx++] = position[neibors[j]] + 1;
    }

    length = length.view({batch.size(0), 1});
    return std::make_tuple(index, length);
  }
};

} // namespace graph
} // namespace angel

#endif //GRAPH_ANGEL_SUBGRAPH_H
