//
// Created by leleyu on 2019-04-04.
//

#include <angel/graph/layer/mean.h>

namespace angel {
namespace graph {

MeanOptions::MeanOptions(int in, int out) : in_(in), out_(out) {}

MeanImpl::MeanImpl(MeanOptions options) : options(options) {
  reset();
}

void MeanImpl::reset() {
  weight = register_parameter("weight", torch::zeros({options.in_ * 2, options.out_}));
  torch::nn::init::xavier_uniform_(weight);
}

torch::Tensor MeanImpl::Combine(const torch::Tensor &self,
  const torch::Tensor &neighbors) {
  // cat self and neighbors
  std::vector<torch::Tensor> tensors;
  tensors.push_back(self);
  tensors.push_back(neighbors);
  torch::TensorList list(tensors.data(), tensors.size());
  auto combine = torch::cat(list, 1);
  return torch::relu(combine.matmul(weight));
}

torch::Tensor MeanImpl::Forward(
  const torch::Tensor &nodes,
  const SubGraph &sub_graph,
  const torch::Tensor &first,
  const torch::Tensor &embeddings) {

  // construct the index with sub-graph (neighbors)
  torch::Tensor index;
  torch::Tensor value;
  torch::Tensor length;
  std::tie(index, value, length) = sub_graph.GetNeighborIndexValue(nodes, first);
  // index is [batch_size, max_neighbor], value is [batch_size, max_neighbor, 1]
  // length is [batch_size, 1]
  auto neighbors = torch::embedding(embeddings, index, -1, false, true);
  // neighbors is [batch_size, max_neighbor, dim]

  neighbors = neighbors.mul(value);
  // calculate average for each node.
  neighbors = neighbors.sum(2).div(length);
  auto self = torch::embedding(embeddings, nodes, -1, false, true);

  return Combine(self, neighbors);
}

torch::Tensor MeanImpl::Forward(const torch::Tensor &nodes,
    const SubGraph &sub_graph,
    const torch::Tensor &embeddings) {
  torch::Tensor index;
  torch::Tensor value;
  torch::Tensor length;
  std::tie(index, value, length) = sub_graph.GetNeighborIndexValue(nodes);

  auto neighbors = torch::embedding(embeddings, index, -1, false, true);
  neighbors = neighbors.mul(value);
  neighbors = neighbors.sum(2).div(length);
  auto self = torch::embedding(embeddings, nodes, -1, false, true);
  return Combine(self, neighbors);
}

} // namespace graph
} // namespace angel
