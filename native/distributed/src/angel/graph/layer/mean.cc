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
  const torch::Tensor &neibors) {
  // cat self and neibor
  std::vector<torch::Tensor> tensors;
  tensors.push_back(self);
  tensors.push_back(neibors);
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
  torch::Tensor length;
  std::tie(index, length) = sub_graph.NeighborIndex(nodes, first);
  auto neighbors = torch::embedding(embeddings, index, -1, false, true);
  // calculate average for each node.
  neighbors = neighbors.sum(1).div(length);
  auto self = torch::embedding(embeddings, nodes, -1, false, true);

  return Combine(self, neighbors);
}

torch::Tensor MeanImpl::Forward(const torch::Tensor &nodes,
    const SubGraph &sub_graph,
    const torch::Tensor &embeddings) {
  torch::Tensor index;
  torch::Tensor length;
  std::tie(index, length) = sub_graph.NeighborIndex(nodes);

  auto neighbors = torch::embedding(embeddings, index, -1, false, true);
  neighbors = neighbors.sum(1).div(length);
  auto self = torch::embedding(embeddings, nodes, -1, false, true);
  return Combine(self, neighbors);
}

torch::Tensor MeanImpl::Forward(const torch::Tensor &nodes,
  const torch::Tensor &self_embeddings,
  const torch::Tensor &neibor_embeddings) {

  // batch_size = nodes.size(0)
  // embedding_dim = self_embeddings.size(1)

  assert(self_embeddings.dim() == 2);
  assert(self_embeddings.dim() == 2);

  // self is [batch_size, embedding_dim]
  auto self = torch::embedding(self_embeddings, nodes, -1, false, true);
  // neighbor is [batch_size, embedding_dim]
  auto neibor = torch::embedding(neibor_embeddings, nodes, -1, false, true);

  return Combine(self, neibor);
}
} // namespace graph
} // namespace angel
