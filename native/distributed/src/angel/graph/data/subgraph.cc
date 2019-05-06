//
// Created by leleyu on 19-4-25.
//

#include <angel/graph/data/subgraph.h>
#include <angel/commons.h>

namespace angel {
namespace graph {

torch::Tensor SubGraph::GetFirstOrder(const torch::Tensor &batch) const {
  std::set<int32_t> set;
  DEFINE_ACCESSOR_DIM1_INT32(batch);
  for (int32_t i = 0; i < batch_acr.size(0); i++) {
    int32_t node = batch_acr[i];
    set.insert(node);
    for (int64_t j = nodes[node]; j < nodes[node + 1]; j++)
      set.insert(neighbors[j]);
  }

  auto size = static_cast<int64_t>(set.size());
  DEFINE_ZEROS_DIM1_INT32(first, size);
  DEFINE_ACCESSOR_DIM1_INT32(first);
  size_t idx = 0;
  for (auto node: set)
    first_acr[idx++] = node;

  return first;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
SubGraph::GetNeighborIndexValue(const torch::Tensor &batch, const torch::Tensor &orders) const {
  // map store the order->index
  std::unordered_map<int64_t, int64_t> position;
  auto f1 = orders.accessor<int64_t, 1>();
  for (size_t i = 0; i < f1.size(0); i++)
    position.insert(std::make_pair(f1[i], i));

  DEFINE_ZEROS_DIM2_INT64(index, batch.size(0), max_neighbor);
  DEFINE_ZEROS_DIM2_FLOAT(value, batch.size(0), max_neighbor);
  DEFINE_ZEROS_DIM1_INT64(length, batch.size(0));

  DEFINE_ACCESSOR_DIM2_INT64(index);
  DEFINE_ACCESSOR_DIM1_INT32(batch);
  DEFINE_ACCESSOR_DIM1_INT64(length);
  DEFINE_ACCESSOR_DIM2_FLOAT(value);

  for (size_t i = 0; i < batch_acr.size(0); i++) {
    int32_t node = batch_acr[i];
    length_acr[i] = nodes[node + 1] - nodes[node];
    size_t idx = 0;
    for (int32_t j = nodes[node]; j < nodes[node + 1]; j++) {
      index_acr[i][idx] = position[neighbors[j]];
      value_acr[i][idx] = 1.0f;
      idx++;
    }
  }

  length = length.view({batch.size(0), 1});
  value = value.view({batch.size(0), max_neighbor, 1});
  return std::make_tuple(index, value, length);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
SubGraph::GetNeighborIndexValue(const torch::Tensor &batch) const {
  DEFINE_ZEROS_DIM2_INT64(index, batch.size(0), max_neighbor);
  DEFINE_ZEROS_DIM2_FLOAT(value, batch.size(0), max_neighbor);
  DEFINE_ZEROS_DIM1_INT64(length, batch.size(0));

  DEFINE_ACCESSOR_DIM1_INT32(batch);
  DEFINE_ACCESSOR_DIM2_INT64(index);
  DEFINE_ACCESSOR_DIM2_FLOAT(value);
  DEFINE_ACCESSOR_DIM1_INT64(length);

  for (size_t i = 0; i < batch_acr.size(0); i++) {
    auto node = batch_acr[i];
    length_acr[i] = nodes[node + 1] - nodes[node];
    size_t idx = 0;
    for (int32_t j = nodes[node]; j < nodes[node + 1]; j++) {
      index_acr[i][idx] = neighbors[j];
      value_acr[i][idx] = 1.0f;
      idx++;
    }
  }

  length = length.view({batch.size(0), 1});
  value = value.view({batch.size(0), max_neighbor, 1});
  return std::make_tuple(index, value, length);
}


} // namespace graph
} // namespace angel
