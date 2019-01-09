#include <graph/graphsage.h>

namespace graph {
namespace nn {

using namespace torch;
using namespace torch::nn;
using namespace graph::dataset;

SupervisedGraphsage::SupervisedGraphsage(int n_class, int n_feature, int hidden_dim) {
  weight = register_parameter("weight", torch::rand({hidden_dim, n_class}));
  layer1 = register_module("layer1", graph::nn::Mean(n_feature, hidden_dim));
  layer2 = register_module("layer2", graph::nn::Mean(hidden_dim, hidden_dim));

  init::xavier_uniform_(weight);
}

Tensor SupervisedGraphsage::include_neibours(const Tensor& nodes,
  const AdjList &adj) {

  std::set<int> neibours;
  int n_nodes = static_cast<int>(nodes.size(0));
  auto accessor = nodes.accessor<int, 1>();
  for (int i = 0; i < n_nodes; i++) {
    int node = accessor[i];
    neibours.insert(node);

    auto it = adj.src_to_index.find(node);
    if (it != adj.src_to_index.end()) {
      int index = it->second;

      for (int j = adj.starts[index]; j < adj.starts[index + 1]; j++)
        neibours.insert(adj.dsts[j]);
    }
  }

  auto tensor = torch::empty({static_cast<int>(neibours.size())}, TensorOptions().dtype(torch::kInt32));
  int idx = 0;
  for (auto n : neibours) tensor[idx++] = n;
  return tensor;
}

Tensor SupervisedGraphsage::forward(const Tensor& nodes,
  const Tensor& features,
  const std::unordered_map<int, int>& node_to_index,
  const AdjList& adj) {

  // Include neibours of first-order
  auto first = include_neibours(nodes, adj);

  std::unordered_map<int, int> index1;
  int64_t n_first = first.size(0);
  auto accessor = first.accessor<int, 1>();
  for (int i = 0; i < n_first; i ++)
    index1[accessor[i]] = i;

  // h1 is [number_of_first, hidden_dim]
  auto h1 = layer1->forward(first, features, node_to_index, adj);

  // h2 is [number_of_node, hidden_dim]
  auto h2 = layer2->forward(nodes, h1, index1, adj);

  // output is [number_of_node, n_class]
  return relu(h2.mm(weight));
}

} // nn
} // graph