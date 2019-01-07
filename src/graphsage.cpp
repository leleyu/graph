#include <graph/graphsage.h>

namespace graph {
namespace nn {

SupervisedGraphsage::SupervisedGraphsage(int n_class, int n_feature, int hidden_dim) {
  weight = register_parameter("weight", torch::rand({hidden_dim, n_class}));
  layer1 = register_module("layer1", graph::nn::Mean(n_feature, hidden_dim));
  layer2 = register_module("layer2", graph::nn::Mean(hidden_dim, hidden_dim));

  torch::nn::init::xavier_uniform_(weight);
}

torch::Tensor SupervisedGraphsage::include_neibours(const torch::Tensor &nodes,
  const graph::dataset::AdjList &adj) {

  std::set<int> neibours;
  int n_nodes = static_cast<int>(nodes.size(0));
  for (int i = 0; i < n_nodes; i++) {
    int node = nodes[i].item().toInt();
    neibours.insert(node);

    int index = adj.src_to_index.find(node)->second;
  
    for (int j = adj.starts[index]; j < adj.starts[index + 1]; j++)
      neibours.insert(adj.dsts[j]);
  }

  auto tensor = torch::empty({static_cast<int>(neibours.size())});
  int idx = 0;
  for (auto n : neibours) tensor[idx++] = n;
  return tensor;
}

torch::Tensor SupervisedGraphsage::forward(const torch::Tensor &nodes,
  const torch::Tensor &features,
  const std::unordered_map<int, int>& node_to_index,
  const graph::dataset::AdjList &adj) {

  auto first = include_neibours(nodes, adj);
  std::cout << "first=" << first << std::endl;

  std::unordered_map<int, int> index1;
  int64_t n_first = first.size(0);
  for (int i = 0; i < n_first; i ++)
    index1[first[i].item().toInt()] = i;

  auto h1 = layer1->forward(first, features, node_to_index, adj);

  std::cout << "h1.dim()=" << h1.dim() << " h1.shape()=" << h1.size(0) << "," << h1.size(1) << std::endl;

  auto h2 = layer2->forward(nodes, h1, index1, adj);

  std::cout << "h2.dim()=" << h2.dim() << " h2.shape()=" << h2.size(0) << "," << h2.size(1) << std::endl;


  return torch::relu(h2.mm(weight));
}

} // nn
} // graph