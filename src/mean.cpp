
#include <graph/mean.h>

namespace graph {
namespace nn {

MeanOptions::MeanOptions(int in, int out): in_(in), out_(out) {}

MeanImpl::MeanImpl(MeanOptions options): options(options) {
  reset();
}

torch::Tensor MeanImpl::forward(const torch::Tensor& nodes,
  const torch::Tensor& features,
  const std::unordered_map<int, int>& node_to_index,
  const graph::dataset::AdjList& adj) {

  assert(nodes.dim() == 1);

  // neibour_features is [n_node, n_feature]
  auto neibour_features = aggregate(nodes, features, node_to_index, adj);

//  std::cout << "neibour_features dim=" << neibour_features.dim() << " shape=" << neibour_features.size(0) << ","
//    << neibour_features.size(1) << std::endl;
  
  int n_nodes = static_cast<int>(nodes.size(0));

  // self is [n_node, n_feature]
  auto self = torch::zeros({n_nodes, options.in_});
  for (int i = 0; i < n_nodes; i ++) {
    int node = nodes[i].item().toInt();
    int index = node_to_index.find(node)->second;
    self[i] = features[index];
  }

//  std::cout << "self dim=" << self.dim() << " shape=" << self.size(0) << "," << self.size(1) << std::endl;

  // torch::TensorList list({self, neibour_features});
  std::vector<torch::Tensor> tensors;
  tensors.push_back(self);
  tensors.push_back(neibour_features);
  torch::TensorList list(tensors.data(), tensors.size());

  // combined is [n_node, n_feature * 2]
  auto combined = torch::cat(list, 1);

//  std::cout << "combine dim=" << combined.dim() << " shape=" << combined.size(0) << "," << combined.size(1) << std::endl;

  // output [n_node, output_dim]
  return relu(combined.mm(weight));
}

torch::Tensor MeanImpl::aggregate(const torch::Tensor &nodes,
  const torch::Tensor &features,
  const std::unordered_map<int, int> &node_to_index,
  const graph::dataset::AdjList &adj) {

//  std::cout << "aggregate" << std::endl;

  assert(nodes.dim() == 1);
  int64_t num_nodes = nodes.size(0);
  // aggregate and calculate the mean value of neibours
  auto neibours = torch::zeros({num_nodes, options.in_});

  for (int i = 0; i < num_nodes; i ++) {
    int node = nodes[i].item().toInt();
    auto it = adj.src_to_index.find(node);
    if (it != adj.src_to_index.end()) {
      int index = it->second;
      int num_neibours = adj.starts[index+1] - adj.starts[index];
      auto indices = torch::zeros({num_neibours}, torch::TensorOptions().dtype(torch::kInt64));
      auto neibour_features = torch::zeros({num_neibours, options.in_});
      for (int j = 0; j < num_neibours; j ++) {
        int n = adj.dsts[adj.starts[index] + j];
        int idx = node_to_index.find(n)->second;
        neibour_features[j] = features[idx];
      }
      neibours[i] = neibour_features.mean();
    }
  }
  return neibours;
}

void MeanImpl::reset() {
  // weight, [2*in, out]
  weight = register_parameter("weight", torch::zeros({options.in_ * 2, options.out_}));
  torch::nn::init::xavier_uniform_(weight);
}

} // namespace nn
} // namespace graph