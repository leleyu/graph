
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
  
  int n_nodes = static_cast<int>(nodes.size(0));
  auto accessor = nodes.accessor<int, 1>();

  // self is [n_node, n_feature]
  std::vector<torch::Tensor> selfs;
  for (int i = 0; i < n_nodes; i ++) {
    int node = accessor[i];
    int index = node_to_index.find(node)->second;
    selfs.push_back(features[index].view({1, options.in_}));
  }

  torch::TensorList self_lists(selfs.data(), selfs.size());
  auto self = torch::cat(self_lists, 0);

  std::vector<torch::Tensor> tensors;
  tensors.push_back(self);
  tensors.push_back(neibour_features);
  torch::TensorList list(tensors.data(), tensors.size());

  // combined is [n_node, n_feature * 2]
  auto combined = torch::cat(list, 1);

  // output [n_node, output_dim]
  return relu(combined.mm(weight));
}

torch::Tensor MeanImpl::aggregate(const torch::Tensor &nodes,
  const torch::Tensor &features,
  const std::unordered_map<int, int> &node_to_index,
  const graph::dataset::AdjList &adj) {

  assert(nodes.dim() == 1);
  int64_t num_nodes = nodes.size(0);
  auto accessor = nodes.accessor<int, 1>();

  // aggregate and calculate the mean value of neibours

  std::vector<torch::Tensor> means;

  for (int i = 0; i < num_nodes; i ++) {
    int node = accessor[i];
    auto it = adj.src_to_index.find(node);
    auto nb = torch::zeros({options.in_});

    if (it != adj.src_to_index.end()) {

      int index = it->second;
      int num_neibours = adj.starts[index+1] - adj.starts[index];

      for (int j = 0; j < num_neibours; j ++) {
        int n = adj.dsts[adj.starts[index] + j];
        int idx = node_to_index.find(n)->second;
        nb.add_(features[idx]);
      }

      nb.div_(num_neibours);
    }

    means.push_back(nb.view({1, options.in_}));
  }

  torch::TensorList list(means.data(), means.size());
  return torch::cat(list, 0);
}

void MeanImpl::reset() {
  // weight, [2*in, out]
  weight = register_parameter("weight", torch::zeros({options.in_ * 2, options.out_}));
  torch::nn::init::xavier_uniform_(weight);
}

} // namespace nn
} // namespace graph