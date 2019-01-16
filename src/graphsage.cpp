#include <graph/graphsage.h>

namespace graph {
namespace nn {

using namespace torch;
using namespace torch::nn;
using namespace graph::dataset;

UnSupervisedGraphsage::UnSupervisedGraphsage(int n_feature, int hidden_dim) {
  layer1 = register_module("layer1", graph::nn::Mean(n_feature, hidden_dim));
  layer2 = register_module("layer2", graph::nn::Mean(hidden_dim, hidden_dim));
}

Tensor UnSupervisedGraphsage::include_neibours(const Tensor& nodes,
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

Tensor UnSupervisedGraphsage::forward(const Tensor& nodes,
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

  // do normalization
  auto norm = h2.norm(2, 1).view({h2.size(0), 1}).clamp_min(10e-15);
  h2 = h2.div(norm);

  return h2;
}

Tensor UnSupervisedGraphsage::pairwise_loss(const at::Tensor &src, const at::Tensor &dst,
    const at::Tensor &negs) {
  // src [batch_size, dim], dst [batch_size, dim], negs [batch_size, neg_num, dim]
  auto batch_size = src.size(0);
  auto dim = src.size(1);
  auto sv = src.view({batch_size, dim, 1});
  auto dv = dst.view({batch_size, 1, dim});
  // positive loss between srcs and dsts
  auto pos_loss = -torch::log(torch::sigmoid(dv.matmul(sv)));
  // negative loss between srcs and negative nodes
  auto neg_loss = -torch::log(torch::sigmoid(negs.matmul(sv)));
  // return average loss
  auto size = batch_size + batch_size * neg_loss.size(1);
  return (pos_loss.sum() + neg_loss.sum()) / size;
}

Tensor UnSupervisedGraphsage::pairwise_loss(const at::Tensor &src, const at::Tensor &dst) {
  // src [batch_size, dim], dst [batch_size, dim]
  auto batch_size = src.size(0);
  auto dim = src.size(1);
  auto sv = src.view({batch_size, dim, 1});
  auto dv = dst.view({batch_size, 1, dim});
  // positive loss between srcs and dsts
  auto pos_loss = -torch::log(torch::sigmoid(dv.matmul(sv)));

  return pos_loss.squeeze().mean();
}

void UnSupervisedGraphsage::save(const std::string &path,
    const graph::dataset::Nodes &nodes,
    const graph::dataset::AdjList &adj) {

  int64_t n_node = nodes.node_to_index.size();
  Tensor node_ids = torch::empty({n_node}, TensorOptions().dtype(kInt32));
  auto accessor = node_ids.accessor<int, 1>();
  size_t idx = 0;
  for (auto node: nodes.node_to_index) {
    accessor[idx++] = node.first;
  }

  auto embeddings = forward(node_ids, nodes.features, nodes.node_to_index, adj);
  FILE* f = fopen((path + "/embedding.b").c_str(), "wb");

  fwrite(embeddings.data_ptr(), sizeof(float), n_node * embeddings.size(1), f);
  fclose(f);

  f = fopen((path + "/id_map.b").c_str(), "wb");

  fwrite(node_ids.data_ptr(), sizeof(int), n_node, f);
  fclose(f);
}

SupervisedGraphsage::SupervisedGraphsage(int n_class, int n_feature, int hidden_dim):
  UnSupervisedGraphsage(n_feature, hidden_dim) {
  weight = register_parameter("weight", torch::rand({hidden_dim, n_class}));
  init::xavier_uniform_(weight);
}

Tensor SupervisedGraphsage::forward(const Tensor& nodes,
  const Tensor& features,
  const std::unordered_map<int, int>& node_to_index,
  const AdjList& adj) {

  auto h2 = UnSupervisedGraphsage::forward(nodes, features, node_to_index, adj);

  // output is [number_of_node, n_class]
  auto output =  relu(h2.mm(weight));
  return output;
}

} // nn
} // graph