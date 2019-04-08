#include <graph/graphsage.h>

namespace graph {
namespace nn {

UnSupervisedGraphsage::UnSupervisedGraphsage(
    int64_t input_dim,
    const graph::Graph &graph,
    const std::vector<int32_t> &output_dims,
    const std::vector<int32_t> &num_samples,
    const graph::sampler::NeibourSampler &sampler)
    : graph(graph), sampler(sampler), num_samples(num_samples) {
  size_t n_layers = output_dims.size();
  for (size_t i = 0; i < n_layers; i++) {
    layers.push_back(register_module("layer" + std::to_string(i), graph::nn::Mean(input_dim, output_dims[i])));
    input_dim = output_dims[i];
  }
}

torch::Tensor UnSupervisedGraphsage::Forward(const NodeArray &nodes) {
  auto output = ComputeOutput(nodes, layers.size() - 1);
  // normalization
  auto norm = output.norm(2, 1)
      .view({-1, 1})
      .clamp_min(10e-15);
  return output.div(norm);
}

torch::Tensor
UnSupervisedGraphsage::ComputeOutput(const NodeArray &nodes,
  int layer) {

  size_t num_neighbors = nodes.size() * num_samples[layer];
  NodeArray neighbors;
  neighbors.resize(num_neighbors);
  std::set<NodeId> set;
  sampler.sample(graph, nodes, num_samples[layer], &neighbors, &set);

  if (layer > 0) {
    // Compute the output of its neighbors and self
    NodeArray first_order;
    first_order.resize(set.size());
    size_t idx = 0;

    for (auto node : set)
      first_order[idx++] = node;

    auto output = ComputeOutput(first_order, layer - 1);
    SparseNodeEmbedding embedding(first_order, output);
    return layers[layer]->Forward(nodes, neighbors, embedding, num_samples[layer]);
  } else {
    // use the input embeddings
    const SparseNodeEmbedding &embedding = graph.GetInputEmbedding();
    return layers[layer]->Forward(nodes, neighbors, embedding, num_samples[layer]);
  }
}

torch::Tensor UnSupervisedGraphsage::PairwiseLoss(const torch::Tensor &src,
                                                  const torch::Tensor &dst,
                                                  const torch::Tensor &negs) {
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

torch::Tensor UnSupervisedGraphsage::PairwiseLoss(const torch::Tensor &src,
                                                  const at::Tensor &dst) {
  // src [batch_size, dim], dst [batch_size, dim]
  auto batch_size = src.size(0);
  auto dim = src.size(1);
  auto sv = src.view({batch_size, dim, 1});
  auto dv = dst.view({batch_size, 1, dim});
  // positive loss between srcs and dsts
  auto pos_loss = -torch::log(torch::sigmoid(dv.matmul(sv)));

  return pos_loss.squeeze().mean();
}

void UnSupervisedGraphsage::SaveOutput(const std::string &path) {
  size_t num_node = graph.GetNumNode();
  NodeArray nodes;
  nodes.resize(num_node);
  auto nodes_set = graph.GetNodeSet();
  size_t idx = 0;
  for (auto node : nodes_set)
    nodes[idx++] = node;

  auto output = Forward(nodes);
  FILE *f = fopen((path + "/embedding.b").c_str(), "wb");
  fwrite(output.data_ptr(), sizeof(float), num_node * output.size(1), f);
  fclose(f);

  f = fopen((path + "/id_map.b").c_str(), "wb");
  fwrite(nodes.data(), sizeof(NodeId), num_node, f);
  fclose(f);
}

SupervisedGraphsage::SupervisedGraphsage(
    int32_t class_num,
    int64_t input_dim,
    const graph::Graph &graph,
    const std::vector<int32_t> &output_dims,
    const std::vector<int32_t> &num_samples,
    const sampler::NeibourSampler &sampler)
    : UnSupervisedGraphsage(input_dim, graph,
                            output_dims, num_samples, sampler) {
  int32_t dim = *output_dims.rbegin();
  weight = register_parameter("weight", torch::rand({dim, class_num}));
  torch::nn::init::xavier_uniform_(weight);
}

torch::Tensor SupervisedGraphsage::SingleOutputLoss(torch::Tensor y_pred, torch::Tensor y_true) {
  return torch::nll_loss(log_softmax(y_pred, 1), y_true);
}

torch::Tensor SupervisedGraphsage::MultiOutputLoss(torch::Tensor y_pred, torch::Tensor y_true) {
  auto loss =  -y_true * torch::log_sigmoid(y_pred) + (1 - y_true) * torch::log_sigmoid(-y_pred);
  loss = loss.sum(1) / y_pred.size(1);
  return loss.mean();
}

torch::Tensor SupervisedGraphsage::Forward(
    const NodeArray &nodes) {
  auto output = UnSupervisedGraphsage::Forward(nodes);

  // output is [number_of_node, class_num]
  return relu(output.mm(weight));
}

} // nn
} // graph
