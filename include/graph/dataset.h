//
// Created by leleyu on 2018-12-25.
//

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <torch/torch.h>

namespace graph {
namespace dataset {


torch::data::Example<> parseLibSVM(std::string str);
    
torch::data::Example<> parseLibSVM(std::string str, size_t n_dim);


struct LibsvmDataset : torch::data::datasets::Dataset<LibsvmDataset> {
  explicit LibsvmDataset(std::string path, size_t n_dim) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
      auto example = graph::dataset::parseLibSVM(line);
      examples.push_back(example);
    }
  }

  torch::data::Example<> get(size_t index) override {
    return examples[index];
  }

  torch::optional<size_t> size() const override {
    return examples.size();
  }

private:
  std::vector<torch::data::Example<>> examples;
};

struct NodeDataset: torch::data::datasets::Dataset<NodeDataset, int> {
  explicit NodeDataset(const std::vector<int>& nodes, size_t size): nodes(nodes), num(size) {}

  int get(size_t index) override {
    return nodes[index];
  }

  torch::optional<size_t> size() const override {
    return num;
  }

  const std::vector<int>& nodes;
  size_t num;
};

struct EdgeDataset: torch::data::datasets::Dataset<EdgeDataset, torch::Tensor> {

  explicit EdgeDataset(const std::vector<int>& srcs,
      const std::vector<int>& dsts): srcs(srcs), dsts(dsts) {}

  torch::Tensor get(size_t index) override {
    return torch::zeros({1});
  }

  std::vector<torch::Tensor> get_batch(torch::ArrayRef<size_t> indices) override {
    int size = static_cast<int>(indices.size());
    auto edges = torch::empty({2, size}, torch::TensorOptions().dtype(torch::kInt32));
    auto src = edges[0];
    auto dst = edges[1];
    auto src_accessor = src.accessor<int, 1>();
    auto dst_accessor = dst.accessor<int, 1>();

    int idx = 0;
    for (const auto i : indices) {
      src_accessor[idx] = srcs[i];
      dst_accessor[idx] = dsts[i];
      idx++;
    }
    std::vector<torch::Tensor> r;
    r.push_back(edges);
    return r;
  }

  torch::optional<size_t> size() const override {
    return srcs.size();
  }

  const std::vector<int>& srcs;
  const std::vector<int>& dsts;
};

struct AdjList {
  std::vector<int> starts;
  std::vector<int> dsts;
  std::unordered_map<int, int> src_to_index;
};

struct Edges {
  std::vector<int> srcs;
  std::vector<int> dsts;
};

struct Nodes {
  torch::Tensor features;
  torch::Tensor labels;
  std::unordered_map<int, int> node_to_index;
};

void load_edges(const string& path, AdjList* adj);

void load_edges(const string& path, Edges* edges);

void load_features(const string& path, Nodes *node,
  int n_feature, int n_node);

// Generate random walk with Adj, for each node, we generate `n_walks` walks
// with each walk of `n_length`. The generated walks is stored in a tensor with
// dim [n_walks*size(nodes), n_length]
torch::Tensor random_walk(const AdjList& adj, int n_walks, int n_length);

// Generate negative sampling for nodes in `nodes`, for each node, we generate
// `n_neg` negative samples that are not similar with this node.
// The return tensor contains size(nodes)*n_neg negatives samples.
torch::Tensor negative_sampling(const AdjList& adj,
  const torch::Tensor& nodes, int n_neg, int n_nodes);

} // namespace dataset
} // namespace graph


#endif //TEST_UTILS_H
