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
  explicit NodeDataset(const std::vector<int>& nodes): nodes(nodes) {}

  int get(size_t index) override {
    return nodes[index];
  }

  torch::optional<size_t> size() const override {
    return nodes.size();
  }

  const std::vector<int>& nodes;
};

struct AdjList {
  std::vector<int> starts;
  std::vector<int> dsts;
  std::unordered_map<int, int> src_to_index;
};

struct Nodes {
  torch::Tensor features;
  torch::Tensor labels;
  std::unordered_map<int, int> node_to_index;
};

void load_edges(const string& path, AdjList* adj);

void load_features(const string& path, Nodes *node,
  int n_feature, int n_node);

} // namespace dataset
} // namespace graph


#endif //TEST_UTILS_H
