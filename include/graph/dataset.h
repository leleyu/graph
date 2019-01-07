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
