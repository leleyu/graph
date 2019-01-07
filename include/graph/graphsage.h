//
// Created by leleyu on 2019-01-07.
//

#ifndef TEST_GRAPHSAGE_H
#define TEST_GRAPHSAGE_H

#include <graph/mean.h>


namespace graph {
namespace nn {

struct SupervisedGraphsage: torch::nn::Module {

  explicit SupervisedGraphsage(int n_class, int n_feature, int hidden_dim);

  torch::Tensor forward(const torch::Tensor& nodes,
    const torch::Tensor& features,
    const std::unordered_map<int, int>& node_to_index,
    const graph::dataset::AdjList& adj);

  torch::Tensor include_neibours(const torch::Tensor& nodes,
    const graph::dataset::AdjList& adj);

  torch::Tensor weight;
  int n_feature;
  graph::nn::Mean layer1{nullptr};
  graph::nn::Mean layer2{nullptr};
};
} // namespace nn
} // namespace graph

#endif //TEST_GRAPHSAGE_H
