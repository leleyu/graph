//
// Created by leleyu on 2019-01-07.
//

#ifndef GRAPH_GRAPHSAGE_H
#define GRAPH_GRAPHSAGE_H

#include <graph/mean.h>


namespace graph {
namespace nn {

using namespace torch::nn;
using namespace torch;
using namespace graph::dataset;

/// Supervised GraphSage Model
struct SupervisedGraphsage: Module {

  explicit SupervisedGraphsage(int n_class, int n_feature, int hidden_dim);

  Tensor forward(const Tensor& nodes,
    const Tensor& features,
    const std::unordered_map<int, int>& node_to_index,
    const AdjList& adj);

  Tensor include_neibours(const Tensor& nodes,
    const AdjList& adj);

  // The learned weight with dim [hidden_dim, n_class]
  Tensor weight;

  // number of input features
  int n_feature;

  // Two layers with mean aggregate
  graph::nn::Mean layer1{nullptr};
  graph::nn::Mean layer2{nullptr};
};
} // namespace nn
} // namespace graph

#endif //GRAPH_GRAPHSAGE_H
