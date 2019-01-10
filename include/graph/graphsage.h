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

class UnSupervisedGraphsage: public Module {
public:
  explicit UnSupervisedGraphsage(int n_feature, int hidden_dim);

  virtual Tensor forward(const Tensor& nodes,
                 const Tensor& features,
                 const std::unordered_map<int, int>& node_to_index,
                 const AdjList& adj);

  Tensor include_neibours(const Tensor& nodes,
                          const AdjList& adj);

  // Two layers with mean aggregate
  graph::nn::Mean layer1{nullptr};
  graph::nn::Mean layer2{nullptr};
};

/// Supervised GraphSage Model
class SupervisedGraphsage: public UnSupervisedGraphsage {
public:
  explicit SupervisedGraphsage(int n_class, int n_feature, int hidden_dim);

  Tensor forward(const Tensor& nodes,
    const Tensor& features,
    const std::unordered_map<int, int>& node_to_index,
    const AdjList& adj) override;

  // The learned weight with dim [hidden_dim, n_class]
  Tensor weight;
};

} // namespace nn
} // namespace graph

#endif //GRAPH_GRAPHSAGE_H
