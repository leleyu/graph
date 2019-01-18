//
// Created by leleyu on 2019-01-07.
//

#ifndef GRAPH_GRAPHSAGE_H
#define GRAPH_GRAPHSAGE_H

#include <graph/layer.h>


namespace graph {
namespace nn {

using namespace torch::nn;
using namespace torch;
using namespace graph::dataset;

class UnSupervisedGraphsage: public Module {
public:
  explicit UnSupervisedGraphsage(int32_t input_dim, 
      const std::vector<int32_t>& output_dims,
      const std::vector<int32_t>& num_samples,
      const NeibourSampler& sampler);

  virtual Tensor forward(const Tensor& nodes,
                 const Tensor& features,
                 const std::unordered_map<int, int>& node_to_index,
                 const AdjList& adj);

  Tensor forward(const Tensor& nodes,
      const Tensor& features,
      const std::unordered_map<int32_t, int32_t>& node_to_index,
      const AdjList& adj);

  Tensor include_neibours(const Tensor& nodes,
                          const AdjList& adj);
  
  // construct neibours for a batch of nodes using the neibour sampler ``sampler``.
  // return a vector of pair <nodes, neibours> for each layer.
  // the number of samples for neibor is given is ``num_samples`` for each layer.
  std::vector<std::pair<Tensor, Tensor>> neibours(const Tensor& nodes,
      const AdjList& adj,
      NeibourSampler* sampler,
      const std::vector<int> num_samples);

  // loss function for unsupervised graphsage
  Tensor pairwise_loss(const Tensor& src, const Tensor& dst, const Tensor& negs);

  // loss function without negtive samples
  Tensor pairwise_loss(const Tensor& src, const Tensor& dst);

  // save the embeddings of nodes and the nodes id_map
  void save(const std::string& path, const Nodes& nodes, const AdjList& adj);

  // Two layers with mean aggregate
  std::vector<graph::nn::Mean> layers;
  const NeibourSampler& sampler;
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
