//
// Created by leleyu on 2019-01-07.
//

#ifndef GRAPH_GRAPHSAGE_H
#define GRAPH_GRAPHSAGE_H

#include <graph/layer.h>


namespace graph {
namespace nn {

namespace th = torch;
namespace gd = graph::dataset;
namespace gs = graph::sampler;

class UnSupervisedGraphsage : public th::nn::Module {
public:
  explicit UnSupervisedGraphsage(int32_t input_dim,
                                 const std::vector<int32_t> &output_dims,
                                 const std::vector<int32_t> &num_samples,
                                 const gs::NeibourSampler &sampler);

  virtual th::Tensor forward(const th::Tensor &nodes,
                         const th::Tensor &features,
                         const std::unordered_map<int, int> &node_to_index,
                         const gd::AdjList &adj);

  th::Tensor include_neibours(const th::Tensor &nodes,
                          const gd::AdjList &adj);

  // construct neibours for a batch of nodes using the neibour sampler ``sampler``.
  // return a vector of pair <nodes, neibours> for each layer.
  // the number of samples for neibor is given is ``num_samples`` for each layer.
  std::vector<std::pair<th::Tensor, th::Tensor>> neibours(
      const th::Tensor &nodes,
      const gd::AdjList &adj,
      const gs::NeibourSampler &sampler,
      const std::vector<int> num_samples);

  // loss function for unsupervised graphsage
  th::Tensor pairwise_loss(const th::Tensor &src, const th::Tensor &dst, const th::Tensor &negs);

  // loss function without negtive samples
  th::Tensor pairwise_loss(const th::Tensor &src, const th::Tensor &dst);

  // save the embeddings of nodes and the nodes id_map
  void save(const std::string &path, const gd::Nodes &nodes, const gd::AdjList &adj);

  // Two layers with mean aggregate
  std::vector<graph::nn::Mean0> layers;
  const gs::NeibourSampler &sampler;
  const std::vector<int32_t> &num_samples;
};

/// Supervised GraphSage Model
class SupervisedGraphsage : public UnSupervisedGraphsage {
public:
  explicit SupervisedGraphsage(int32_t n_class, int32_t input_dim, int hidden_dim);

  th::Tensor forward(const th::Tensor &nodes,
                 const th::Tensor &features,
                 const std::unordered_map<int, int> &node_to_index,
                 const gd::AdjList &adj) override;

  // The learned weight with dim [hidden_dim, n_class]
  th::Tensor weight;
};

} // namespace nn
} // namespace graph

#endif //GRAPH_GRAPHSAGE_H
