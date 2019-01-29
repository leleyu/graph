//
// Created by leleyu on 2019-01-07.
//

#ifndef GRAPH_GRAPHSAGE_H
#define GRAPH_GRAPHSAGE_H

#include <graph/layer.h>


namespace graph {
namespace nn {

class UnSupervisedGraphsage : public torch::nn::Module {
 public:
  UnSupervisedGraphsage(int32_t input_dim,
                        const graph::Graph &graph,
                        const std::vector<int32_t> &output_dims,
                        const std::vector<int32_t> &num_samples,
                        const sampler::NeibourSampler &sampler);

  // Forward given a batch of nodes. This output the hidden
  // output of the last layer
  virtual torch::Tensor Forward(const NodeArray &nodes);

  // Compute the hidden output of nodes for layer ``layer``.
  torch::Tensor ComputeOutput(const NodeArray &nodes,
                              int layer);

  // loss function for unsupervised graphsage
  torch::Tensor PairwiseLoss(const torch::Tensor &src,
                             const torch::Tensor &dst,
                             const torch::Tensor &negs);

  // loss function without negative samples
  torch::Tensor PairwiseLoss(const torch::Tensor &src,
                             const torch::Tensor &dst);

  // save the embeddings of nodes and the nodes id_map
  void SaveOutput(const std::string &path);

  // Two layers with mean aggregate
  std::vector<graph::nn::Mean0> layers;
  const sampler::NeibourSampler &sampler;
  const std::vector<int32_t> &num_samples;
  const graph::Graph &graph;
};

/// Supervised GraphSage Model
class SupervisedGraphsage : public UnSupervisedGraphsage {
 public:
  SupervisedGraphsage(int32_t class_num,
                      int32_t input_dim,
                      const graph::Graph &graph,
                      const std::vector<int32_t> &output_dims,
                      const std::vector<int32_t> &num_samples,
                      const sampler::NeibourSampler &sampler);

  torch::Tensor Forward(const NodeArray& nodes) override;

  // The learned weight with dim [hidden_dim, n_class]
  torch::Tensor weight;
};

} // namespace nn
} // namespace graph

#endif //GRAPH_GRAPHSAGE_H
