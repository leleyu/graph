//
// Created by leleyu on 2019-01-07.
//

#ifndef GRAPH_LAYER_H
#define GRAPH_LAYER_H

#include <torch/nn/cloneable.h>
#include <torch/nn/module.h>
#include <torch/nn/pimpl.h>
#include <torch/types.h>

#include "sampler.h"

namespace graph {
namespace nn {

struct MeanOptions {
  MeanOptions(int in, int out);

  TORCH_ARG(int, in);

  TORCH_ARG(int, out);
};

class MeanImpl : public torch::nn::Cloneable<MeanImpl> {
 public:
  MeanImpl(int in, int out) : MeanImpl(MeanOptions(in, out)) {}

  explicit MeanImpl(MeanOptions options);

  torch::Tensor Forward(const NodeArray &nodes,
                        const NodeArray &neighbors,
                        const SparseNodeEmbedding &embedding,
                        const size_t num_sample);

  void reset() override;

  /// The learned weight with dim [in * 2, out]
  torch::Tensor weight;

  MeanOptions options;

};

TORCH_MODULE(Mean);

} // namespace nn
} // namespace graph

#endif //GRAPH_LAYER_H
