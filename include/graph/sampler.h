//
// Created by leleyu on 19-1-16.
//

#ifndef GRAPH_SAMPLER_H
#define GRAPH_SAMPLER_H

#include <torch/torch.h>
#include <graph/dataset.h>

namespace graph {
namespace sampler {

using namespace torch;
using namespace graph::dataset;

class NeibourSampler {

public:
  virtual Tensor sample(const AdjList& adj, const Tensor& nodes, int num_sample) const;

};



class UniformSampler: public NeibourSampler {

public:
  Tensor sample(const AdjList &adj, const Tensor &nodes, int num_sample) const override;
};

} // sampler
} // graph

#endif //GRAPH_SAMPLER_H
