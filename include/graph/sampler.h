//
// Created by leleyu on 19-1-16.
//

#ifndef GRAPH_SAMPLER_H
#define GRAPH_SAMPLER_H

#include <torch/torch.h>

#include <graph/dataset.h>
#include <graph/graph.h>

namespace graph {
namespace sampler {

class NeibourSampler {

public:
  virtual torch::Tensor sample(const graph::Graph& graph,
                               const NodeArray& nodes,
                               size_t num_sample) const = 0;

};



class UniformSampler: public NeibourSampler {

public:
  virtual torch::Tensor sample(const graph::Graph &graph,
                               const NodeArray &nodes,
                               size_t num_sample) const override;
};

} // sampler
} // graph

#endif //GRAPH_SAMPLER_H
