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
  virtual void sample(const graph::Graph& graph,
                      const NodeArray& nodes,
                      size_t num_sample,
                      NodeArray* neighbors,
                      std::set<NodeId>* set) const = 0;

};



class UniformSampler: public NeibourSampler {

 public:
  // Uniformly sample ``num_sample`` neighbors for each node in ``nodes`` and
  // store the neighbors in ``neighbors``.
  // Note that this method will shuffle the order of neighbors for some nodes.
  virtual void sample(const graph::Graph &graph,
                      const NodeArray &nodes,
                      size_t num_sample,
                      NodeArray* neighbors,
                      std::set<NodeId>* set) const override;
};

} // sampler
} // graph

#endif //GRAPH_SAMPLER_H
