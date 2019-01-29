//
// Created by leleyu on 19-1-16.
//

#include <algorithm>

#include <graph/sampler.h>

namespace graph {
namespace sampler {

void UniformSampler::sample(const graph::Graph &graph,
                            const NodeArray &nodes,
                            size_t num_sample,
                            NodeArray* neighbors,
                            std::set<NodeId>* set) const {
  size_t num_node = nodes.size();
  for (size_t i = 0; i < num_node; i++) {
    NodeId node = nodes[i];
    set->insert(node);
    size_t start = i * num_sample;
    size_t degree = graph.GetDegree(node);
    if (degree != 0) {
      // has neighbor
      NodeId *ptr = const_cast<graph::Graph &>(graph).GetNeighborPtr(node);
      // WARN: the follow code will shuffle the order of neighbors.
      if (degree > num_sample)
        // more neighbors, shuffle
        std::random_shuffle(ptr, ptr + degree);

      // copy first min(num_sample, degree) element
      size_t len = std::min(num_sample, degree);
      for (size_t j = 0; j < len; j++) {
        (*neighbors)[start + j] = ptr[j];
        set->insert(ptr[j]);
      }

      // set the left as NAN_NODE_ID if need
      for (size_t j = len; j < num_sample; j++)
        (*neighbors)[start + j] = NAN_NODE_ID;
    }
  }
}


} // sampler
} // graph