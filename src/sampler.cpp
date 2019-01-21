//
// Created by leleyu on 19-1-16.
//

#include <graph/sampler.h>

namespace graph {
namespace sampler {

using namespace torch;
using namespace graph::dataset;

Tensor UniformSampler::sample(const graph::Graph &graph,
                              const NodeArray &nodes,
                              size_t num_sample) const {
  size_t num_node = nodes.size();

  auto neighbors = torch::zeros({static_cast<int64_t>(num_node),
                                 static_cast<int64_t>(num_sample)},
                                     TensorOptions().dtype(kInt32));

  TensorOptions().dtype(kI32);
  auto an = neighbors.accessor<int, 2>();
  for (int i = 0; i < an.size(0); i ++)
    for (int j = 0; j < an.size(1); j ++)
      an[i][j] = -1;


  for (size_t i = 0; i < num_node; i ++) {
    int node = a[i];
    if (graph.GetDegree(node))
    auto it = adj.src_to_index.find(node);
    if (it != adj.src_to_index.end()) {
      int idx = it->second;
      int n_neibors = adj.starts[idx+1] - adj.starts[idx];
      if (n_neibors <= num_sample) {
        // directly copy
        for (int j = 0; j < n_neibors; j ++)
          an[i][j] = adj.dsts[adj.starts[idx]+j];
      } else {
        // shuffle and take the first num_sample elements
        std::random_shuffle(&dsts[adj.starts[idx]], &dsts[adj.starts[idx+1]]);
        for (int j = 0; j < num_sample;j ++)
          an[i][j] = adj.dsts[adj.starts[idx]+j];
      }
    }
  }
  return neibors;
}


} // sampler
} // graph