//
// Created by leleyu on 19-1-16.
//

#include <graph/sampler.h>

namespace graph {
namespace sampler {

using namespace torch;
using namespace graph::dataset;

Tensor UniformSampler::sample(const graph::dataset::AdjList &adj,
    const torch::Tensor &nodes, int num_sample) const {
  auto a = nodes.accessor<int, 1>();

  auto neibors = torch::zeros({a.size(0), num_sample}, TensorOptions().dtype(kInt32));
  auto an = neibors.accessor<int, 2>();
  for (int i = 0; i < an.size(0); i ++)
    for (int j = 0; j < an.size(1); i ++)
      an[i][j] = -1;


  for (int i = 0; i < a.size(0); i ++) {
    int node = a[i];
    auto it = adj.src_to_index.find(node);
    if (it != adj.src_to_index.end()) {
      int idx = it->second;
      int n_neibors = adj.starts[idx+1] - adj.starts[idx];
      for (int j = adj.starts[idx]; j < adj.starts[idx+1]; j++) {

      }
    }
  }

}


} // sampler
} // graph