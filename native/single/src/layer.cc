
#include <graph/layer.h>

namespace graph {
namespace nn {

MeanOptions::MeanOptions(int in, int out) : in_(in), out_(out) {}

torch::Tensor MeanImpl::Forward(
    const NodeArray &nodes,
    const NodeArray &neighbors,
    const SparseNodeEmbedding &embedding,
    const size_t num_sample) {
  auto num_node = static_cast<int64_t>(nodes.size());
  int64_t dim = embedding.GetDim();
  assert (dim == options.in_);

  // combine is [batch_size x 2 x dim]
  auto combine = torch::zeros({num_node, 2, dim});
//  combine.set_requires_grad(false);


  for (size_t i = 0; i < num_node; i++)
    combine[i][0] = embedding.Lookup(nodes[i]);

  for (size_t i = 0; i < nodes.size(); i++) {
    size_t start = i * num_sample;
    auto f = combine[i][1];
    int64_t cnt = 0;
    for (size_t j = 0; j < num_sample; j++) {
      NodeId neighbor = neighbors[start + j];
      if (neighbor != NAN_NODE_ID) {
        f.add_(embedding.Lookup(neighbor));
        cnt++;
      }
    }
    if (cnt > 0)
      f.div_(cnt);
  }

  combine = combine.view({num_node, 2 * dim});
  return torch::relu(combine.mm(weight));

}

MeanImpl::MeanImpl(MeanOptions options) : options(options) {
  reset();
}

void MeanImpl::reset() {
  weight = register_parameter("weight", torch::zeros({options.in_ * 2, options.out_}));
  torch::nn::init::xavier_uniform_(weight);
}

} // namespace nn
} // namespace graph
