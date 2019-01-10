//
// Created by leleyu on 19-1-10.
//

#include <graph/graphsage.h>


using namespace torch;
using namespace torch::data::datasets;
using namespace torch::data::samplers;
using namespace torch::optim;
using namespace graph::dataset;
using namespace graph::nn;


// calculate loss
void loss(Tensor u, Tensor v, Tensor neg) {

}

void train_unsupervised_graphsage(const AdjList& adj,
    const Nodes& nodes, const Edges& edges,
    int batch_size, int hidden_dim) {
  auto dataset = EdgeDataset(edges.srcs, edges.dsts);
  auto sampler = RandomSampler(dataset.size().value());
  auto options = torch::data::DataLoaderOptions(batch_size);
  auto loader  = torch::data::make_data_loader(dataset, options, sampler);

  int n_feature = static_cast<int>(nodes.features.size(1));
  UnSupervisedGraphsage net(n_feature, hidden_dim);
  SGD optim(net.parameters(), 0.01);


  for (int epoch = 0; epoch < 10; epoch ++) {

    for (auto batch : *loader) {
      optim.zero_grad();

      auto srcs = batch[0][0];
      auto dsts = batch[0][1];

      int batch_size = srcs.size(0);

      auto src_output = net.forward(srcs, nodes.features, nodes.node_to_index, adj);
      auto dst_output = net.forward(dsts, nodes.features, nodes.node_to_index, adj);

//      std::cout << src_output.sizes() << std::endl;
//      std::cout << dst_output.sizes() << std::endl;


      auto dot = torch::matmul(src_output.view({batch_size, 1, hidden_dim,}),
          dst_output.view({batch_size, hidden_dim, 1})).squeeze();

//      std::cout << dot.sizes() << std::endl;

      torch::Tensor undefined;
      auto loss = torch::binary_cross_entropy_with_logits(dot, torch::ones(srcs.size(0)),
          undefined, undefined, 1);
      loss.backward();
      optim.step();

      std::cout << loss.item() << std::endl;
    }
  }
}

int main() {
  std::string edge_path = "../data/cora/cora.adjs";
  std::string node_path = "../data/cora/cora.content.id";
  AdjList adj;
  Nodes nodes;
  Edges edges;
  load_edges(edge_path, &adj);
  load_edges(edge_path, &edges);
  load_features(node_path, &nodes, 1433, 2708);
  train_unsupervised_graphsage(adj, nodes, edges, 128, 10);
}