#include <iostream>
#include "torch/torch.h"
#include <graph/dataset.h>
#include <graph/libsvm.h>
#include <graph/model.h>
#include <graph/dataset.h>
#include <graph/graphsage.h>



void train_lr(const std::string& input, size_t n_dim, size_t batch_size) {

  auto dataset = LibsvmDataset(input, n_dim);
  auto options = torch::data::DataLoaderOptions(batch_size);
  auto samplers = torch::data::samplers::SequentialSampler(dataset.size().value());
  auto data_loader = torch::data::make_data_loader(dataset, options, samplers);

  LogisticRegression lr(n_dim);

  // Instantiate an Adam optimizer algorithm to update the parameters
  torch::optim::SGD optimizer(lr.parameters(), 0.01);

  for (size_t epoch = 1; epoch <= 10; ++epoch) {
    float loss_sum = 0.0;
    int n_right = 0;
    int n_total = 0;
    for (auto batch : *data_loader) {
      optimizer.zero_grad();
      auto data = batch[0].data;
      auto label = batch[0].target;
      auto prediction = lr.forward(data);
      if (prediction.item().toFloat() >= 0.5 and label.item().toFloat() >= 0.5)
        n_right ++;
      if (prediction.item().toFloat() < 0.5 and label.item().toFloat() < 0.5)
        n_right ++;
      auto loss = torch::binary_cross_entropy(prediction, label);
      loss.backward();
      optimizer.step();
      loss_sum += loss.item().toFloat();
      n_total ++;
    }
    
    std::cout << "epoch=" << epoch << " loss=" << loss_sum / n_total << std::endl;
    std::cout << "precision=" << n_right * 1.0 / n_total << std::endl;
  }
}

void train_graphsage(const std::string& edge_path, const std::string& node_path,
  int n_feature, int n_node, int n_class, int n_hidden) {
  using namespace graph::dataset;
  using namespace graph::nn;
  using namespace torch::optim;

  AdjList adj;
  load_edges(edge_path, &adj);

  Nodes nodes;
  load_features(node_path, &nodes, n_feature, n_node);

  SupervisedGraphsage net(n_class, n_feature, n_hidden);

  Adam optim(net.parameters(), 0.0001);

  for (int epoch = 1; epoch <= 10; epoch ++) {
    auto n = torch::empty({1}, TensorOptions().dtype(kInt32));
    auto l = torch::empty({1}, TensorOptions().dtype(kInt64));
    for (int node = 0; node < n_node; node ++) {

      optim.zero_grad();
      n[0] = node;

      int index = nodes.node_to_index.find(node)->second;

      l[0] = nodes.labels[index].item().toLong();
      std::cout << "node=" << node << std::endl;

      auto it = adj.src_to_index.find(node);
      if (it != adj.src_to_index.end()) {
        std::cout << "node=" << node << std::endl;
        auto output = net.forward(n, nodes.features, nodes.node_to_index, adj);
        auto loss = nll_loss(log_softmax(output, 1), l);
        std::cout << output << std::endl;
        std::cout << loss.item() << std::endl;
        loss.backward();
        optim.step();

      }
     
    }
  }



}

void run_lr() {
  std::string input = "../data/a9a_123d_train.libsvm";
  size_t n_dim = 124;
  train_lr(input, n_dim, 1);
}

void run_graphsage() {
  std::string edge_path = "../data/cora/cora.adjs";
  std::string node_path = "../data/cora/cora.content.id";
  train_graphsage(edge_path, node_path, 1433, 2708, 7, 128);

}

int main() {
  run_graphsage();
  return 0;
}
