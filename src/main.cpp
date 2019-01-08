#include <iostream>
#include "torch/torch.h"
#include <graph/dataset.h>
#include <graph/model.h>
#include <graph/dataset.h>
#include <graph/graphsage.h>



void train_lr(const std::string& input, size_t n_dim, size_t batch_size) {

  auto dataset = graph::dataset::LibsvmDataset(input, n_dim);
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
  int n_feature, int n_node, int n_class, int n_hidden, int batch_size = 126) {
  using namespace graph::dataset;
  using namespace graph::nn;
  using namespace torch::optim;
  using namespace torch::data;

  std::vector<int> node_ids;
  node_ids.resize(n_node);
  for (int i = 0; i < n_node; i++) node_ids[i] = i;
  auto dataset = NodeDataset(node_ids);
  auto options = DataLoaderOptions(batch_size);
  auto sampler = samplers::RandomSampler(dataset.size().value());
  auto loader  = make_data_loader(dataset, options, sampler);


  AdjList adj;
  load_edges(edge_path, &adj);

  Nodes nodes;
  load_features(node_path, &nodes, n_feature, n_node);

  SupervisedGraphsage net(n_class, n_feature, n_hidden);

  SGD optim(net.parameters(), 0.05);

  auto n = torch::empty({batch_size}, TensorOptions().dtype(kInt32));
  auto l = torch::empty({batch_size}, TensorOptions().dtype(kInt64));

  for (int epoch = 1; epoch < 10; epoch ++) {
    auto loss_sum = 0.0;
    int cnt = 0;
    for (auto batch: *loader) {
      batch_size = batch.size();
      
      if (batch_size != n.size(0)) {
        n.resize_({batch_size});
        l.resize_({batch_size});
      }

      memcpy(n.data_ptr(), batch.data(), batch_size*sizeof(int));
      for (int i = 0; i < batch_size; i ++) {
        int index = nodes.node_to_index.find(batch[i])->second;
        l[i] = nodes.labels[index].item().toLong();
      }

      auto output = net.forward(n, nodes.features, nodes.node_to_index, adj);
      auto loss = nll_loss(log_softmax(output, 1), l);
      loss.backward();
      optim.step();
      loss_sum += loss.item().toDouble() * batch_size;
      cnt += batch_size;
    }

    std::cout << loss_sum / cnt << std::endl;
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
