#include <iostream>
#include <torch/torch.h>
#include <graph/graphsage.h>
#include <graph/metric.h>

void TrainSupervisedGraphSage(graph::nn::SupervisedGraphsage *net,
                              const graph::NodeArray &train,
                              const graph::NodeArray &validate,
                              const graph::NodeLabels &labels,
                              size_t batch_size,
                              size_t num_epoch) {

  auto dataset = graph::NodeDataset(train, train.size());
  auto options = torch::data::DataLoaderOptions(batch_size);
  auto sampler = torch::data::samplers::RandomSampler(dataset.size().value());
  auto loader  = torch::data::make_data_loader(dataset, options, sampler);

  torch::optim::SGD optim(net->parameters(), 0.01);

  for (size_t epoch = 1; epoch <= num_epoch; epoch++) {
    size_t total_right = 0;
    for (auto const& batch : *loader) {
      auto y_pred = net->Forward(batch);
      auto y_true = graph::LookupLabels(batch, labels);
      auto loss = torch::nll_loss(log_softmax(y_pred, 1), y_true);
      auto batch_right = graph::metric::PrecisionScore(y_pred, y_true) * y_pred.size(0);
      total_right += static_cast<long>(batch_right);
      loss.backward();
      optim.step();
    }

    auto y_pred = net->Forward(validate);
    auto y_true = graph::LookupLabels(validate, labels);
    auto valid_precision = graph::metric::PrecisionScore(y_pred, y_true);
    auto train_precision = total_right * 1.0 / dataset.size().value();
    std::cout << "epoch " << epoch << " "
              << "train precision=" << train_precision << " "
              << "validate precision=" << valid_precision
              << std::endl;
  }
}


void RunSupervisedGraphSage() {
  const std::string edge_path = "../data/cora/cora.edge";
  const std::string feature_path = "../data/cora/cora.feature";
  const std::string label_path = "../data/cora/cora.label";

  int64_t n_node = 2708;
  int64_t n_dim = 1433;
  int32_t n_class = 7;

  graph::SparseNodeEmbedding input_embedding(n_node, n_dim);
  graph::LoadSparseNodeEmbedding(feature_path, &input_embedding);

  graph::NodeLabels labels;
  graph::LoadNodeLabels(label_path, &labels);

  graph::Graph graph(input_embedding);
  graph::LoadGraph(edge_path, &graph);

  graph.Build();

  graph::sampler::UniformSampler sampler;
  graph::nn::SupervisedGraphsage net(n_class, n_dim, graph, {50, 50}, {5, 5}, sampler);

  // split train and valid
  auto nodes = graph.GetNodeSet();

  graph::NodeArray train;
  graph::NodeArray validate;

  srand(time(NULL));
  for (auto node: nodes) {
    if (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < 0.5)
      train.push_back(node);
    else
      validate.push_back(node);
  }

  std::cout << "train.size()=" << train.size() << " "
            << "validate.size()=" << validate.size() << " "
            << std::endl;

  TrainSupervisedGraphSage(&net, train, validate, labels, 64, 50);
}

int main() {
  RunSupervisedGraphSage();
  return 0;
}
