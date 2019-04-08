#include <iostream>
#include <torch/torch.h>
#include <graph/graphsage.h>
#include <graph/metric.h>
#include <graph/preprocessing.h>

void TrainSupervisedGraphSage(graph::nn::SupervisedGraphsage *net,
                              const graph::NodeArray &train,
                              const graph::NodeArray &validate,
                              const graph::preprocessing::LabelIndex &labels,
                              size_t batch_size,
                              size_t num_epoch) {

  auto dataset = graph::NodeDataset(train, train.size());
  auto options = torch::data::DataLoaderOptions(batch_size);
  auto sampler = torch::data::samplers::RandomSampler(dataset.size().value());
  auto loader  = torch::data::make_data_loader(dataset, options, sampler);

  torch::optim::SGD optim(net->parameters(), 0.1);

  for (size_t epoch = 1; epoch <= num_epoch; epoch++) {
    size_t total_right = 0;

    size_t batch_index = 0;
    TIMER_START(epoch);

    for (auto batch : *loader) {
      optim.zero_grad();
      auto y_pred = net->Forward(batch);
      auto y_true = labels.Transform(batch);
      auto loss = labels.Loss(y_pred, y_true);
//      auto batch_right = graph::metric::PrecisionScore(y_pred, y_true) * y_pred.size(0);
      auto batch_right = labels.PrecisionScore(y_pred, y_true) * y_pred.size(0);
      total_right += batch_right;
      loss.backward();
      optim.step();
      batch_index++;
//      std::cout << "epoch=" << epoch << " "
//                << "batch_index=" << batch_index << " "
//                << "loss=" << loss << " "
//                << std::endl;
    }

    auto y_pred = net->Forward(validate);
    auto y_true = labels.Transform(validate);
    auto valid_precision = labels.PrecisionScore(y_pred, y_true);
    auto train_precision = total_right * 1.0 / dataset.size().value();

    TIMER_STOP(epoch);
    std::cout << "epoch " << epoch << " "
              << "train_precision=" << train_precision << " "
              << "valid_precision=" << valid_precision << " "
              << "epoch_time=" << TIMER_SEC(epoch) << "s"
              << std::endl;
  }
}


void RunSupervisedGraphSage(const graph::Graph& graph,
                            const graph::preprocessing::LabelIndex& labels,
                            int32_t n_class, int64_t n_dim) {

  graph::sampler::UniformSampler sampler;
  graph::nn::SupervisedGraphsage net(n_class, n_dim, graph, {64, 64}, {10, 10}, sampler);

  // split train and valid
  auto nodes = graph.GetNodeSet();

  graph::NodeArray train;
  graph::NodeArray validate;

  srand(time(NULL));
  for (auto node: nodes) {
    if (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < 0.3)
      train.push_back(node);
    else
      validate.push_back(node);
  }

  std::cout << "train.size()=" << train.size() << " "
            << "validate.size()=" << validate.size() << " "
            << std::endl;

  TrainSupervisedGraphSage(&net, train, validate, labels, 64, 50);
}

void RunCora() {
  const std::string edge_path = "../data/cora/cora.edge";
  const std::string feature_path = "../data/cora/cora.feature";
  const std::string label_path = "../data/cora/cora.label";

  int64_t n_node = 2708;
  int64_t n_dim = 1433;
  int32_t n_class = 7;

  graph::SparseNodeEmbedding input_embedding(n_node, n_dim);
  graph::LoadSparseNodeEmbedding(feature_path, &input_embedding);

  graph::preprocessing::SingleLabelIndex labels;
  labels.Load(label_path);

  graph::Graph graph(input_embedding);
  graph::LoadGraph(edge_path, &graph);

  graph.Build();

  RunSupervisedGraphSage(graph, labels, n_class, n_dim);
}

void RunBlogCatalog() {
  const std::string edge_path = "../data/blogCatalog/bc.edge";
  const std::string label_path = "../data/blogCatalog/bc_labels.txt";

  int64_t n_node = 10312;
  int64_t n_dim = 128;
  int32_t n_class = 1;

  graph::SparseNodeEmbedding input_embedding(n_node, n_dim);
  graph::Graph graph(input_embedding);
  graph::LoadGraph(edge_path, &graph);
  graph.Build();

  input_embedding.RandomInit(graph.GetNodeSet());

  graph::preprocessing::MultiLabelIndex labels;
  labels.Load(label_path);
  labels.SetCurrentClass(3);

  RunSupervisedGraphSage(graph, labels, n_class, n_dim);
}

int main() {
  RunCora();
//  RunBlogCatalog();
//  test_speed();
  return 0;
}
