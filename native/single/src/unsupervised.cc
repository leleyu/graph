//
// Created by leleyu on 19-1-10.
//
#include <iostream>
#include <torch/torch.h>
#include <graph/graphsage.h>
#include <graph/metric.h>

void TrainUnSupervisedGraphSage(graph::nn::UnSupervisedGraphsage *net,
                                const graph::Graph &graph,
                                size_t batch_size,
                                size_t num_epoch) {
  auto dataset = graph::EdgeDataset(graph.GetEdgeInfo());
  auto options = torch::data::DataLoaderOptions(batch_size);
  auto sampler = torch::data::samplers::RandomSampler(dataset.size().value());
  auto loader = torch::data::make_data_loader(dataset, options, sampler);

  torch::optim::Adam optim(net->parameters(), 0.005);

  for (size_t epoch = 1; epoch <= num_epoch; epoch++) {
    double loss_sum = 0.0;
    size_t batch_index = 0;
    for (auto batch : *loader) {
      optim.zero_grad();
      TIMER_START(forward);
      auto output = net->Forward(batch);
      int64_t batch_size = batch.size();
      auto srcs = output.slice(0, 0, batch_size / 2);
      auto dsts = output.slice(0, batch_size / 2, batch_size);
      auto loss = net->PairwiseLoss(srcs, dsts);
      TIMER_STOP(forward);

      TIMER_START(backward);
      loss.backward();
      TIMER_STOP(backward);

      TIMER_START(optim);
      optim.step();
      TIMER_STOP(optim);

      loss_sum += loss.item().toDouble() * batch.size() / 2;

      std::cout << "epoch " << epoch << " "
                << "batch_index=" << batch_index++ << " "
                << "batch_size=" << batch_size << " "
                << "forward_time=" << TIMER_SEC(forward) << "s "
                << "backward_time=" << TIMER_SEC(backward) << "s "
                << "optim_time=" << TIMER_SEC(optim) << "s "
                << "loss=" << loss.item().toFloat()
                << std::endl;
    }

    std::cout << "epoch " << epoch << " "
              << "loss=" << loss_sum / dataset.size().value()
              << std::endl;
  }
}

void RunUnSupervisedGraphSage(const graph::Graph &graph, std::string output = "") {
  graph::sampler::UniformSampler sampler;

  int64_t input_dim = graph.GetInputEmbedding().GetDim();
  graph::nn::UnSupervisedGraphsage net(input_dim, graph,
                                       {40, 10}, {10, 5}, sampler);

  TrainUnSupervisedGraphSage(&net, graph, 64, 1);

  if (output.size() > 0)
    net.SaveOutput(output);
}

void RunCora() {
  const std::string edge_path = "../data/cora/cora.edge";
  const std::string feature_path = "../data/cora/cora.feature";

  int64_t n_node = 2708;
//  int64_t n_dim = 1433;
  int64_t n_dim = 100;

  graph::SparseNodeEmbedding input_embedding(n_node, n_dim);
//  graph::LoadSparseNodeEmbedding(feature_path, &input_embedding);

  graph::Graph graph(input_embedding);
  graph::LoadGraph(edge_path, &graph);
  graph.Build();

  input_embedding.RandomInit(graph.GetNodeSet());

  RunUnSupervisedGraphSage(graph, "cora");
}

void RunBlogCatalog() {
  const std::string edge_path = "../data/blogCatalog/bc.edge";

  int64_t n_node = 10312;
  int64_t n_dim = 128;

  graph::SparseNodeEmbedding input_embedding(n_node, n_dim);
  graph::Graph graph(input_embedding);
  graph::LoadGraph(edge_path, &graph);
  graph.Build();

  std::cout << graph.GetNumNode() << std::endl;
  std::cout << graph.GetNumEdge() << std::endl;

  input_embedding.RandomInit(graph.GetNodeSet());

  RunUnSupervisedGraphSage(graph, "bc");
}

int main() {
//  RunCora();
  RunBlogCatalog();
  return 0;
}