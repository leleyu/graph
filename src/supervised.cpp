#include <iostream>
#include <torch/torch.h>
#include <graph/dataset.h>
#include <graph/graphsage.h>

//void train_graphsage(const std::string& edge_path, const std::string& node_path,
//  int n_feature, int n_node, int n_class, int n_hidden, int batch_size = 128) {
//  using namespace graph::dataset;
//  using namespace graph::nn;
//  using namespace torch::optim;
//  using namespace torch::data;
//
//  std::vector<int> node_ids;
//  node_ids.resize(n_node);
//  int n_val = 500;
//
//  for (int i = 0; i < n_node; i++) node_ids[i] = i;
//
//  std::random_shuffle(node_ids.begin(), node_ids.end());
//
//  auto dataset = NodeDataset(node_ids, n_node - n_val);
//  auto options = DataLoaderOptions(batch_size);
//  auto sampler = samplers::RandomSampler(dataset.size().value());
//  auto loader  = make_data_loader(dataset, options, sampler);
//
//  AdjList adj;
//  load_edges(edge_path, &adj);
//  Nodes nodes;
//  load_features(node_path, &nodes, n_feature, n_node);
//
//  SupervisedGraphsage net(n_class, n_feature, n_hidden);
//
//  SGD optim(net.parameters(), 0.02);
//
//  auto n = torch::empty({batch_size}, TensorOptions().dtype(kInt32));
//  auto l = torch::empty({batch_size}, TensorOptions().dtype(kInt64));
//
//  // validate data
//  auto val_n = torch::empty({n_val}, TensorOptions().dtype(kInt32));
//  auto val_l = torch::empty({n_val}, TensorOptions().dtype(kInt64));
//
//  for (int i = 0; i < n_val; i ++) {
//    int node = node_ids[n_node - 1 - i];
//    val_n[i] = node;
//    int index = nodes.node_to_index.find(node)->second;
//    val_l[i] = nodes.labels[index];
//  }
//
//  for (int epoch = 1; epoch < 10; epoch ++) {
//    auto loss_sum = 0.0;
//    int cnt = 0;
//    int n_right = 0;
//    int n_total = 0;
//    auto start = std::chrono::system_clock::now();
//
//    for (auto batch: *loader) {
//      // If this batch is not enough data, pass it.
//      if (batch.size() != batch_size)
//        continue;
//
//      // copy nodes data
//      memcpy(n.data_ptr(), batch.data(), batch_size*sizeof(int));
//      // copy labels
//      for (int i = 0; i < batch_size; i ++) {
//        int index = nodes.node_to_index.find(batch[i])->second;
//        l[i] = nodes.labels[index].item().toLong();
//      }
//
//      auto output = net.forward(n, nodes.features, nodes.node_to_index, adj);
//      auto loss = nll_loss(log_softmax(output, 1), l);
//
//      auto y = output.argmax(1).to(TensorOptions().dtype(kInt64));
//      n_right += torch::eq(y, l).sum().item().toInt();
//      n_total += batch_size;
//
//      loss.backward();
//      optim.step();
//      loss_sum += loss.item().toDouble() * batch_size;
//      cnt += batch_size;
//    }
//
//    auto val_output = net.forward(val_n, nodes.features, nodes.node_to_index, adj);
//    auto p = calculate_precision(val_output, val_l);
//
//    auto end = std::chrono::system_clock::now();
//    std::chrono::duration<double> cost = end - start;
//
//    std::cout << " loss=" << loss_sum / cnt
//              << " precision=" << n_right * 1.0 / n_total
//              << " val_precision=" << p
//              << " time=" << cost.count() << "s" << std::endl;
//  }
//}

void TrainSupervisedGraphSage(graph::nn::SupervisedGraphsage *net,
                              const graph::NodeDataset &train,
                              const graph::NodeArray &validate,
                              size_t batch_size,
                              size_t num_epoch) {

  auto options = torch::data::DataLoaderOptions(batch_size);
  auto sampler = torch::data::samplers::RandomSampler(train.size().value());
  auto loader = make_data_loader(train, options, sampler);

  for (size_t epoch = 1; epoch <= num_epoch; epoch++) {
    for (auto batch : *loader) {
      auto output = net->Forward(batch);
    }
  }
}


void RunSupervisedGraphSage() {
  const std::string edge_path = "../data/cora/cora.edge";
  const std::string feature_path = "../data/cora/cora.feature";

  int64_t n_node = 2708;
  int64_t n_dim = 1433;
  int32_t n_class = 7;


  graph::SparseNodeEmbedding input_embedding(n_node, n_dim);
  graph::LoadSparseNodeEmbedding(feature_path, &input_embedding);

  graph::Graph graph(input_embedding);
  graph::LoadGraph(edge_path, &graph);

  graph.Build();

  graph::sampler::UniformSampler sampler;
  graph::nn::SupervisedGraphsage net(n_class, n_dim, graph, {20, 10}, {10, 5}, sampler);

//  graph::NodeId node_id = 2;
//  size_t d = graph.GetDegree(node_id);
//  graph::NodeId *ptr = graph.GetNeighborPtr(node_id);
//  for (size_t i = 0; i < d; i++)
//    std::cout << ptr[i] << std::endl;
}

int main() {
  RunSupervisedGraphSage();
  return 0;
}
