//
// Created by leleyu on 19-1-10.
//

#include <graph/graphsage.h>
#include <graph/model.h>
#include <graph/metric.h>

using namespace torch;
using namespace torch::data;
using namespace torch::data::datasets;
using namespace torch::data::samplers;
using namespace torch::optim;
using namespace graph::dataset;
using namespace graph::nn;
using namespace graph::metric;

UnSupervisedGraphsage train(const AdjList& adj,
    const Nodes& nodes, const Edges& edges,
    const int batch_size,
    const int hidden_dim,
    const int n_epoch,
    const int n_node,
    const int n_negative = 3) {

  auto dataset = EdgeDataset(edges.srcs, edges.dsts);
  auto sampler = RandomSampler(dataset.size().value());
  auto options = torch::data::DataLoaderOptions(batch_size);
  auto loader  = torch::data::make_data_loader(dataset, options, sampler);

  int n_feature = static_cast<int>(nodes.features.size(1));
  UnSupervisedGraphsage net(n_feature, hidden_dim);
  SGD optim(net.parameters(), 0.1);

  for (int epoch = 0; epoch < n_epoch; epoch ++) {

    int batch_cnt = 0;
    for (auto batch : *loader) {
      auto start = std::chrono::system_clock::now();
      optim.zero_grad();

      auto srcs = batch[0][0];
      auto dsts = batch[0][1];

      int src = srcs.item().toInt();
      int dst = dsts.item().toInt();
      std::cout << "src " << src << " degree " << adj.degree(src) << std::endl;
      std::cout << "dst " << dst << " degree " << adj.degree(dst) << std::endl;
      
      int batch_size = srcs.size(0);
//      auto negs = negative_sampling(adj, srcs, n_negative, n_node).view({batch_size * n_negative});

      std::cout << "forward src" << std::endl;

      auto src_output = net.forward(srcs, nodes.features, nodes.node_to_index, adj);
      std::cout << "forward dst" << std::endl;
      auto dst_output = net.forward(dsts, nodes.features, nodes.node_to_index, adj);
//      std::cout << "forward neg" << std::endl;
//      auto neg_output = net.forward(negs, nodes.features, nodes.node_to_index, adj).view({batch_size, n_negative, hidden_dim});

      std::cout << "pariwise_loss" << std::endl;
      auto loss = net.pairwise_loss(src_output, dst_output);
      std::cout << "backward" << std::endl;
      loss.backward();

      std::cout << "step" << std::endl;
      optim.step();
      auto end = std::chrono::system_clock::now();
      std::chrono::duration<double> cost = end - start;

      std::cout << " epoch=" << epoch
                << " batch=" << batch_cnt
                << " batch_size=" << batch_size
                << " loss=" << loss.item()
                << " time=" << cost.count() << "s"
                << std::endl;
      batch_cnt ++;
    }
  }
  return net;
}

void train_mlp(const Tensor& features,
               const Tensor& targets,
               int n_feature, int n_node, int n_class, int batch_size) {
  LogisticRegression net(n_feature, n_class);
  SGD optim(net.parameters(), 0.5);

  std::vector<int> node_ids;
  node_ids.resize(n_node);
  for (int i = 0; i < n_node; i ++) node_ids[i] = i;
  std::random_shuffle(node_ids.begin(), node_ids.end());

  int n_val = 500;


  auto dataset = NodeDataset(node_ids, n_node - n_val);
  auto sampler = RandomSampler(dataset.size().value());
  auto option  = DataLoaderOptions(batch_size);
  auto loader  = make_data_loader(dataset, option, sampler);

  // batch data holder
  auto batch_f = torch::empty({batch_size, n_feature});
  batch_f.set_requires_grad(false);
  auto batch_t = torch::empty({batch_size}, TensorOptions().dtype(kInt64));

  // validate data
  auto validate_f = torch::empty({n_val, n_feature});
  auto validate_t = torch::empty({n_val}, TensorOptions().dtype(kInt64));

  for (int i = 0; i < n_val; i ++) {
    int node = node_ids[n_node - 1 - i];
    validate_f[i] = features[node];
    validate_t[i] = targets[node];
  }

  for (int epoch = 0; epoch < 100; epoch ++) {
    int idx = 0;
    for (auto batch: *loader) {
      optim.zero_grad();
      if (batch.size() != batch_size)
        continue;

      // build batch input
      for (int i = 0; i < batch_size; i ++) {
        int idx = batch[i];
        batch_f[i] = features[idx];
        batch_t[i] = targets[idx];
      }

      auto output = net.forward(batch_f);
      auto loss = nll_loss(log_softmax(output, 1), batch_t);
      auto p = precision_score(output, batch_t);

      loss.backward();
      optim.step();
    }

    // validate
    auto validate_output = net.forward(validate_f);
    auto validate_p = precision_score(validate_output, validate_t);
    std::cout << " epoch=" << epoch
              << " val_p=" << validate_p
              << std::endl;
  }

}

void classify(int n_feature, int n_class, int n_node, const Nodes& nodes) {
  Tensor features, targets;
  torch::load(features, "features.pt");
  torch::load(targets, "targets.pt");

  std::cout << features.sizes() << std::endl;
  std::cout << targets.sizes()  << std::endl;

  features.set_requires_grad(false);
  std::vector<Tensor> tensors;
  tensors.push_back(features);
  tensors.push_back(nodes.features);
  TensorList list(tensors.data(), tensors.size());
  auto cat_features = torch::cat(list, 1);

  cat_features.set_requires_grad(false);

//  std::cout << features[0] << std::endl;

  train_mlp(cat_features, targets, cat_features.size(1), n_node, n_class, 128);
}

void run_cora() {
  std::string edge_path = "../data/cora/cora.adjs";
  std::string node_path = "../data/cora/cora.content.id";
  AdjList adj;
  Nodes nodes;
  Edges edges;
  int n_node = 2708;
//  int n_features = 1433;
  int n_features = 256;
  load_edges(edge_path, &adj);
  load_edges(edge_path, &edges);
//  load_features(node_path, &nodes, n_features, n_node);
  random_features(node_path, &nodes, n_features, n_node);
  int batch_size = 1;
  int dim = 128;
  auto graphsage = train(adj, nodes, edges, batch_size, dim, 10, n_node);
  graphsage.save("unsupervised_cora", nodes, adj);
}

void run_blogcatalog() {
  std::string edge_path = "../data/blogCatalog/bc_adjlist.txt";

  AdjList adj;
  Nodes nodes;
  Edges edges;
  int n_node = 10312;
  int n_features = 256;

  std::cout << "loading adjs" << std::endl;
  load_edges(edge_path, &adj);
  std::cout << "loading edges" << std::endl;
  load_edges(edge_path, &edges);

  std::cout << "random features" << std::endl;
  random_features(edge_path, &nodes, n_features, n_node);
  int batch_size = 1;
  int dim = 128;

  auto graphsage = train(adj, nodes, edges, batch_size, dim, 10, n_node);
  graphsage.save("unsupervised_blog", nodes, adj);
}



int main() {
//  run_cora();
  run_blogcatalog();
  return 0;
}