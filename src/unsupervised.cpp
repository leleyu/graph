//
// Created by leleyu on 19-1-10.
//

#include <graph/graphsage.h>
#include <graph/model.h>


using namespace torch;
using namespace torch::data;
using namespace torch::data::datasets;
using namespace torch::data::samplers;
using namespace torch::optim;
using namespace graph::dataset;
using namespace graph::nn;


double calculate_precision(torch::Tensor output, torch::Tensor target) {
  auto y = output.argmax(1).to(TensorOptions().dtype(kInt64));
  int n_right = torch::eq(y, target).sum().item().toInt();
  int n_total = output.size(0);
  return n_right * 1.0 / n_total;
}

// calculate loss
Tensor calculate_loss(Tensor u, Tensor v, Tensor neg) {
  // u [batch_size, dim]
  // v [batch_size, dim]
  // neg [batch_size, neg_num, dim]
  u = u.view({u.size(0), u.size(1), 1});
  v = v.view({v.size(0), 1, v.size(1)});
  auto pos_loss = -torch::log(torch::sigmoid(v.matmul(u)));
  auto neg_loss = -torch::log(torch::sigmoid(neg.matmul(u)));

  auto size = pos_loss.size(0) + neg_loss.size(0) * neg_loss.size(1);
  return (pos_loss.sum() + neg_loss.sum()) / size;
}

UnSupervisedGraphsage train_unsupervised_graphsage(const AdjList& adj,
    const Nodes& nodes, const Edges& edges,
    const int batch_size,
    const int hidden_dim,
    const int n_negative = 3) {

  auto dataset = EdgeDataset(edges.srcs, edges.dsts);
  auto sampler = RandomSampler(dataset.size().value());
  auto options = torch::data::DataLoaderOptions(batch_size);
  auto loader  = torch::data::make_data_loader(dataset, options, sampler);

  int n_feature = static_cast<int>(nodes.features.size(1));
  UnSupervisedGraphsage net(n_feature, hidden_dim);
  SGD optim(net.parameters(), 0.01);

  for (int epoch = 0; epoch < 10; epoch ++) {

    int batch_cnt = 0;
    for (auto batch : *loader) {
      auto start = std::chrono::system_clock::now();
      optim.zero_grad();

      auto srcs = batch[0][0];
      auto dsts = batch[0][1];
      
      int batch_size = srcs.size(0);
      auto negs = negative_sampling(adj, srcs, n_negative, 2708).view({batch_size * n_negative});

      auto src_output = net.forward(srcs, nodes.features, nodes.node_to_index, adj);
      auto dst_output = net.forward(dsts, nodes.features, nodes.node_to_index, adj);
      auto neg_output = net.forward(negs, nodes.features, nodes.node_to_index, adj).view({batch_size, n_negative, hidden_dim});

      auto loss = calculate_loss(src_output, dst_output, neg_output);
      loss.backward();
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
  SGD optim(net.parameters(), 0.1);

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

  for (int epoch = 0; epoch < 10; epoch ++) {
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
      auto p = calculate_precision(output, batch_t);

      loss.backward();
      optim.step();
      std::cout << loss.item() << std::endl;
    }

    // validate
    auto validate_output = net.forward(validate_f);
    auto validate_p = calculate_precision(validate_output, validate_t);
    std::cout << " epoch=" << epoch
              << " val_p=" << validate_p
              << std::endl;
  }

}

void classify(int n_feature, int n_class, int n_node) {
  Tensor features, targets;
  torch::load(features, "features.pt");
  torch::load(targets, "targets.pt");

  std::cout << features.sizes() << std::endl;
  std::cout << targets.sizes()  << std::endl;

  features.set_requires_grad(false);

  std::cout << features[0] << std::endl;

  train_mlp(features, targets, n_feature, n_node, n_class, 128);
}

void save_embedding(UnSupervisedGraphsage net, int n_feature, int n_class, int n_node,
                    const Nodes& nodes, const AdjList& adj) {
  std::vector<int> node_ids;
  node_ids.resize(n_node);
  for (int i = 0; i < n_node; i ++) node_ids[i] = i;

  auto features = net.forward(torch::from_blob(node_ids.data(), {n_node}, TensorOptions().dtype(kInt32)),
                              nodes.features, nodes.node_to_index, adj);
  auto targets = torch::empty({n_node}, TensorOptions().dtype(kInt64));
  for (int i = 0; i < n_node; i ++) {
    targets[i] = nodes.labels[nodes.node_to_index.find(node_ids[i])->second];
  }

  torch::save(features, "features.pt");
  torch::save(targets, "targets.pt");
}

int main() {
  std::string edge_path = "../data/cora/cora.adjs";
  std::string node_path = "../data/cora/cora.content.id";
  AdjList adj;
  Nodes nodes;
  Edges edges;
  int n_nodes = 2708;
  int n_features = 1433;
  load_edges(edge_path, &adj);
  load_edges(edge_path, &edges);
  load_features(node_path, &nodes, n_features, n_nodes);
  int dim = 128;
//  auto graphsage = train_unsupervised_graphsage(adj, nodes, edges, 128, dim, 3);
  classify(dim, 7, n_nodes);
//  save_embedding(graphsage, dim, 7, n_nodes, nodes, adj);
  return 0;
}