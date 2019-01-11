//
// Created by leleyu on 19-1-11.
//


#include <graph/model.h>
#include <graph/dataset.h>

using namespace torch;
using namespace torch::optim;
using namespace torch::data;
using namespace torch::data::samplers;
using namespace graph::dataset;

double calculate_precision(torch::Tensor output, torch::Tensor target) {
  auto y = output.argmax(1).to(TensorOptions().dtype(kInt64));
  int n_right = torch::eq(y, target).sum().item().toInt();
  int n_total = output.size(0);
  return n_right * 1.0 / n_total;
}

void train_mlp(const Tensor& features,
    const Tensor& targets,
    const std::unordered_map<int, int>& node_to_index,
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
  auto batch_t = torch::empty({batch_size}, TensorOptions().dtype(kInt64));

  // validate data
  auto validate_f = torch::empty({n_val, n_feature});
  auto validate_t = torch::empty({n_val}, TensorOptions().dtype(kInt64));

  for (int i = 0; i < n_val; i ++) {
    int node = node_ids[n_node - 1 - i];
    int index = node_to_index.find(node)->second;
    validate_f[i] = features[index];
    validate_t[i] = targets[index];
  }

  for (int epoch = 0; epoch < 100; epoch ++) {

    for (auto batch: *loader) {
      optim.zero_grad();
      if (batch.size() != batch_size)
        continue;

      // build batch input
      for (int i = 0; i < batch_size; i ++) {
        int idx = node_to_index.find(batch[i])->second;
        batch_f[i] = features[idx];
        batch_t[i] = targets[idx];
      }

      auto output = net.forward(batch_f);
      auto loss = nll_loss(log_softmax(output, 1), batch_t);
      auto p = calculate_precision(output, batch_t);

      loss.backward();
      optim.step();
    }

    // validate
    auto validate_output = net.forward(validate_f);
    auto validate_p = calculate_precision(validate_output, validate_t);
    std::cout << " epoch=" << epoch
              << " val_p=" << validate_p
              << std::endl;
  }

}

int main() {
  std::string node_path = "../data/cora/cora.content.id";
  Nodes nodes;
  int n_node = 2708;
  int n_feature = 1433;
  load_features(node_path, &nodes, n_feature, n_node);
  train_mlp(nodes.features, nodes.labels, nodes.node_to_index, n_feature, n_node, 7, 128);
  return 0;
}
