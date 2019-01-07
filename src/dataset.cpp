//
// Created by leleyu on 2018-12-25.
//


#include <torch/torch.h>
#include "graph/dataset.h"
#include <string.h>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <vector>


namespace graph {
namespace dataset {

torch::data::Example<> parseLibSVM(std::string str) {

  std::vector<std::string> strings;
  std::istringstream f(str);
  string s, t;

  // label
  getline(f, s, ' ');
  float l = std::stof(s);
  if (l < 0.0) l = 0.0;
  auto label = torch::tensor(l);

  // kvs
  std::vector<int> keys;
  std::vector<float> vals;

  while (getline(f, s, ' ')) {
    std::istringstream kv(s);
    getline(kv, t, ':');
    keys.push_back(std::stoi(t));
    getline(kv, t, ':');
    vals.push_back(std::stof(t));
  }

  int size = static_cast<int>(keys.size());
  // indices
  auto indices = at::zeros({2, size}, at::TensorOptions().dtype(at::kLong));
  for (size_t i = 0; i < size; i++) indices[0][i] = 0;
  for (size_t i = 0; i < size; i++) indices[1][i] = keys[i];
  // values
  auto values = at::zeros(size, at::TensorOptions().dtype(at::kFloat));
  for (size_t i = 0; i < size; i++) values[i] = vals[i];

  auto data = torch::sparse_coo_tensor(indices, values, {1, 124});
  return {data, label};
}

torch::data::Example<> parseLibSVM(std::string str, size_t n_dim) {
  auto data = torch::zeros(n_dim, torch::TensorOptions().dtype(torch::kFloat32));

  std::istringstream f(str);
  std::string s, t;

  // label
  getline(f, s, ' ');
  float l = std::stof(s);
  if (l < 0.0) l = 0.0;
  auto label = torch::tensor(l);

  // kvs
  while (getline(f, s, ' ')) {
    std::istringstream kv(s);
    getline(kv, t, ':');
    int k = std::stoi(t);
    getline(kv, t, ':');
    float v = std::stof(t);
    data[k] = v;
  }

  return {data, label};
}

void load_edges(const std::string& path, AdjList* adj) {
  std::ifstream in(path);
  std::string line, c;

  adj->starts.push_back(0);

  int cnt = 0;
  while (getline(in, line)) {
    std::istringstream is(line);

    // src_id
    getline(is, c, ' ');
    auto node_id = std::stoi(c);

    adj->src_to_index[node_id] = cnt;

    while (getline(is, c, ' ')) {
      auto dst = std::stoi(c);
      adj->dsts.push_back(dst);
    }

    adj->starts.push_back(static_cast<int>(adj->dsts.size()));
    cnt ++;
  }
}

void load_features(const std::string& path, Nodes* nodes,
  int n_feature, int n_node) {
  std::ifstream in(path);
  std::string line, c;

  nodes->features = torch::empty({n_node, n_feature});
  nodes->labels   = torch::empty({n_node});
  nodes->features.set_requires_grad(false);

  int cnt = 0;
  while (getline(in, line)) {
    std::istringstream is(line);
    // node_id
    getline(is, c, ' ');
    auto node_id = std::stoi(c);
    nodes->node_to_index[node_id] = cnt;
    // label
    getline(is, c, ' ');
    nodes->labels[cnt] = std::stof(c);
    // features
    auto f = nodes->features[cnt];
    while (getline(is, c, ' ')) {
      auto index = std::stoi(c);
      f[index] = 1.0f;
    }

    cnt ++;
  }
}


} //namespace dataset
} //graph
