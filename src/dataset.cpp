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

void load_edges(const std::string& path, Edges* edges) {
  std::ifstream in(path);
  std::string line, c;

  std::vector<int>& srcs = edges->srcs;
  std::vector<int>& dsts = edges->dsts;

  while (getline(in, line)) {
    std::istringstream is(line);

    // src_id
    getline(is, c, ' ');
    auto src_id = std::stoi(c);

    while (getline(is, c, ' ')) {
      auto dst_id = std::stoi(c);
      srcs.push_back(src_id);
      dsts.push_back(dst_id);
    }
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

// Generate random walk with Adj, for each node, we generate `n_walks` walks
// with each walk of `n_length`. The generated walks is stored in a tensor with
// dim [n_walks*n_nodes, n_length]
torch::Tensor random_walk(const AdjList& adj, int n_walks, int n_length) {

  int n_nodes = adj.src_to_index.size();
  auto walks = torch::zeros({n_nodes*n_walks, n_length}, torch::TensorOptions().dtype(torch::kInt32));
  int walk_idx = 0;
  srand(time(NULL));

  for (auto it = adj.src_to_index.begin(); it != adj.src_to_index.end(); it ++) {
    int node = it->first;

    for (int i = 0; i < n_walks; i ++) {
      walks[walk_idx] = -1;
      auto current_walk = walks[walk_idx];
      auto accessor = current_walk.accessor<int, 1>();
      int current = node;
      for (int j = 0; j < n_length; j++) {
        accessor[j] = current;
        // last one
        if (j == n_length - 1)
          continue;
        // find next one
        auto n_it = adj.src_to_index.find(current);
        if (n_it != adj.src_to_index.end()) {
          int index = n_it->second;
          int n_nb = adj.starts[index + 1] - adj.starts[index];
          int next = rand() % n_nb;
          current = adj.dsts[adj.starts[index] + next];
        } else
          break;
      }
      walk_idx++;
    }
  }

  return walks;
}

// Generate negative sampling for nodes in `nodes`, for each node, we generate
// `n_neg` negative samples that are not similar with this node.
// The return tensor contains size(nodes)*n_neg negatives samples.
torch::Tensor negative_sampling(const AdjList& adj,
    const torch::Tensor& nodes, int n_neg, int n_nodes) {
  
  auto negs = torch::zeros({nodes.size(0), n_neg}, torch::TensorOptions().dtype(torch::kInt32));
  auto accessor = nodes.accessor<int, 1>();
  srand(time(NULL));
  
  std::set<int> nbs;
  for (int i = 0; i < nodes.size(0); i ++) {
    int node = accessor[i];
    nbs.clear();
    
    auto it = adj.src_to_index.find(node);
    if (it != adj.src_to_index.end()) {
      int index = it->second;
      for (int j = adj.starts[index]; j < adj.starts[index+1]; j ++)
        nbs.insert(adj.dsts[j]);
    }
    
    auto current_neg = negs[i];
    auto f = current_neg.accessor<int, 1>();
    for (int j = 0; j < n_neg; j ++) {
      int n;
      do {
        n = rand() % n_nodes;
      } while ((n == node) || (nbs.find(n) != nbs.end()));
      f[j] = n;
    }
  }

  return negs;
}

} //namespace dataset
} //graph
