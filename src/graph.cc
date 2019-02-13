//
// Created by leleyu on 19-1-21.
//

#include <graph/graph.h>

namespace graph {

void DirectedGraph::Build() {
  std::unordered_map<NodeId, int32_t> degree;
  size_t num_edge = edges_.srcs.size();

  for (size_t i = 0; i < num_edge; i++) {
    auto it = degree.find(edges_.srcs[i]);
    if (it != degree.end())
      degree[edges_.srcs[i]] = it->second + 1;
    else
      degree[edges_.srcs[i]] = 1;
  }

  size_t num_node = degree.size();
  adj_.starts.resize(num_node + 1);
  adj_.neighbors.resize(num_edge);
  adj_.starts[0] = 0;

  for (size_t i = 0; i < num_edge; i++) {
    NodeId src = edges_.srcs[i];
    auto it = adj_.lookup_table.find(src);
    int32_t start = 0;

    // start position of each src node
    if (it != adj_.lookup_table.end())
      start = adj_.starts[it->second];
    else {
      int32_t num = adj_.lookup_table.size();
      adj_.lookup_table[src] = num;
      adj_.starts[num + 1] = adj_.starts[num] + degree[src];
      start = adj_.starts[num];
    }

    // position of neighbor
    int32_t position = start + degree[src] - 1;
    adj_.neighbors[position] = edges_.dsts[i];
    degree[src] = degree[src] - 1;
  }
}


size_t DirectedGraph::GetOutDegree(NodeId node) const {
  auto it = adj_.lookup_table.find(node);
  if (it != adj_.lookup_table.end()) {
    int32_t index = it->second;
    return adj_.starts[index + 1] - adj_.starts[index];
  } else
    return 0;
}

NodeId *DirectedGraph::GetOutNeighborPtr(NodeId node) {
  auto it = adj_.lookup_table.find(node);
  if (it != adj_.lookup_table.end()) {
    int32_t index = it->second;
    return &adj_.neighbors[adj_.starts[index]];
  } else
    return nullptr;
}

void LoadGraph(const std::string &path, Graph *graph) {
  std::ifstream in(path);
  std::string line, c;

  while (getline(in, line)) {
    std::istringstream is(line);
    // src
    getline(is, c, ' ');
    NodeId src = std::stoi(c);
    // dst
    getline(is, c, ' ');
    NodeId dst = std::stoi(c);

    graph->AddEdge(src, dst);
  }
}

void LoadSparseNodeEmbedding(const std::string &path,
                             SparseNodeEmbedding *embedding) {
  std::ifstream in(path);
  std::string line, c;

  int64_t dim = embedding->GetDim();

  while (getline(in, line)) {
    std::istringstream is(line);
    // NodeId
    getline(is, c, ' ');
    NodeId node_id = std::stoi(c);

    auto f = torch::zeros({dim});
    // feature
    while (getline(is, c, ' ')) {
      auto index = std::stoi(c);
      f[index] = 1.0f;
    }

    embedding->insert(node_id, f);
  }
}

void LoadNodeLabels(const std::string &path,
                    NodeLabels *labels) {
  std::ifstream in(path);
  std::string line, c;

  while (getline(in, line)) {
    std::istringstream is(line);
    // NodeId
    getline(is, c, ' ');
    NodeId node_id = std::stoi(c);

    // label
    getline(is, c, ' ');
    Label label = std::stoi(c);

    labels->insert(std::make_pair(node_id, label));
  }
}

torch::Tensor LookupLabels(const NodeArray &nodes,
                           const NodeLabels &labels) {
  int64_t size = static_cast<int64_t >(nodes.size());
  auto tensor = torch::empty({size}, torch::TensorOptions().dtype(torch::kInt64));
  auto f = tensor.accessor<int64_t, 1>();
  for (size_t i = 0; i < size; i++)
    f[i] = labels.find(nodes[i])->second;
  return tensor;
}


}; // namespace graph

