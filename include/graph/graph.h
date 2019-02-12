//
// Created by leleyu on 19-1-21.
//

#ifndef GRAPH_GRAPH_H
#define GRAPH_GRAPH_H

#include <stdint.h>

#include <vector>
#include <unordered_map>

#include <torch/torch.h>

namespace graph {

#define NAN_NODE_ID -1

typedef int32_t NodeId;
typedef int32_t Label;
typedef std::unordered_map<NodeId, int32_t> IndexLookupTable;
typedef std::vector<int32_t> IndexArray;
typedef std::vector<NodeId> NodeArray;
typedef std::set<NodeId> NodeSet;
typedef std::unordered_map<NodeId, Label> NodeLabels;

struct AdjList {
  IndexArray starts;
  NodeArray neighbors;
  IndexLookupTable lookup_table;
};

struct EdgeInfo {
  NodeArray srcs;
  NodeArray dsts;
};

class SparseNodeEmbedding {
 public:
  SparseNodeEmbedding(int64_t num_node, int64_t dim) {
    data_ = torch::empty({num_node, dim});
  }

  SparseNodeEmbedding(const NodeArray &nodes, const torch::Tensor &embedding) {
    data_ = embedding;
    for (size_t i = 0; i < nodes.size(); i++)
      table_[nodes[i]] = static_cast<int32_t>(i);
  }

  void insert(NodeId node, torch::Tensor feature) {
    int32_t index = static_cast<int32_t>(table_.size());
    table_[node] = index;
    data_[index] = feature;
  }

  inline torch::Tensor lookup(NodeId node) const {
    return data_[table_.find(node)->second];
  }

  torch::Tensor lookup(const NodeArray &nodes) const {
    auto result = torch::empty({static_cast<int64_t>(nodes.size()),
                                data_.size(1)});
    for (size_t i = 0; i < nodes.size(); i++)
      result[i] = lookup(nodes[i]);
    return result;
  }

  int64_t GetDim() const {
    return data_.size(1);
  }

 private:
  torch::Tensor data_;
  IndexLookupTable table_;
};


// Undirected Graph
class DirectedGraph {
 public:
  DirectedGraph(const SparseNodeEmbedding &embedding)
      : input_embeddings_(embedding) {}

  inline void AddEdge(NodeId src, NodeId dst) {
    edges_.srcs.push_back(src);
    edges_.dsts.push_back(dst);
    nodes_.insert(src);
    nodes_.insert(dst);
  }

  inline void AddNode(NodeId node) {
    nodes_.insert(node);
  }

  // Build Adjacent List for this graph
  void Build();

  // Methods for graph accessing
  size_t GetOutDegree(NodeId node) const;

  const SparseNodeEmbedding &GetInputEmbedding() const {
    return input_embeddings_;
  }

  size_t GetNumNode() const {
    return nodes_.size();
  }

  const NodeSet &GetNodeSet() const {
    return nodes_;
  }

  size_t GetNumEdge() {
    return edges_.srcs.size();
  }

  bool HasNode(NodeId node) {
    return nodes_.find(node) != nodes_.end();
  }

  NodeId *GetOutNeighborPtr(NodeId node);

 private:
  AdjList adj_;
  EdgeInfo edges_;
  NodeSet nodes_;
  const SparseNodeEmbedding &input_embeddings_;

};

typedef DirectedGraph Graph;

class NodeDataset : torch::data::datasets::Dataset<NodeDataset, NodeId> {
 public:
  NodeDataset(const NodeArray &nodes, size_t size) : nodes(nodes), num(size) {}

  NodeId get(size_t index) override {
    return nodes[index];
  }

  torch::optional<size_t> size() const override {
    return num;
  }

 private:
  const NodeArray &nodes;
  size_t num;
};

void LoadGraph(const std::string &path, Graph *graph);

void LoadSparseNodeEmbedding(const std::string &path,
                             SparseNodeEmbedding *embedding);

void LoadNodeLabels(const std::string &path, NodeLabels* labels);
} // namespace graph

#endif //GRAPH_GRAPH_H
