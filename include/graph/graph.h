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
    data_ = torch::zeros({num_node, dim});
  }

  SparseNodeEmbedding(const NodeArray &nodes, const torch::Tensor &embedding) {
    data_ = embedding;
    for (size_t i = 0; i < nodes.size(); i++)
      table_[nodes[i]] = static_cast<int32_t>(i);
  }

  void RandomInit(const NodeArray &nodes) {
    assert(nodes.size() == data_.size(0));
    for (int32_t i = 0; i < nodes.size(); i++) {
      table_[nodes[i]] = i;
    }
    torch::nn::init::xavier_uniform_(data_);
  }

  void RandomInit(const NodeSet &nodes) {
    assert(nodes.size() == data_.size(0));
    int32_t idx = 0;
    for (auto node : nodes)
      table_[node] = idx++;
    torch::nn::init::xavier_uniform_(data_);
  }

  void Insert(NodeId node, torch::Tensor feature) {
    int32_t index = static_cast<int32_t>(table_.size());
    table_[node] = index;
    data_[index] = feature;
  }

  inline torch::Tensor Lookup(NodeId node) const {
    auto it = table_.find(node);
    if (it == table_.end())
      std::cout << "cannot find embedding for node " << node << std::endl;
    assert(it != table_.end());
    return data_[it->second];
  }

  torch::Tensor Lookup(const NodeArray &nodes) const {
    auto result = torch::empty({static_cast<int64_t>(nodes.size()),
                                data_.size(1)});
    for (size_t i = 0; i < nodes.size(); i++)
      result[i] = Lookup(nodes[i]);
    return result;
  }

  int64_t GetDim() const {
    return data_.size(1);
  }

 private:
  torch::Tensor data_;
  IndexLookupTable table_;
};


// Directed Graph
class DirectedGraph {
 public:
  explicit DirectedGraph(const SparseNodeEmbedding &embedding)
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

  const EdgeInfo &GetEdgeInfo() const {
    return edges_;
  }

  size_t GetNumEdge() const {
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

struct NodeDataset : torch::data::datasets::Dataset<NodeDataset, NodeId> {
  NodeDataset(const NodeArray &nodes, size_t size) : nodes(nodes), num(size) {}

  NodeId get(size_t index) override {
    return nodes[index];
  }

  torch::optional<size_t> size() const override {
    return num;
  }

  const NodeArray &nodes;
  size_t num;
};

struct EdgeDataset : torch::data::datasets::Dataset<EdgeDataset, NodeId> {
  EdgeDataset(const EdgeInfo& edges): edges(edges) {}

  NodeId get(size_t index) override {
    return NAN_NODE_ID;
  }

  std::vector<NodeId> get_batch(torch::ArrayRef<size_t> indices) override {
    int64_t size = static_cast<int64_t >(indices.size());
    std::vector<NodeId> batch;
    batch.resize(size * 2);
    for (size_t i = 0; i < size; i++) {
      size_t index = indices[i];
      batch[i] = edges.srcs[index];
      batch[i + size] = edges.dsts[index];
    }
    return batch;
  }

  torch::optional<size_t> size() const override {
    return edges.srcs.size();
  }

  const EdgeInfo& edges;
};

void LoadGraph(const std::string &path,
               Graph *graph);

void LoadSparseNodeEmbedding(const std::string &path,
                             SparseNodeEmbedding *embedding);

void LoadNodeLabels(const std::string &path,
                    NodeLabels *labels);

torch::Tensor LookupLabels(const NodeArray &nodes,
                           const NodeLabels &labels);
} // namespace graph

#endif //GRAPH_GRAPH_H
