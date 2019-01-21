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

typedef int32_t NodeId;
typedef std::unordered_map<NodeId, int32_t> IndexLookupTable;
typedef std::vector<int32_t> IndexArray;
typedef std::vector<NodeId> NodeArray;
typedef std::set<NodeId> NodeSet;

struct AdjList {
  IndexArray starts;
  NodeArray neighbors;
  IndexLookupTable lookup_table;
};

struct NodeInfo {
  torch::Tensor features;
  torch::Tensor labels;
  IndexLookupTable lookup_table;
};

struct EdgeInfo {
  NodeArray srcs;
  NodeArray dsts;
};


// Undirected Graph
class UndirectedGraph {
public:
  // Methods for graph creatation
  void AddEdge(NodeId src, NodeId dst) {
    edges_.srcs.push_back(src);
    edges_.dsts.push_back(dst);
    nodes_.insert(src);
    nodes_.insert(dst);
  }

  void AddNode(NodeId node) {
    nodes_.insert(node);
  }

  // Build Adjacent List for this graph
  void Build();

  // Methods for graph accessing
  size_t GetDegree(NodeId node);

  size_t GetNumNode() {
    return nodes_.size();
  }

  size_t GetNumEdge() {
    return edges_.srcs.size();
  }

  bool HasNode(NodeId node) {
    return nodes_.find(node) != nodes_.end();
  }

  bool HasEdge(NodeId src, NodeId dst);

  NodeId* GetNeiborPtr(NodeId node);

private:
  AdjList adj_;
  EdgeInfo edges_;
  NodeSet nodes_;
};

typedef UndirectedGraph Graph;
} // namespace graph

#endif //GRAPH_GRAPH_H
