package com.tencent.angel.graph.data;

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;

import java.util.Random;

public class SubGraph {

  // start index of nodes, assuming that the nodes index is encoded from 1. length = n + 1
  private int[] nodes;
  private int[] neighbors;
  private int maxNeighbor;
  public Int2IntOpenHashMap index;

  public SubGraph(int[] nodes, int[] neighbors, int maxNeighbor) {
    this.nodes = nodes;
    this.neighbors = neighbors;
    this.maxNeighbor = maxNeighbor;
  }

  public SubGraph(int[] nodes, int[] neighbors, int maxNeighbor, Int2IntOpenHashMap index) {
    this(nodes, neighbors, maxNeighbor);
    this.index = index;
  }


  public int[] getNodes() {
    return nodes;
  }

  public int[] getNeighbors() {
    return neighbors;
  }

  public int getMaxNeighbor() {
    return maxNeighbor;
  }

  public SubGraph() {}

  public void build(int N, int[] srcs, int[] dsts) {
    nodes = new int[N + 1];
    int[] dcnt = new int[N];
    for (int i = 0; i < srcs.length; i++)
      dcnt[srcs[i]] ++;

    nodes[0] = 0;
    neighbors = new int[dsts.length];

    // build start index
    for (int i = 0; i < N; i++)
      nodes[i + 1] = nodes[i] + dcnt[i];

    for (int i = 0; i < dsts.length; i++) {
      int src = srcs[i];
      int start = nodes[src];
      neighbors[start + --dcnt[src]] = dsts[i];
    }
  }

  public void addNodeToSubGraph(int node, Int2IntOpenHashMap index,
                                IntArrayList startIndex,
                                IntArrayList subNeighbors,
                                int numSample,
                                Random rand) {
    if (index.containsKey(node))
      return;

    index.put(node, index.size());

    // sample its neighbor
    int len = nodes[node + 1] - nodes[node];
    int currentSize = startIndex.getInt(startIndex.size() - 1);
    if (len < numSample) {
      startIndex.add(currentSize + len);
      for (int j = nodes[node]; j < nodes[node + 1]; j++) {
        int neighbor = neighbors[j];
        subNeighbors.add(neighbor);
      }
    } else {
      startIndex.add(currentSize + numSample);
      for (int j = 0; j < numSample; j++) {
        int neighbor = neighbors[nodes[node] + rand.nextInt(len)];
        subNeighbors.add(neighbor);
      }
    }
  }

  public SubGraph sample(int[] batch, int order, int numSample) {
    Random rand = new Random(System.currentTimeMillis());

    Int2IntOpenHashMap index = new Int2IntOpenHashMap();
    IntArrayList startIndex = new IntArrayList();
    IntArrayList subNeighbors = new IntArrayList();

    startIndex.add(0);

    IntOpenHashSet roots = new IntOpenHashSet();
    for (int i = 0; i < batch.length; i++)
      roots.add(batch[i]);

    int start = 0;

    for (int k = 0; k < order; k++) {
      if (k == 0) {
        // use batch as expanding nodes
        for (int j = 0; j < batch.length; j++) {
          int node = batch[j];
          // add node
          addNodeToSubGraph(node, index, startIndex, subNeighbors, numSample, rand);
        }
      } else {
        for (int j = start; j < subNeighbors.size(); j++) {
          int node = subNeighbors.getInt(j);
          addNodeToSubGraph(node, index, startIndex, subNeighbors, numSample, rand);
        }

        start = subNeighbors.size();
      }
    }

    for (int i = start; i < subNeighbors.size(); i++) {
      if (!index.containsKey(subNeighbors.getInt(i)))
        index.put(subNeighbors.getInt(i), index.size());
    }

    return new SubGraph(startIndex.toIntArray(), subNeighbors.toIntArray(), numSample, index);
  }
}
