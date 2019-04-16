package com.tencent.angel.graph.data;

import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;

public class NodeIndex {

  private Int2IntOpenHashMap index;

  public NodeIndex() {
    index = new Int2IntOpenHashMap();
    index.defaultReturnValue(-1);
  }

  public void put(int node) {
    int size = index.size();
    index.put(node, size);
  }

  public int getIndex(int node) {
    return index.get(node);
  }
}
