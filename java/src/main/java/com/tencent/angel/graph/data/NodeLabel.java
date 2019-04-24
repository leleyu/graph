package com.tencent.angel.graph.data;

public class NodeLabel {
  private int[] labels;

  public NodeLabel(int[] labels) {
    this.labels = labels;
  }

  public int getLabel(int node) {
    return labels[node];
  }

  public int[] getLabels(int[] nodes) {
    int[] results = new int[nodes.length];
    for (int i = 0; i < results.length; i++)
      results[i] = labels[nodes[i]];
    return results;
  }
}
