package com.tencent.angel.graph.data;


public class NodeLabel {
  private int[] labels;

  public NodeLabel(int[] labels) {
    this.labels = labels;
  }

  public int getLabel(int node) {
    return labels[node];
  }
}
