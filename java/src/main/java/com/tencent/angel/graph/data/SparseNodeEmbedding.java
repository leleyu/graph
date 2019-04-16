package com.tencent.angel.graph.data;

public class SparseNodeEmbedding {
  private float[] embeddings;
  private int dim; // dimension for one embedding vector
  private int size; // number of embedding vectors

  public SparseNodeEmbedding(int size, int dim) {
    embeddings = new float[size * dim];
    this.dim = dim;
    this.size = size;
  }

  public void set(int index, int offset, float value) {
    embeddings[index * dim + offset] = value;
  }

  public float[] getEmbeddings() {
    return embeddings;
  }

  public int getSize() {
    return size;
  }

  public void setSize(int size) {
    this.size = size;
  }

  public int getDim() {
    return dim;
  }

  public void copy(int index, int start, float[] destination) {
    System.arraycopy(embeddings, index * dim, destination, start * dim, dim);
  }

  public void aggregate(int node, SubGraph graph, int numSamples, int index, float[] destination) {
    int[] nodes = graph.getNodes();
    int length = nodes[node + 1] - nodes[node];

  }


}
