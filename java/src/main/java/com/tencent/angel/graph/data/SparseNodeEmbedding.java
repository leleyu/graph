package com.tencent.angel.graph.data;

public class SparseNodeEmbedding {
  private float[] embeddings;
  private int dim; // dimension for one embedding vector
  private int size; // number of embedding vectors

  public SparseNodeEmbedding(int size, int dim) {
    this(size, dim, new float[size * dim]);
  }

  public SparseNodeEmbedding(int size, int dim, float[] embeddings) {
    this.dim = dim;
    this.size = size;
    this.embeddings = embeddings;
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

  public void aggregate(int node, SubGraph graph, int index, float[] destination) {
    int[] nodes = graph.getNodes();
    int[] neighbors = graph.getNeighbors();
    int length = nodes[node + 1] - nodes[node];

    // sum for neighbors
    for (int j = 0; j < length; j++) {
      int neighbor = neighbors[nodes[node] + j];
      for (int k = 0; k < dim; k++)
        destination[index * dim + k] += embeddings[neighbor * dim + k];
    }

    // average
    for (int k = 0; k < dim; k++)
      destination[index * dim + k] /= length;
  }

  public SparseNodeEmbedding subEmbeddings(NodeArray nodes) {
    int[] ns = nodes.getNodes();
    int len = ns.length;
    float[] values = new float[len * dim];
    for (int i = 0; i < len; i += dim) {
      int node = ns[i];
      for (int j = 0; j < dim; j++) {
        values[i + j] = embeddings[node * dim + j];
      }
    }
    return new SparseNodeEmbedding(len, dim, values);
  }


}
