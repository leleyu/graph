package com.tencent.angel.graph.data;

import it.unimi.dsi.fastutil.ints.Int2IntMap;
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import it.unimi.dsi.fastutil.objects.ObjectIterator;

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

  public SparseNodeEmbedding subEmbeddings(SubGraph subGraph) {
    int size = subGraph.index.size();
    float[] values = new float[size * dim];
    ObjectIterator<Int2IntMap.Entry> it = subGraph.index.int2IntEntrySet().fastIterator();
    while (it.hasNext()) {
      Int2IntMap.Entry entry = it.next();
      int nodeIdx = entry.getIntValue();
      int node = entry.getIntKey();
      System.arraycopy(embeddings, node * dim, values, nodeIdx * dim, dim);
    }
    return new SparseNodeEmbedding(size, dim, values);
  }


}
