package com.tencent.angel.graph.model;

import com.tencent.angel.graph.data.NodeArray;
import com.tencent.angel.graph.data.SparseNodeEmbedding;
import com.tencent.angel.graph.data.SubGraph;

public abstract class GraphSage {
  protected int[] outputDims;
  protected int inputDim;
  protected long ptr;

  public GraphSage(int inputDim, int[] outputDims) {
    this.inputDim = inputDim;
    this.outputDims = outputDims;
  }

  public String[] keys() {
    return getKeys(ptr);
  }

  public void destory() {
    destroyNetwork(ptr);
  }

  public float[] forward(SparseNodeEmbedding inputEmbeddings,
                         NodeArray nodes,
                         SubGraph subGraph) {
    return forward(ptr, inputEmbeddings.getEmbeddings(),
      nodes.getNodes(), subGraph.getMaxNumNgb(),
       subGraph.getNodes(), subGraph.getNeighbors());
  }

  private native float[] forward(long ptr, // model ptr
                                 float[] selfEmbeddings, // embedding vectors
                                 int[] batch, // batch of nodes
                                 // graph structure
                                 int maxNeighbor,
                                 int[] nodes, int[] neighbors);


  private native void destroyNetwork(long ptr);
  private native String[] getKeys(long ptr);
}
