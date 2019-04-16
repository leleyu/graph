package com.tencent.angel.graph.model;

import com.tencent.angel.graph.data.NodeArray;
import com.tencent.angel.graph.data.NodeIndex;
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

  public void destory() {
    destroyNetwork(ptr);
  }

  public float[] forward(SparseNodeEmbedding selfEmbeddings,
                         SparseNodeEmbedding neiborEmbeddings,
                         NodeArray nodes,
                         SubGraph subGraph,
                         NodeIndex index) {
    return forward(ptr, selfEmbeddings.getEmbeddings(), neiborEmbeddings.getEmbeddings(),
      nodes.getNodes(), subGraph.getMaxNeighbor(), subGraph.getNodes(), subGraph.getNeighbors());
  }

  private native float[] forward(long ptr, // model ptr
                                 float[] selfEmbeddings, float[] neiborEmbeddings, // embedding vectors
                                 int[] batch, // batch of nodes
                                 // graph structure
                                 int maxNeibor,
                                 int[] nodes, int[] neibors);


  private native void destroyNetwork(long ptr);
}
