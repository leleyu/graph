package com.tencent.angel.graph.model;

import com.tencent.angel.graph.data.NodeArray;
import com.tencent.angel.graph.data.NodeIndex;
import com.tencent.angel.graph.data.SparseNodeEmbedding;
import com.tencent.angel.graph.data.SubGraph;

public class SupervisedGraphSage extends GraphSage {

  private int numClass;

  public SupervisedGraphSage(int numClass, int inputDim, int[] outputDims) {
    super(inputDim, outputDims);
    this.numClass = numClass;
    ptr = initNetwork(inputDim, numClass, outputDims);
    System.out.println(ptr);
  }


  public float[] forward(SparseNodeEmbedding selfEmbeddings,
                         SparseNodeEmbedding neiborEmbeddings,
                         NodeArray nodes,
                         SubGraph subGraph,
                         NodeIndex index) {
    return new float[0];
  }

  public float[] backward(SparseNodeEmbedding selfEmbeddings,
                          SparseNodeEmbedding neiborEmbeddings,
                          NodeArray nodes,
                          SubGraph subGraph,
                          NodeIndex index,
                          NodeArray targets) {
    return new float[0];
  }

  private native float[][] backward(long ptr, // model ptr
                                    float[] selfEmbeddings, float[] neiborEmbeddings, // embedding vectors
                                    int[] batch, // batch of nodes
                                    // graph structure
                                    int maxNeibor,
                                    int[] nodes, int[] neibors,
                                    int[] targets,
                                    // return keys
                                    String[] keys);

  private native long initNetwork(int inputDim, int numClass, int[] outputDims);
}
