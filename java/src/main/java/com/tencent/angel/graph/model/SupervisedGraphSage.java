package com.tencent.angel.graph.model;

import com.tencent.angel.graph.data.NodeArray;
import com.tencent.angel.graph.data.NodeIndex;
import com.tencent.angel.graph.data.SparseNodeEmbedding;
import com.tencent.angel.graph.data.SubGraph;

import java.util.HashMap;
import java.util.Map;

public class SupervisedGraphSage extends GraphSage {

  private int numClass;
  private String[] keys;

  public SupervisedGraphSage(int numClass, int inputDim, int[] outputDims) {
    super(inputDim, outputDims);
    this.numClass = numClass;
    ptr = initNetwork(inputDim, numClass, outputDims);
    System.out.println(ptr);
    keys = keys();
  }


  public float[] forward(SparseNodeEmbedding selfEmbeddings,
                         SparseNodeEmbedding neiborEmbeddings,
                         NodeArray nodes,
                         SubGraph subGraph,
                         NodeIndex index) {
    return new float[0];
  }

//  public float[] backward(SparseNodeEmbedding selfEmbeddings,
//                          SparseNodeEmbedding neiborEmbeddings,
//                          NodeArray nodes,
//                          SubGraph subGraph,
//                          NodeIndex index,
//                          NodeArray targets) {
//    return new float[0];
//  }

  public Map<String, float[]> backward(SparseNodeEmbedding selfEmbeddings,
                                       SparseNodeEmbedding neighborEmbeddings,
                                       NodeArray nodes,
                                       SubGraph subGraph,
                                       NodeIndex index,
                                       NodeArray targets) {
    float[][] gradients = backward(ptr, selfEmbeddings.getEmbeddings(),
       neighborEmbeddings.getEmbeddings(),
       nodes.getNodes(),
       subGraph.getMaxNeighbor(),
       subGraph.getNodes(),
       subGraph.getNodes(),
       targets.getNodes(),
       keys);

    Map<String, float[]> results = new HashMap<String, float[]>();
    for (int i = 0; i < keys.length; i++) {
      results.put(keys[i], gradients[i]);
    }
    return results;
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
