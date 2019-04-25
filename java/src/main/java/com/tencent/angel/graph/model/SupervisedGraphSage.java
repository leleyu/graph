package com.tencent.angel.graph.model;

import com.tencent.angel.graph.data.NodeArray;
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
    ptr = initNetwork(inputDim, this.numClass, outputDims);
    keys = keys();
  }

  public Map<String, float[]> backward(SparseNodeEmbedding inputEmbeddings,
                                       NodeArray nodes,
                                       SubGraph subGraph,
                                       NodeArray targets) {
    float[][] gradients = backward(ptr, inputEmbeddings.getEmbeddings(),
       nodes.getNodes(),
       subGraph.getMaxNumNgb(),
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
                                    float[] selfEmbeddings,// embedding vectors
                                    int[] batch, // batch of nodes
                                    // graph structure
                                    int maxNeibor,
                                    int[] nodes, int[] neibors,
                                    int[] targets,
                                    // return keys
                                    String[] keys);

  private native long initNetwork(int inputDim, int numClass, int[] outputDims);
}
