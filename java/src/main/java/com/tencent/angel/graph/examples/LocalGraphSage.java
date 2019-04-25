package com.tencent.angel.graph.examples;

import com.tencent.angel.graph.data.NodeArray;
import com.tencent.angel.graph.data.NodeLabel;
import com.tencent.angel.graph.data.SparseNodeEmbedding;
import com.tencent.angel.graph.data.SubGraph;
import com.tencent.angel.graph.model.SupervisedGraphSage;
import it.unimi.dsi.fastutil.ints.Int2IntMap;
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntArrays;
import it.unimi.dsi.fastutil.objects.ObjectIterator;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class LocalGraphSage {

  static {
    System.loadLibrary("torch_graph");
  }

  private int N; // number of nodes
  private int F; // input embedding dimensions

  private int numSample = 5; // num samples

  private NodeLabel nodeLabel;
  private SubGraph graph;
  private SparseNodeEmbedding inputEmbeddings;

  public LocalGraphSage(int N, int F) {
    this.N = N;
    this.F = F;
  }

  public void loadGraph(String path) throws IOException {
    IntArrayList srcs = new IntArrayList();
    IntArrayList dsts = new IntArrayList();

    BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
    String line;
    while ((line = reader.readLine()) != null) {
      String[] parts = line.split(" ");
      int src = Integer.parseInt(parts[0]);
      int dst = Integer.parseInt(parts[1]);
      srcs.add(src);
      dsts.add(dst);
    }

    graph = new SubGraph();
    graph.build(N, srcs.toIntArray(), dsts.toIntArray());
  }

  public void loadInputNodeEmbeddings(String path) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
    String line;

    inputEmbeddings = new SparseNodeEmbedding(N, F);
    while ((line = reader.readLine()) != null) {
      String[] parts = line.split(" ");
      int node = Integer.parseInt(parts[0]);
      for (int i = 1; i < parts.length; i++) {
        int index = Integer.parseInt(parts[i]);
        inputEmbeddings.set(node, index, 1.0f);
      }
    }
  }

  public void loadNodeLabel(String path) throws IOException {
    int[] labels = new int[N];
    BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
    String line;

    while ((line = reader.readLine()) != null) {
      String[] parts = line.split(" ");
      int node = Integer.parseInt(parts[0]);
      int label = Integer.parseInt(parts[1]);
      labels[node] = label;
    }

    nodeLabel = new NodeLabel(labels);
  }

  public void train(int[] trainNodes, int[] testNodes,
                    int numEpoch, int batchSize,
                    SupervisedGraphSage model) {

    IntArrays.shuffle(trainNodes, new Random(System.currentTimeMillis()));
    int[] batch = new int[batchSize];
    int idx = 0;
    for (int epoch = 1; epoch <= numEpoch; epoch++) {
      for (int i = 0; i < trainNodes.length; i++) {
        batch[idx++] = trainNodes[i];
        if (idx >= batchSize) {
          int[] targets = nodeLabel.getLabels(batch);
          trainBatch(batch, targets, model);
          idx = 0;
        }
      }

      // some left nodes
    }

  }

  public double trainBatch(int[] trainNodes, int[] targets, SupervisedGraphSage model) {
    SubGraph subGraph = graph.sample(trainNodes, 3, numSample);
    SparseNodeEmbedding batchEmbeddings = inputEmbeddings.subEmbeddings(subGraph);
    subGraph.reindex();
    float[] output = model.forward(batchEmbeddings, new NodeArray(trainNodes), subGraph);

    return 0.0;
  }

  public static void main(String[] argv) throws IOException {
    String edgePath = "data/cora/cora.edge";
    String featurePath = "data/cora/cora.feature";
    String labelPath = "data/cora/cora.label";

    int N = 2708;
    int F = 1433;
    int numClass = 7;

    LocalGraphSage graphsage = new LocalGraphSage(N, F);
    graphsage.loadGraph(edgePath);
    graphsage.loadInputNodeEmbeddings(featurePath);
    graphsage.loadNodeLabel(labelPath);

    int[] outputDims = {10, 5};
    SupervisedGraphSage model = new SupervisedGraphSage(numClass, F, outputDims);
    String[] keys = model.keys();
    for (int i = 0; i < keys.length; i++)
      System.out.println(keys[i]);


    // split nodes
    IntArrayList trainNodes = new IntArrayList();
    IntArrayList testNodes = new IntArrayList();

    Random random = new Random(System.currentTimeMillis());
    for (int node = 0; node < N; node++) {
      if (random.nextFloat() < 0.5)
        trainNodes.add(node);
      else
        testNodes.add(node);
    }

    int numEpoch = 10;
    int batchSize = 32;

    graphsage.train(trainNodes.toIntArray(),
       testNodes.toIntArray(),
       numEpoch, batchSize,
       model);

    model.destory();

  }
}
