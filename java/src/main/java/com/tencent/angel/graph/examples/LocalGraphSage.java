package com.tencent.angel.graph.examples;

import com.tencent.angel.graph.data.NodeArray;
import com.tencent.angel.graph.data.NodeLabel;
import com.tencent.angel.graph.data.SparseNodeEmbedding;
import com.tencent.angel.graph.data.SubGraph;
import com.tencent.angel.graph.model.SupervisedGraphSage;
import it.unimi.dsi.fastutil.ints.IntArrayList;

import java.io.BufferedReader;
import java.io.File;

import java.io.FileReader;
import java.io.IOException;

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

  public SparseNodeEmbedding fetchSelfEmbeddings(NodeArray batch) {
    int dim = inputEmbeddings.getDim();
    int[] nodes = batch.getNodes();
    SparseNodeEmbedding selfEmbedding = new SparseNodeEmbedding(nodes.length, dim);
    for (int i = 0; i < nodes.length; i++)
      inputEmbeddings.copy(nodes[i], i, selfEmbedding.getEmbeddings());

    return selfEmbedding;
  }

  public SparseNodeEmbedding fetchNeiborEmbeddings(NodeArray batch) {
    int dim = inputEmbeddings.getDim();
    int[] nodes = batch.getNodes();
    SparseNodeEmbedding neiborEmbedding = new SparseNodeEmbedding(nodes.length, dim);

    return null;
  }

  public void train(int[] trainNodes, int[] testNodes, SupervisedGraphSage model) {

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
    System.out.println("here");
    for (int i = 0; i < keys.length; i++)
      System.out.println(keys[i]);

    model.destory();

  }
}
