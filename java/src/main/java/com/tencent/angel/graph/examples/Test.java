package com.tencent.angel.graph.examples;

import it.unimi.dsi.fastutil.ints.IntArrayList;

public class Test {

  public static void main(String[] argv) {
    IntArrayList ints = new IntArrayList();
    ints.add(1);
    ints.add(3);
    ints.add(0);

    int[] array = ints.toIntArray();
    for (int i = 0; i < array.length; i++)
      System.out.println(array[i]);
  }

}
