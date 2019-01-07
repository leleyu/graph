#!/usr/bin/env python

num_edges = 0
with open('cora.adjs') as fp:
  for line in fp:
    splits = line.strip().split()
    num_edges += len(splits) - 1

print(num_edges)

edges = set()
with open('cora.cites') as fp:
  for line in fp:
    u, v = line.strip().split()
    edges.add('%s_%s' % (v, u))

print(len(edges))