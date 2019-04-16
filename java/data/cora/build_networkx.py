#!/usr/bin/env python

import networkx as nx
from networkx.readwrite import json_graph
import json
import random

graph = nx.DiGraph()

infile = 'cora.adjs'
with open(infile) as f:
    for line in f:
        parts = line.strip().split()
        node = int(parts[0])
        adjs = [int(x) for x in parts[1:]]
        for dst in adjs:
            graph.add_edge(node, dst, train_removed=False, test_removed=False)

    
    nodes = list(graph.nodes())
    random.shuffle(nodes)
    ## train
    train = nodes[0:1500]
    valid = nodes[1500:2000]
    test  = nodes[2000:]

    for node in train:
        graph.nodes[node]['test'] = False
        graph.nodes[node]['val']  = False

    for node in valid:
        graph.nodes[node]['test'] = False
        graph.nodes[node]['val']  = True

    for node in test:
        graph.nodes[node]['test'] = True
        graph.nodes[node]['val']  = False

    
data = json_graph.node_link_data(graph)
data = json.dumps(data)
with open('cora-G.json', 'w') as f:
    f.write(data)

print(graph.out_degree(0))
print(list(graph.neighbors(0)))
print(graph.out_degree(387))
print(list(graph.neighbors(387)))

print(graph.number_of_edges())



## id_map
id_map = dict()
for n in list(graph.nodes()):
    id_map[n] = n

with open('cora-id_map.json', 'w') as f:
    f.write(json.dumps(id_map))


## class_map
class_map = dict()
with open('cora.content.id') as f:
    for line in f:
        parts = line.strip().split()
        node = int(parts[0])
        label = int(parts[1])
        class_map[node] = label

with open('cora-class_map.json', 'w') as f:
    f.write(json.dumps(class_map))
