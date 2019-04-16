#!/usr/bin/env python

''' count #edges, #nodes, max_node_id '''

nodes = {}
with open('bc_edgelist.txt') as f:
    for line in f:
        src, dst = line.strip().split()
        nodes[int(src)] = 1
        nodes[int(dst)] = 1

nodes = list(nodes.keys())
number_of_nodes = len(nodes)
max_node_id = max(nodes)

print('number_of_nodes', number_of_nodes)
print('max_node_id', max_node_id)

labels = {}
with open('bc_labels.txt') as f:
    for line in f:
        p = line.strip().split()
        ls = p[1:]
        for l in ls:
            labels[int(l)] = 1

labels = list(labels.keys())
number_labels = len(labels)
max_label = max(labels)
print('number of labels', number_labels)
print('max_label', max_label)


