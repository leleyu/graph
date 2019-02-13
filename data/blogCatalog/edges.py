#!/usr/bin/env python

writer = open('bc.edge', 'w')
with open('bc_edgelist.txt') as f:
    for line in f:
        s, d = line.strip().split()
        writer.write('%s %s\n' % (str(s), str(d)))
