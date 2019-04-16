#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import torch
import json
import random

from argparse import ArgumentParser

''' To evaluate the embeddings, we run a logistic regression 
after running the unsupervised training.
'''

def run_logistic_regression(train_embeddings, train_targets, test_embeddings, test_targets):
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import precision_score
    from sklearn.metrics import f1_score

    lr = SGDClassifier(loss='log', max_iter=200)
    lr.fit(train_embeddings, train_targets)

    def precision(output, target):
        output = np.array(output)
        target = np.array(target)
        rights = sum(output == target)
        return rights * 1.0 / len(output)

    
    train_output = lr.predict(train_embeddings)
    print("train precision", precision(train_output, train_targets), 
            "f1_score", f1_score(train_output, train_targets, average='micro'))

    output = lr.predict(test_embeddings)
    print("test precision", precision(output, test_targets), 
            "f1_score", f1_score(output, test_targets, average='micro'))


def load_labels(path):
    return json.loads(open(path).read())


def load_embeddings(path):
    embeddings = np.fromfile(path + '/' + 'embedding.b', dtype=np.float32)
    id_map = np.fromfile(path + '/' + 'id_map.b', dtype=np.int32)
    embeddings = embeddings.reshape((id_map.shape[0], -1))
    return embeddings, id_map

def load_raw_features(path):
    examples = []
    max_id = 0
    n_node = 0
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            node_id = int(parts[0])
            label   = int(parts[1])
            feats   = [int(x) for x in parts[2:]]
            max_id = max(max_id, max(feats))
            examples.append((node_id, label, feats))
            n_node += 1

    feats = np.zeros([n_node, max_id+1], dtype=np.float32)
    labels = np.empty([n_node], dtype=np.int32)
    for idx, example in enumerate(examples):
        node_id, label, feat = example
        f = feats[idx]
        for i in feat:
            feats[idx][i] = float(1.0)
        labels[idx] = label
        
    return feats, labels


if __name__ == '__main__':
    parser = ArgumentParser("Run evaluation on Cora data.")
    parser.add_argument('-d', '--embed_dir', type=str, default='')
    parser.add_argument('-l', '--label_path', type=str, default='')
    parser.add_argument('-f', '--features', type=int, default=0)

    args = parser.parse_args()
    
    if args.features == 1:
        embeddings, labels = load_raw_features(args.embed_dir)
    else:
        embeddings, id_map = load_embeddings(args.embed_dir)
        labels = load_labels(args.label_path)
        labels = [labels[str(node)] for node in id_map]
        labels = np.array(labels)

    indices = [x for x in range(0, len(labels))]
    random.shuffle(indices)

    train_index = indices[0:2000]
    test_index  = indices[2000:]
    train_embeddings = embeddings[train_index]
    train_labels     = labels[train_index]

    test_embeddings  = embeddings[test_index]
    test_labels      = labels[test_index]

    
    run_logistic_regression(train_embeddings, train_labels, test_embeddings, test_labels)
