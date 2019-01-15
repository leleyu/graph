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

    lr = SGDClassifier(loss='log', max_iter=1000)
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


if __name__ == '__main__':
    parser = ArgumentParser("Run evaluation on Cora data.")
    parser.add_argument("embed_dir")
    parser.add_argument("label_path")

    args = parser.parse_args()
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
