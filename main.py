from matplotlib import pyplot as plt
from ID3 import ID3
import csv
import numpy as np
import math
from KFoldCrossValidation import KFoldCrossValidation as KFCV
from numpy import genfromtxt
from sklearn import tree
from graphviz import Digraph


def processInput(csv_file):
    r = csv.reader(open(csv_file))
    lines = list(r)
    for line in lines:
        if line[0] == 'M':
            line[0] = 1
        else:
            line[0] = 0
    return np.array(lines, float)


def main():

    data = processInput('train.csv')
    mini_data = data[40:56, :6]
    clf = ID3("test")
    clf.fit_predict(data, data[:, 1:])
    acc = clf.eval_accuracy(data[:,:1].flatten())
    print(f"predict acc on train data is {acc}")
    clf = tree.DecisionTreeClassifier(random_state=0, criterion="entropy")

    k_cross = KFCV(1, ID3, data, shuffle=True)
    k = 5
    k_cross = KFCV(k, ID3, data, shuffle=True)
    print(f"accuracy with {k}-folds is {k_cross.getError()} %")


if __name__ == '__main__':
    main()