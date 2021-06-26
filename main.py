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
    r = csv.reader(open('test.csv'))
    lines = list(r)
    test_set = np.array(lines, float)
    mini_data = data[40:56, :6]
    clf = tree.DecisionTreeClassifier(random_state=0, criterion="entropy")
    X = test_set[:, 1:]
    y = test_set[:, 0]
    clf.fit(X, y)
    fig = plt.figure(figsize=(25, 20))
    tree.plot_tree(clf, class_names=['Sick', 'Healthy'], filled=True)
    fig.savefig("decistion_tree_entropy.png")
    k = 5
    k_cross = KFCV(k, ID3, data, shuffle=True)
    print(f"accuracy with {k}-folds is {k_cross.getError()} %")


if __name__ == '__main__':
    main()