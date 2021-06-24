from ID3 import ID3
import csv
import numpy as np
import math
from KFoldCrossValidation import KFoldCrossValidation as KFCV
from numpy import genfromtxt


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
    learner = ID3()
    k = 5
    k_cross = KFCV(k, learner, data)
    print(f"accuracy with {k}-folds is {k_cross.getError()} %")


if __name__ == '__main__':
    main()