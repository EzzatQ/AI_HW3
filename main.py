from matplotlib import pyplot as plt

from ID3 import ID3
from ID3 import experiment
import csv
import numpy as np
from KFoldCrossValidation import KFoldCrossValidation
from PersonalizedID3 import PersonalizedID3


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
    r = csv.reader(open('train.csv'))
    lines = list(r)
    data = np.array(lines)
    # x = [0,    0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1,    1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2,    2.0,  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.0,  3.1,  3.2,  3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4]
    # y = [27.0, 23.8, 23.8, 24.2, 24.0, 25.8, 25.8, 25.8, 25.8, 23.6, 21.8, 21.8, 21.8, 21.8, 20.2, 20.2, 20.2, 20.2, 25.6, 25.6, 25.6, 25.6, 24.0, 24.0, 24.0, 19.2, 19.2, 19.2, 19.2, 19.2, 19.2, 19.2, 18.0, 18.0, 18.0, 18.0, 17.8, 21.4, 21.4, 21.4, 21.4, 21.4]
    # plt.plot(x, y, "-o")
    # plt.xlabel('Hyper Parameter')
    # plt.ylabel('Loss')
    # plt.savefig(f"hyperparameter tuning.png")
    # plt.show()

    k=5
    kf = KFoldCrossValidation(k, PersonalizedID3, data, shuffle=True, filename="gini",
                             eval_type="loss", entropy_param=3.5, split_by="gini")
    print(f"gini, loss: {kf.getError()}")
    kf = KFoldCrossValidation(k, PersonalizedID3, data, shuffle=True, filename="gini",
                             eval_type="accuracy", entropy_param=3.5, split_by="gini")
    print(f"gini, accuracy: {kf.getError()}")

    # PersonalizedID3.hyperParamTuning(data)


if __name__ == '__main__':
    main()
