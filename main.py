from ID3 import ID3
from ID3 import experiment
import csv
import numpy as np
from KFoldCrossValidation import KFoldCrossValidation
import PersonalizedID3


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
    # ID3.experiment(data)
    # mini_data = data[40:56, :6]
    # clf = ID3("test")
    # clf.fit_predict(data, data[:, 1:])
    # acc = clf.eval_accuracy(data[:,:1].flatten())
    # print(f"predict acc on train data is {acc}")
    # clf = tree.DecisionTreeClassifier(random_state=0, criterion="entropy")
    # print("without_normalization")
    # clf = ID3("without_normalization", early_pruning=True, early_pruning_param=12)
    # clf.fit_predict(data[:250, :], data[250:, 1:])
    # print(f"loss: {clf.loss(data[250:, :1])}")
    # print(f"acc: {clf.eval_accuracy(data[250:, :1])}")
    #
    # print("with_normalization")
    # clf = ID3("with_normalization", early_pruning=True, early_pruning_param=12, normalise=True)
    # clf.fit_predict(data[:250, :], data[250:, 1:])
    # print("with_normalization")
    # clf.predictions = np.zeros_like(clf.predictions)
    # print(f"loss: {clf.loss(data[250:, :1])}")
    # print(f"acc: {clf.eval_accuracy(data[250:, :1])}")
    # k_cross = KFCV(1, ID3, data, shuffle=True)
    # k = 5
    # k_cross = KFoldCrossValidation(k, ID3, data, shuffle=True, early_pruning=True, early_pruning_param=12,
    #                                filename="normalID3")
    # print(f"loss for ID3 {k_cross.getError()} %")
    # k_cross = KFoldCrossValidation(k, PersonalizedID3, data, shuffle=True, early_pruning=True, early_pruning_param=12,
    #                                filename="personalizedID3")
    # print(f"loss for PersonalizedID3 {k_cross.getError()} %")
    PersonalizedID3.hyperParamTuning(data)


if __name__ == '__main__':
    main()
