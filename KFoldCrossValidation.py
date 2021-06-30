import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

eval_options = ["accuracy", "loss"]

class KFoldCrossValidation:

    def __init__(self, k, classifier, data_set: np.ndarray, shuffle=False, filename='KFCV',
                 early_pruning=False, tune_pruning_param=False, early_pruning_param=5,
                 early_pruning_tuning_options=None, eval_type="loss", hyperparam=1):
        self.k = k
        self.classifier = classifier
        self.data_set = data_set
        self.filename = filename
        self.kf = KFold(n_splits=5, shuffle=shuffle, random_state=318931870)
        self.early_pruning = early_pruning
        self.tune_pruning_param = tune_pruning_param
        self.early_pruning_param = early_pruning_param
        self.eval_type = eval_type
        self.hyperparam = hyperparam
        if eval_type not in eval_options:
            raise Exception("Invalid evaluation type")
        self.early_pruning_tuning_options = [5, 6, 7, 8, 9, 10] if early_pruning_tuning_options is None else early_pruning_tuning_options
        if tune_pruning_param:
            self.tuneEarlyPruning()

    def getError(self, pruning_param=None):
        if pruning_param is None:
            pruning_param = self.early_pruning_param
        total_accuracy = 0
        i = 0
        for train_idx, test_idx in self.kf.split(self.data_set):
            clf = self.classifier(filename=f"{self.filename}_{i}th_fold_classificationTree",
                                  early_pruning=self.early_pruning, early_pruning_param=pruning_param, hyperparam=self.hyperparam)
            print(f'evaluating fold {i + 1}')
            test_set = self.data_set[test_idx][:, 1:]
            test_real_labels = (self.data_set[test_idx][:, :1]).flatten()
            train_set = self.data_set[train_idx]
            clf.fit_predict(train_set, test_set)
            if self.eval_type == "accuracy":
                acc = clf.eval_accuracy(test_real_labels)
                print(f"accuracy: {acc}")
            else:
                acc = clf.loss(test_real_labels)
                print(f"loss: {acc}")
            total_accuracy += acc
            i += 1
        return total_accuracy/self.k

    def tuneEarlyPruning(self):
        best_param = None
        best_accuracy = None
        filename = self.filename
        pruning_params = []
        pruning_acc = []
        for i, param in enumerate(self.early_pruning_tuning_options):
            self.filename = filename + f"_with_early_pruning_param_{param}"
            acc = self.getError(param)
            pruning_params.append(param)
            pruning_acc.append(acc)
            if best_accuracy is None:
                best_accuracy = acc
            else:
                if best_accuracy < acc:
                    best_accuracy, best_param = acc, param
            print(f"accuracy with early pruning param {param}: {acc}")
            plt.plot(pruning_params, pruning_acc, "-o")
            plt.xlabel('Pruning Parameter')
            plt.ylabel('Accuracy')
            plt.show()
            plt.savefig(self.filename + "_plot")
            self.early_pruning_param = best_param
            self.filename = filename




