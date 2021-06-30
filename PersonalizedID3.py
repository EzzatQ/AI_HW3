import numpy as np
from matplotlib import pyplot as plt

from KFoldCrossValidation import KFoldCrossValidation
from PersonalizedDecisionTreeClassifier import PersonalizedDecisionTree, DTCNode


class PersonalizedID3:

    def __init__(self, filename='PersonalizedID3', early_pruning=False, early_pruning_param=5, normalise=True, hyperparam=1):
        self.train_set: np.ndarray = None
        self.test_set: np.ndarray = None
        self.num_of_features = None
        self.classifier = PersonalizedDecisionTree(filename, normalise)
        self.predictions = None
        self.early_pruning = early_pruning
        self.early_pruning_param = early_pruning_param
        self.hyperparam = hyperparam

    def fit_predict(self, train_set: np.ndarray, test_set: np.ndarray) -> np.ndarray:
        """
        this function will train the model on the train_set and the return a classification for each item in the test_set
        :param train_set:
            train set given as a numpy.ndarray of dimension (#number of samples, number of features + 1)
            where the first column is the label (1/0).
        :param test_set:
            test set given as a numpy.ndarray of dimension (#number of samples, number of features).
            unlike train, the label isn't given in the first column
        :returns: a numpy.ndarray of dimension (#number of samples) with classification of the test set
        """
        self.train(train_set)
        return self.predict(test_set)

    def train(self, train_set: np.ndarray):
        """
        trains the model
        :param train_set
            train set given as a numpy.ndarray of dimension (#number of samples, number of features + 1)
            where the first column is the label (1/0).
        :returns: classifier of type PersonalizedDecisionTreeClassifier
        """
        self.classifier.train(train_set)
        if self.early_pruning:
            self.earlyPruning(self.early_pruning_param)
        self.classifier.buildGraph(self.classifier.root_node, None)
        self.classifier.showTree()

    def predict(self, test_set: np.ndarray):
        """
        trains the model
        :param test_set:
            test set given as a numpy.ndarray of dimension (#number of samples, number of features).
            unlike train, the label isn't given in the first column
        :returns: a numpy.ndarray of dimension (#number of samples) with classification of the test set
        """
        self.predictions = self.classifier.predict(test_set)
        return self.predictions

    def eval_accuracy(self, correct_test_set_labels: np.ndarray):
        match = 0
        total = 0
        for i in range(len(self.predictions)):
            if self.predictions[i] == correct_test_set_labels[i]:
                match += 1
            total += 1
        return match / total

    def loss(self, correct_test_set_labels: np.ndarray):
        false_positive, false_negative = 0, 0
        for i in range(len(self.predictions)):
            if self.predictions[i] != correct_test_set_labels[i]:
                if self.predictions[i] == 1:
                    false_positive += 1
                else:
                    false_negative += 1

        return false_positive + 8 * false_negative

    def reset(self):
        self.classifier.reset()

    def earlyPruning(self, sample_limit=12):
        self.__pre_order(self.classifier.root_node, sample_limit)

    def __pre_order(self, node: DTCNode, sample_limit=5):
        if node is None:
            return
        if node.sample_num < sample_limit:
            node.left = None
            node.right = None
            node.is_leaf = True
            node.label = 0 if node.false_sample_num > node.true_sample_num * self.hyperparam else 1

        if node.left is not None:
            self.__pre_order(node.left, sample_limit)
        if node.right is not None:
            self.__pre_order(node.right, sample_limit)


def hyperParamTuning(data_set: np.ndarray):
    # to run experiment, import ID3, and run the command ID3.experiment(datas_set)
    k = 5
    hyperparam_options = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,3,4,5,6,7,8,9,10]
    hyperparams = []
    hyperparams_loss = []
    for param in hyperparam_options:
        file_name = f"experiment_ID3_with_pruning_param_{param}"
        kf = KFoldCrossValidation(k, PersonalizedID3, data_set, shuffle=True, filename=file_name, hyperparam=1)
        acc = kf.getError()
        hyperparams.append(param)
        hyperparams_loss.append(acc)
        print(f"accuracy with early pruning param {param}: {acc}")
        plt.plot(hyperparams, hyperparams_loss, "-o")
        plt.xlabel('Pruning Parameter')
        plt.ylabel('Accuracy')
        plt.savefig(f"hyperparam_tuning_with_{param}.png")
        plt.show()

