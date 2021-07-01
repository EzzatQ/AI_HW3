import numpy as np
from matplotlib import pyplot as plt

from KFoldCrossValidation import KFoldCrossValidation
from DecisionTreeClassifier import DecisionTree, DTCNode


class ID3:

    def __init__(self, filename='ID3', early_pruning=False, early_pruning_param=5, normalise=True, hyperparam=1):
        self.train_set: np.ndarray = None
        self.test_set: np.ndarray = None
        self.num_of_features = None
        self.classifier = DecisionTree(filename, normalise)
        self.predictions = None
        self.early_pruning = early_pruning
        self.early_pruning_param = early_pruning_param

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
        self.train(self.__processInput(train_set))
        return self.predict(self.__processInput(test_set))

    def train(self, train_set: np.ndarray):
        """
        trains the model
        :param train_set
            train set given as a numpy.ndarray of dimension (#number of samples, number of features + 1)
            where the first column is the label (1/0).
        :returns: classifier of type DecisionTreeClassifier
        """
        self.classifier.train(train_set)
        if self.early_pruning:
            self.earlyPruning(self.early_pruning_param)
        # self.classifier.buildGraph(self.classifier.root_node, None)
        # self.classifier.showTree()

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
        correct_test_set_labels = np.array([0 if a == "B" else 1 for a in correct_test_set_labels])
        match = 0
        total = 0
        for i in range(len(self.predictions)):
            if self.predictions[i] == correct_test_set_labels[i]:
                match += 1
            total += 1
        return match / total

    def loss(self, correct_test_set_labels: np.ndarray):
        correct_test_set_labels = np.array([0 if a == "B" else 1 for a in correct_test_set_labels])
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


    def __pre_order(self, node: DTCNode, sample_limit=12):
        if node is None:
            return
        if node.sample_num < sample_limit:
            node.left = None
            node.right = None
            node.is_leaf = True
            node.label = 1 if node.true_sample_num > node.false_sample_num else 0

        if node.left is not None:
            self.__pre_order(node.left, sample_limit)
        if node.right is not None:
            self.__pre_order(node.right, sample_limit)

    def __processInput(self, data: np.ndarray):
        for line in data:
            if line[0] == 'M':
                line[0] = 1
            else:
                line[0] = 0
        return np.array(data, float)


def experiment(data_set: np.ndarray):
    # to run experiment, import ID3, and run the command ID3.experiment(datas_set)
    k = 5
    pruning_param_options = list(range(31))
    pruning_params = []
    pruning_acc = []
    for param in pruning_param_options:
        file_name = f"experiment_ID3_with_pruning_param_{param}"
        kf = KFoldCrossValidation(k, ID3, data_set, shuffle=True, early_pruning=True,
                                  early_pruning_param=param, filename=file_name)
        acc = kf.getError()
        pruning_params.append(param)
        pruning_acc.append(acc)
        # print(f"accuracy with early pruning param {param}: {acc}")
        # plt.plot(pruning_params, pruning_acc, "-o")
        # plt.xlabel('Pruning Parameter')
        # plt.ylabel('Accuracy')
        # plt.savefig(f"pruning_experiment_ID3_plot_{param}.png")
        # plt.show()




