import numpy as np
import pydot
from DecisionTreeClassifier import DecisionTree


class ID3:

    def __init__(self, filename='ID2.dot'):
        self.train_set = None
        self.test_set = None
        self.num_of_features = None
        self.classifier = DecisionTree(filename)
        self.predictions = None

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
        :returns: classifier of type DecisionTreeClassifier
        """
        self.classifier.train(train_set)

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

    def reset(self):
        self.classifier.reset()





