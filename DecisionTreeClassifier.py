import math
import numpy as np
from graphviz import Digraph

class DTCNode:

    def __init__(self, feature_index=None, feature_split_value=None, is_leaf=False, label=None):
        self.is_leaf = is_leaf
        self.label = label
        self.left = None
        self.right = None
        self.feature_index = feature_index
        self.feature_split_value = feature_split_value

    def toString(self):
        str = None
        label = "Healthy" if self.label == 0 else "Sick"
        if self.is_leaf:
            str = f"class = {label}"
        else:
            str = f"feature[{self.feature_index}] < {self.feature_split_value}"
        return str


class DecisionTree:

    def __init__(self, file_name='ID2.dot'):
        self.root_node = None
        self.graph = Digraph(format="png", filename=file_name)
        self.max_node_name = 0
        self.filename=file_name

    def train(self, train_set: np.ndarray):
        if self.root_node is not None:
            return
        self.root_node = self.__developNode(train_set)
        self.buildGraph(self.root_node)
        self.showTree()

    def reset(self):
        self.root_node = None
        self.c

    def predict(self, test_set: np.ndarray):
        labels = np.empty(test_set.shape[0])
        for i, x in enumerate(test_set):
            labels[i] = self.__predictOne(x)
        return labels

    def __predictOne(self, x: np.ndarray):
        assert(len(x.shape) == 1)
        return self.__search(x, self.root_node)

    def __search(self, x: np.ndarray, node: DTCNode):
        if node.is_leaf:
            return node.label
        if x[node.feature_index] < node.feature_split_value:
            return self.__search(x, node.left)
        else:
            return self.__search(x, node.right)

    def __developNode(self, train_set: np.ndarray):
        """
        recursive function that builds the Decision tree.
        :param train_set: set of examples for the current node
        :return: root node of the tree
        """
        if train_set.shape[0] == 0:
            return None
        label = self.__allLabelsEqual(train_set)
        if label != -1:
            node = DTCNode(is_leaf=True, label=label)
            return node
        index, split_value = self.__chooseNextFeature(train_set)
        node = DTCNode(feature_index=index-1, feature_split_value=split_value)
        left_train_set, right_train_set = self.__split_set(train_set, index, split_value)
        node.left = self.__developNode(left_train_set)
        node.right = self.__developNode(right_train_set)
        return node

    def __chooseNextFeature(self, train_set: np.ndarray):
        """
        will choose the best feature to split a node by InformationGain
        :returns: a tuple (index, split_value) where index is the index of the feature to split by and split_value,
        is the value with which to split the feature domain.
        """
        best_index = -1
        max_ig = -1
        best_split_value = None
        for i in range(1, train_set.shape[1]):
            val, ig = self.__bestFeatureSplitValue(train_set, i)
            if ig > max_ig:
                max_ig, best_index, best_split_value = ig, i, val
        return best_index, best_split_value

    def __bestFeatureSplitValue(self, train_set: np.ndarray, feature_index):
        """
        will choose the best value to split the the feature domain for discretisation
        :param train_set: training set
        :param feature_index: index of column in train_set to which you want to find the best aplit value
        :return: split value
        """
        feature_vals = train_set[:, feature_index]
        split_vals = self.__possibleSplitValues(feature_vals)
        best_val, max_ig = -1,  -1
        for splitter in split_vals:
            left_set, right_set = self.__split_set(train_set, feature_index, splitter)
            ig = self.__informationGain(train_set, left_set, right_set)
            if ig > max_ig:
                best_val, max_ig = splitter, ig
        return best_val, max_ig

    def __informationGain(self, parent_set: np.ndarray, left_set: np.ndarray, right_set: np.ndarray):
        parent_entropy = self.__entropy(parent_set)
        weighted_left_entropy = 0 if left_set.shape[0] == 0 else (left_set.shape[0]/parent_set.shape[0]) * self.__entropy(left_set)
        weighted_right_entropy = 0 if right_set.shape[0] == 0 else (right_set.shape[0]/parent_set.shape[0]) * self.__entropy(right_set)
        return parent_entropy - weighted_right_entropy - weighted_left_entropy

    def __entropy(self, train_set):
        true_labels, false_labels = self.__countLabels(train_set)
        total_labels = true_labels + false_labels
        prob_true = true_labels / total_labels
        prob_false = false_labels / total_labels
        return -((0 if prob_true == 0 else prob_true * math.log2(prob_true)) +
                 (0 if prob_false == 0 else prob_false * math.log2(prob_false)))

    def __countLabels(self, train_set: np.ndarray):
        num_true, num_false = 0, 0
        for row in train_set:
            if row[0] == 0:
                num_false += 1
            else:
                num_true += 1
        return num_true, num_false

    def __possibleSplitValues(self, feature_values: np.ndarray):
        feature_values.sort()
        possible_vals = []
        for i in range(feature_values.shape[0] - 1):
            diff = (feature_values[i + 1] + feature_values[i]) / 2
            possible_vals.append(diff)
        return possible_vals

    def __split_set(self, train_set: np.ndarray, feature_index, split_value):
        left_train_set, right_train_set = np.empty((0, train_set.shape[1])), np.empty((0, train_set.shape[1]))
        for i in range(train_set.shape[0]):
            if train_set[i, feature_index] < split_value:
                left_train_set = np.vstack([left_train_set, train_set[i]])
                # np.append(left_train_set, np.expand_dims(row, axis=0), axis=0)
            else:
                right_train_set = np.vstack([right_train_set, train_set[i]])
                # np.append(right_train_set, np.expand_dims(row, axis=0), axis=0)
        return left_train_set, right_train_set

    def __allLabelsEqual(self, train_set: np.ndarray):
        val = train_set[0][0]
        for row in train_set:
            if row[0] != val:
                return -1
        return val

    def showTree(self):
        if self.root_node is None:
            return
        # self.graph.render(filename=self.filename, view=False)

    def buildGraph(self, node: DTCNode, parentName=None):
        self.graph.node(f"{self.max_node_name}", label=node.toString())
        if parentName is not None:
            self.graph.edge(f"{self.max_node_name}", parentName)
        if node.left is not None:
            self.buildGraph(node.left, f"{self.max_node_name}")
        if node.right is not None:
            self.buildGraph(node.right, f"{self.max_node_name}")
        self.max_node_name += 1

