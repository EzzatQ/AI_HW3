import math

import numpy as np
from graphviz import Digraph

colors = ["Orchid",
          "PaleGoldenRod",
          "PaleGreen",
          "PaleTurquoise",
          "PaleVioletRed",
          "PapayaWhip",
          "PeachPuff",
          "Peru",
          "Pink",
          "Plum",
          "PowderBlue",
          "Purple",
          "RebeccaPurple",
          "Red",
          "RosyBrown",
          "RoyalBlue",
          "SaddleBrown",
          "Salmon",
          "SandyBrown",
          "SeaGreen",
          "SeaShell",
          "Sienna",
          "Silver",
          "SkyBlue",
          "SlateBlue",
          "SlateGray",
          "SlateGrey",
          "Snow",
          "SpringGreen",
          "SteelBlue",
          "Tan",
          "Teal",
          "Thistle",
          "Tomato",
          "Turquoise",
          "Violet",
          "Wheat",
          "White",
          "WhiteSmoke",
          "Yellow",
          "YellowGreen"]


class DTCNode:

    def __init__(self, feature_index=None, feature_split_value=None, is_leaf=False, label=None, name=0,
                 true_sample_num=None, false_sample_num=None, information_gain=None, ):
        self.is_leaf = is_leaf
        self.label = label
        self.left = None
        self.right = None
        self.feature_index = feature_index
        self.feature_split_value = feature_split_value
        self.unique_id = name
        self.sample_num = true_sample_num + false_sample_num
        self.true_sample_num = true_sample_num
        self.false_sample_num = false_sample_num
        self.information_gain = information_gain



    def toString(self):
        label = "Healthy" if self.label == 0 else "Sick"
        if self.is_leaf:
            name = f"class = {label}\n"
        else:
            name = f"feature[{self.feature_index}] < {self.feature_split_value}\n"
            name += f"IG = {self.information_gain}\n"
        name += f"sample # = {self.sample_num}\n"
        name += f"true = {self.true_sample_num}, false = {self.false_sample_num}"
        return name


class DecisionTree:

    def __init__(self, file_name='ID2.dot', normalise=True):
        self.root_node = None
        self.graph = Digraph(format="png", filename=file_name)
        self.max_node_name = 0
        self.filename = file_name
        self.feature_maxes = []
        self.feature_mins = []
        self.normalise = normalise

    def train(self, train_set: np.ndarray):
        if self.root_node is not None:
            return
        if self.normalise:
            train_set = self.__normalise_features(train_set)
        self.root_node = self.__developNode(train_set)

    def reset(self):
        self.root_node = None

    def predict(self, test_set: np.ndarray):
        if self.normalise:
           test_set =  self.__normalise_test_set(test_set)
        labels = np.empty(test_set.shape[0])
        for i, x in enumerate(test_set):
            labels[i] = self.__predictOne(x)
        return labels

    def __predictOne(self, x: np.ndarray):
        assert (len(x.shape) == 1)
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
        true_num, false_num = self.__countLabels(train_set)
        total_num = true_num + false_num
        label = 1 if true_num == total_num else 0 if false_num == total_num else -1
        self.max_node_name += 1
        if label != -1:
            node = DTCNode(is_leaf=True, label=label, name=self.max_node_name,
                           true_sample_num=true_num, false_sample_num=false_num)
            return node
        index, split_value, ig = self.__chooseNextFeature(train_set)
        node = DTCNode(feature_index=index - 1, feature_split_value=split_value, name=self.max_node_name,
                       true_sample_num=true_num, false_sample_num=false_num, information_gain=ig)
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
        return best_index, best_split_value, max_ig

    def __bestFeatureSplitValue(self, train_set: np.ndarray, feature_index):
        """
        will choose the best value to split the the feature domain for discretisation
        :param train_set: training set
        :param feature_index: index of column in train_set to which you want to find the best aplit value
        :return: split value
        """
        feature_vals = train_set[:, feature_index]
        split_vals = self.__possibleSplitValues(feature_vals)
        best_val, max_ig = -1, -1
        for splitter in split_vals:
            left_set, right_set = self.__split_set(train_set, feature_index, splitter)
            ig = self.__informationGain(train_set, left_set, right_set)
            if ig > max_ig:
                best_val, max_ig = splitter, ig
                if ig == 1:
                    break
        return best_val, max_ig

    def __informationGain(self, parent_set: np.ndarray, left_set: np.ndarray, right_set: np.ndarray):
        parent_entropy = self.__entropy(parent_set)
        weighted_left_entropy = 0 if left_set.shape[0] == 0 else (left_set.shape[0] / parent_set.shape[
            0]) * self.__entropy(left_set)
        weighted_right_entropy = 0 if right_set.shape[0] == 0 else (right_set.shape[0] / parent_set.shape[
            0]) * self.__entropy(right_set)
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
        features_copy = np.copy(feature_values)
        features_copy.sort()
        possible_vals = []
        for i in range(features_copy.shape[0] - 1):
            diff = (feature_values[i + 1] + feature_values[i]) / 2
            possible_vals.append(diff)
        return possible_vals

    def __split_set(self, train_set: np.ndarray, feature_index, split_value):
        left_train_set, right_train_set = np.empty((0, train_set.shape[1])), np.empty((0, train_set.shape[1]))
        for i in range(train_set.shape[0]):
            if train_set[i, feature_index] < split_value:
                left_train_set = np.vstack([left_train_set, train_set[i]])
            else:
                right_train_set = np.vstack([right_train_set, train_set[i]])
        return left_train_set, right_train_set

    def __allLabelsEqual(self, train_set: np.ndarray):
        val = train_set[0][0]
        for row in train_set:
            if row[0] != val:
                return -1
        return val

    def __normalise_features(self, train_set: np.ndarray):
        for feature in range(1, train_set.shape[1]):
            max_val = train_set[:, feature].max(axis=0)
            min_val = train_set[:, feature].min(axis=0)
            self.feature_maxes.append(max_val)
            self.feature_mins.append(min_val)
            train_set[:, feature] = (train_set[:, feature] - min_val) / (max_val - min_val)
        return train_set

    def __normalise_test_set(self, test_set: np.ndarray):
        for feature in range(0, test_set.shape[1]):
            max_val = self.feature_maxes[feature]
            min_val = self.feature_mins[feature]
            test_set[:, feature] = (test_set[:, feature] - min_val) / (max_val - min_val)
        return test_set

    def showTree(self):
        if self.root_node is None:
            return
        self.graph.render(filename=self.filename, view=False, cleanup=True)

    def buildGraph(self, node: DTCNode, parent: DTCNode):
        if node.is_leaf:
            self.graph.node(f"{node.unique_id}", label=node.toString(), style='filled',
                            fillcolor='green' if node.label == 0 else "red")
        else:
            self.graph.node(f"{node.unique_id}", label=node.toString(), style='filled',
                            fillcolor=colors[node.feature_index], shape="rectangle")
        if parent is not None:
            self.graph.edge(f"{parent.unique_id}", f"{node.unique_id}")
        if node.left is not None:
            self.buildGraph(node.left, node)
        if node.right is not None:
            self.buildGraph(node.right, node)


