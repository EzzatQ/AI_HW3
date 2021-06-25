import numpy as np

class KFoldCrossValidation:

    def __init__(self, k, classifier, data_set: np.ndarray, shuffle=False):
        self.k = k
        self.classifier = classifier
        self.data_set = data_set
        if shuffle:
            np.random.shuffle(self.data_set)
        self.batches = np.array_split(self.data_set, k, axis=0)


    def getError(self):
        total_accuracy = 0
        for i in range(self.k):
            clf = self.classifier(filename=f"{i}th-fold_classificationTree.dot")
            print(f'evaluating fold {i + 1}')
            test_set = self.batches[i][:, 1: self.batches[i].shape[1]]
            test_real_labels = (self.batches[i][:, :1]).flatten()
            train_set = np.concatenate([batch for j, batch in enumerate(self.batches) if i != j])
            clf.fit_predict(train_set, test_set)
            acc = clf.eval_accuracy(test_real_labels)
            print(f"accuracy: {acc}")
            total_accuracy += acc
            self.classifier.reset()
        return total_accuracy/self.k



