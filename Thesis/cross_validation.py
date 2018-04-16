import numpy as np
from dataset import Dataset


class Datasets:
    pass

class CrossValidationFolds(object):
    def __init__(self, data, labels, num_folds, shuffle=True):
        self.data = data
        self.labels = labels
        self.num_folds = num_folds
        self.current_fold = 0

        # Shuffle Dataset
        if shuffle:
            perm = np.random.permutation(self.data.shape[0])
            data = data[perm]
            labels = labels[perm]

    def split(self):
        current = self.current_fold
        size = int(self.data.shape[0] / self.num_folds)

        index = np.arange(self.data.shape[0])
        lower_bound = index >= current * size
        upper_bound = index < (current + 1) * size
        cv_region = lower_bound * upper_bound

        cv_data = self.data[cv_region]
        train_data = self.data[~cv_region]

        cv_labels = self.labels[cv_region]
        train_labels = self.labels[~cv_region]

        self.current_fold += 1

        datasets = Datasets()
        datasets.train = Dataset(train_data, train_labels)
        datasets.validation = Dataset(cv_data, cv_labels)

        return datasets