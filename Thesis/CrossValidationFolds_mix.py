import numpy as np
from dataset import Dataset_2d


class Datasets:
    pass

class CrossValidationFolds(object):
    def __init__(self, data_img, data_his, labels, num_folds, shuffle=True):
        self.data_img = data_img
        self.data_his = data_his
        self.labels = labels
        self.num_folds = num_folds
        self.current_fold = 0

        # Shuffle Dataset
        if shuffle:
            perm = np.random.permutation(self.data_img.shape[0])
            self.data_img = self.data_img[perm]
            self.data_his = self.data_his[perm]
            self.labels = self.labels[perm]

    def split(self):
        current = self.current_fold
        size = int(self.data_img.shape[0] / self.num_folds)

        index = np.arange(self.data_img.shape[0])
        lower_bound = index >= current * size
        upper_bound = index < (current + 1) * size
        cv_region = lower_bound * upper_bound

        cv_data_img = self.data_img[cv_region]
        cv_data_his = self.data_his[cv_region]

        train_data_img = self.data_img[~cv_region]
        train_data_his = self.data_his[~cv_region]

        cv_labels = self.labels[cv_region]
        train_labels = self.labels[~cv_region]

        self.current_fold += 1

        datasets = Datasets()
        datasets.train = Dataset_2d(train_data_img, train_data_his, train_labels)
        datasets.validation = Dataset_2d(cv_data_img, cv_data_his, cv_labels)

        return datasets