import  numpy as np
import json, sys, csv
from os.path import  join, isfile
from os import listdir
from sklearn.model_selection import train_test_split

train_dir = '1gram_train.json'
valid_dir = ''
test_dir = '1gram_test.json'

class   Dataset:

    def __init__(self, _gram, _labels):
        self.gram = _gram
        self.labels = _labels
        self.index_in_epochs = 0
        self.epochs_completed = 0
        self.num_instances = _gram.shape[0]
        self.new_epoch = True

    def next_batch(self, batch_sz):
        start = self.index_in_epochs
        if self.index_in_epochs != 0:
            self.new_epoch = False
        self.index_in_epochs += batch_sz
        if self.index_in_epochs > self.num_instances:
            self.new_epoch = True
            self.epochs_completed += 1
            perm = np.arange(self.num_instances)
            np.random.shuffle(perm)
            self.gram = self.gram[perm]
            self.labels = self.labels[perm]
            start = 0
            self.index_in_epochs = batch_sz
        end = self.index_in_epochs

        return self.gram[start: end], self.labels[start: end]

    def is_new_epoch(self):
        return self.new_epoch

    def get_gram(self):
        return self.gram

    def get_labels(self):
        return self.labels

    def get_epochs_completed(self):
        return self.epochs_completed

def extract_binary_gram(path):
    # TODO
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
        gram = np.array([ins['features'] for ins in data])
        labels = np.array([int (ins['id']) for ins in data])
        f.close()
    return gram, labels


# one hot encoding
def one_hot(Y, C):
    return np.eye(C)[Y.reshape(-1)]


def mnist(path, one_hot_encode=False, num_labels = 10):
    class Datasets:
        pass
    datasets = Datasets()
    mnist = open(path, 'r')
    data = csv.reader(mnist, delimiter=',')
    train = np.reshape([[float(row[x]) / 255 for x in range(1, len(row))] for row in data], (-1, 28, 28, 1))

    mnist = open(path, 'r')
    data = csv.reader(mnist, delimiter=',')
    labels = np.array([int(row[0]) for row in data])
    if one_hot_encode:
        labels = one_hot(labels, num_labels)
    X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2)

    datasets.train = Dataset(X_train, y_train)
    datasets.test = Dataset(X_test, y_test)

    return datasets


def read_data(path, is_malware = False):
    data_ = []
    for f in listdir(path):
        file = join(path, f)
        if isfile(file):
            with open(file, 'r') as js:
                data = [json.loads(line) for line in js]
                data_.append(data[0]['pixels'])
                js.close()
    data_ = np.array(data_)
    labels_ = np.array([0]*data_.shape[0]) if is_malware else np.array([1]*data_.shape[0])
    return (data_, labels_)

def load_data(path_malware, path_benign, one_hot_encode=False, sz = 64, num_labels=2):
    class Datasets:
        pass
    datasets = Datasets()

    data_malware, labels_malware = read_data(path_malware, is_malware=True)
    data_benign, labels_benign = read_data(path_benign)

    data = np.concatenate((data_malware, data_benign), axis=0)
    labels = np.concatenate((labels_malware, labels_benign), axis=0)
    if one_hot_encode:
        labels = one_hot(labels, num_labels)
    data = np.reshape(data, (-1, sz, sz, 1))

    shuffle = np.random.permutation(data.shape[0])
    data = data[shuffle]
    labels = labels[shuffle]

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)

    datasets.train = Dataset(X_train, y_train)
    datasets.test = Dataset(X_test, y_test)

    return datasets


def read_malware_dataset(path, one_hot_encode=False, num_labels=10):

    class Datasets:
        pass
    datasets = Datasets()

    gram, labels = extract_binary_gram(join(path,train_dir))
    if one_hot_encode:
        labels = one_hot(labels, num_labels)
    datasets.train = Dataset(gram, labels)

    # gram, labels = extract_binary_gram(path + valid_dir)
    # datasets.valid = Dataset(gram, labels)

    gram, labels = extract_binary_gram(join(path, test_dir))
    if one_hot_encode:
        labels = one_hot(labels, num_labels)
    datasets.test = Dataset(gram, labels)

    return datasets

# work_dir = sys.argv[1]
# datasets = read_malware_dataset(work_dir, True, 10)
#
# print(datasets.train.get_gram().shape)
#
# labels = datasets.train.get_labels()
#
# print(labels)