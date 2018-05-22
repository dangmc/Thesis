import numpy as np
import json, sys, csv
from os.path import join, isfile
from os import listdir
from sklearn.model_selection import train_test_split

train_dir = '1gram_train.json'
valid_dir = ''
test_dir = '1gram_test.json'


class Dataset:
    def __init__(self, _gram, _labels):
        self.gram = _gram
        self.labels = _labels
        self.index_in_epochs = 0
        self.epochs_completed = 0
        self.num_instances = _gram.shape[0]
        self.new_epoch = False

    def next_batch(self, batch_sz):
        start = self.index_in_epochs
        if self.index_in_epochs != 0:
            self.new_epoch = False

        self.index_in_epochs += batch_sz

        if self.index_in_epochs > self.num_instances and start < self.num_instances:
            self.index_in_epochs = self.num_instances
            self.epochs_completed += 1
            self.new_epoch = True

        if start == self.num_instances:
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


class Dataset_2d:
    def __init__(self, _img, _his, _labels):
        self.img = _img
        self.his = _his
        self.labels = _labels
        self.index_in_epochs = 0
        self.epochs_completed = 0
        self.num_instances = _img.shape[0]
        self.new_epoch = False

    def next_batch(self, batch_sz):
        start = self.index_in_epochs
        if self.index_in_epochs != 0:
            self.new_epoch = False

        self.index_in_epochs += batch_sz

        if self.index_in_epochs > self.num_instances and start < self.num_instances:
            self.index_in_epochs = self.num_instances
            self.epochs_completed += 1
            self.new_epoch = True

        if start == self.num_instances:
            perm = np.arange(self.num_instances)
            np.random.shuffle(perm)
            self.img = self.img[perm]
            self.his = self.his[perm]
            self.labels = self.labels[perm]
            start = 0
            self.index_in_epochs = batch_sz

        end = self.index_in_epochs

        return self.img[start: end], self.his[start: end], self.labels[start: end]

    def is_new_epoch(self):
        return self.new_epoch

    def get_img(self):
        return self.img

    def get_his(self):
        return self.his

    def get_labels(self):
        return self.labels

    def get_epochs_completed(self):
        return self.epochs_completed


def extract_binary_gram(path):
    # TODO
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
        gram = np.array([ins['features'] for ins in data])
        labels = np.array([int(ins['id']) for ins in data])
        f.close()
    return gram, labels


# one hot encoding
def one_hot(Y, C):
    return np.eye(C)[Y.reshape(-1)]


def mnist(path, one_hot_encode=False, num_labels=10):
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


# image feature
def read_data(path, is_malware=False):
    data_ = []
    cnt = 0
    for f in listdir(path):
        cnt += 1
        file = join(path, f)
        if isfile(file):
            with open(file, 'r') as js:
                data = [json.loads(line) for line in js]
                data_.append(data[0]['pixels'])
                js.close()
    data_ = np.array(data_)
    labels_ = np.array([0] * data_.shape[0]) if is_malware else np.array([1] * data_.shape[0])
    return (data_, labels_)


def load_real_data(path_malware, path_benign, one_hot_encode=False, sz=64, num_labels=2):
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

    datasets.real = Dataset(data, labels)

    return datasets


def load_data(path_malware, path_benign, path_real, one_hot_encode=False, sz=64, num_labels=2):
    class Datasets:
        pass

    datasets = Datasets()

    data_malware, labels_malware = read_data(path_malware, is_malware=True)
    data_benign, labels_benign = read_data(path_benign)
    data_real, labels_real = read_data(path_real, is_malware=True)

    data = np.concatenate((data_malware, data_benign), axis=0)
    labels = np.concatenate((labels_malware, labels_benign), axis=0)
    if one_hot_encode:
        labels = one_hot(labels, num_labels)
        labels_real = one_hot(labels_real, num_labels)
    data = np.reshape(data, (-1, sz, sz, 1))
    data_real = np.reshape(data_real, (-1, sz, sz, 1))

    shuffle = np.random.permutation(data.shape[0])
    data = data[shuffle]
    labels = labels[shuffle]

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)

    datasets.train = Dataset(X_train, y_train)
    datasets.test = Dataset(X_test, y_test)
    datasets.real = Dataset(data_real, labels_real)

    return datasets


# histogram feature
def load_test_data_histogram(path_malware=None, path_benign=None, one_hot_encode=False, sz=256, num_labels=2):
    class Datasets:
        pass

    datasets = Datasets()
    data_malware, labels_malware = read_histogram_data(path_malware, is_malware=True)
    data_benign, labels_benign = read_histogram_data(path_benign)

    data = np.concatenate((data_malware, data_benign), axis=0)
    labels = np.concatenate((labels_malware, labels_benign), axis=0)
    # data = data_malware
    # labels = labels_malware
    if one_hot_encode:
        labels = one_hot(labels, num_labels)
    data = np.reshape(data, (-1, sz))

    datasets.real = Dataset(data, labels)
    return datasets


def read_histogram_data(path, is_malware=False):
    data_ = []
    cnt = 0
    for f in listdir(path):
        cnt += 1
        file = join(path, f)
        if isfile(file):
            with open(file, 'r') as js:
                data = [json.loads(line) for line in js]
                s = sum(data[0]["feature"])
                feature = [float(x) / s for x in data[0]["feature"]]
                data_.append(feature)
                js.close()
    data_ = np.array(data_)
    labels_ = np.array([0] * data_.shape[0]) if is_malware else np.array([1] * data_.shape[0])
    return (data_, labels_)


def load_histogram_data(path_malware, path_benign, one_hot_encode=False, sz=256, num_labels=2, is_train=True):
    class Datasets:
        pass

    datasets = Datasets()

    data_malware, labels_malware = read_histogram_data(path_malware, is_malware=True)
    data_benign, labels_benign = read_histogram_data(path_benign)

    data = np.concatenate((data_malware, data_benign), axis=0)
    labels = np.concatenate((labels_malware, labels_benign), axis=0)
    if one_hot_encode:
        labels = one_hot(labels, num_labels)
    data = np.reshape(data, (-1, 256))
    shuffle = np.random.permutation(data.shape[0])
    data = data[shuffle]
    labels = labels[shuffle]

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)
    if is_train:
        datasets.train = Dataset(X_train, y_train)
        datasets.test = Dataset(X_test, y_test)
    else:
        datasets.test = Dataset(X_test, y_test)
    return datasets


# image + histogram feature
def read_histogram_feature(file):
    with open(file, 'r') as js:
        data = [json.loads(line) for line in js]
        s = sum(data[0]["feature"])
        feature = [float(x) / s for x in data[0]["feature"]]
        js.close()
    return feature


def read_image_feature(file):
    with open(file, 'r') as js:
        data = [json.loads(line) for line in js]
        feature = data[0]["pixels"]
        js.close()
    return feature


def read_histogram_image(path_img, path_his, is_malware=False):
    data_img = []
    data_his = []
    for f in listdir(path_his):
        if isfile(path_img + f) == False:
            continue
        file_his = join(path_his, f)
        file_img = join(path_img, f)
        fea_his = read_histogram_feature(file_his)
        fea_img = read_image_feature(file_img)
        data_his.append(fea_his)
        data_img.append(fea_img)
    data_his = np.array(data_his)
    data_img = np.array(data_img)
    labels_ = np.array([0] * data_his.shape[0]) if is_malware else np.array([1] * data_his.shape[0])
    return (data_his, data_img, labels_)


def load_histogram_image(path_malware_img, path_malware_his, path_benign_img, path_benign_his, one_hot_encode=False,
                         sz_img=64, sz_his=256, num_labels=2, split_rate=85):
    class Datasets:
        pass

    datasets = Datasets()
    data_malware_his, data_malware_img, labels_malware = read_histogram_image(path_img=path_malware_img,
                                                                              path_his=path_malware_his,
                                                                              is_malware=True)
    data_benign_his, data_benign_img, labels_benign = read_histogram_image(path_img=path_benign_img,
                                                                           path_his=path_benign_his)

    data_his = np.concatenate((data_malware_his, data_benign_his), axis=0)
    data_img = np.concatenate((data_malware_img, data_benign_img), axis=0)
    labels = np.concatenate((labels_malware, labels_benign), axis=0)

    if one_hot_encode:
        labels = one_hot(labels, num_labels)
    data_his = np.reshape(data_his, (-1, sz_his))
    data_img = np.reshape(data_img, (-1, sz_img, sz_img, 1))

    total_ins = data_img.shape[0]
    train_ins = int(total_ins * split_rate / 100)

    X_train_his = data_his[:train_ins]
    X_train_img = data_img[:train_ins]

    X_test_his = data_his[train_ins:]
    X_test_img = data_img[train_ins:]

    y_train = labels[:train_ins]
    y_test = labels[train_ins:]

    datasets.train = Dataset_2d(X_train_img, X_train_his, y_train)
    datasets.test = Dataset_2d(X_test_img, X_test_his, y_test)

    return datasets


# dll feature
def read_dll_feature(file):
    with open(file, 'r') as js:
        data = [json.loads(line) for line in js]
        feature = data[0]["feature"]
        js.close()
    return feature


def read_dll_data(path, is_malware=False):
    dll = []
    for f in listdir(path):
        file = join(path, f)
        data = read_dll_feature(file)
        dll.append(data)
    dll = np.array(dll)
    labels_ = np.array([0] * dll.shape[0]) if is_malware else np.array([1] * dll.shape[0])
    return (dll, labels_)


def load_dll_data(path_malware, path_benign, one_hot_encode=False, sz=794, num_labels=2, is_train=True):
    class Datasets:
        pass

    datasets = Datasets()

    data_malware, labels_malware = read_dll_data(path_malware, is_malware=True)
    data_benign, labels_benign = read_dll_data(path_benign)

    data = np.concatenate((data_malware, data_benign), axis=0)
    labels = np.concatenate((labels_malware, labels_benign), axis=0)
    if one_hot_encode:
        labels = one_hot(labels, num_labels)
    data = np.reshape(data, (-1, sz))
    shuffle = np.random.permutation(data.shape[0])
    data = data[shuffle]
    labels = labels[shuffle]
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15)
    if is_train:
        datasets.train = Dataset(X_train, y_train)
        datasets.test = Dataset(X_test, y_test)
    else:
        datasets.test = Dataset(X_test, y_test)
    return datasets


# dll + histogram
def read_histogram_dll(path_dll, path_his, is_malware=False):
    data = []
    for f in listdir(path_dll):
        if isfile(path_his + f) == False:
            continue
        file_his = join(path_his, f)
        file_dll = join(path_dll, f)
        fea_his = np.array(read_histogram_feature(file_his))
        fea_dll = np.array(read_dll_feature(file_dll))
        fea = np.concatenate((fea_his, fea_dll), axis=0)
        data.append(fea)
    data = np.array(data)
    labels_ = np.array([0] * data.shape[0]) if is_malware else np.array([1] * data.shape[0])
    return (data, labels_)


def load_histogram_dll(path_malware_dll, path_malware_his, path_benign_dll, path_benign_his, one_hot_encode=False,
                       sz_dll=794, sz_his=256, num_labels=2, split_rate=85, is_train=True):
    class Datasets:
        pass

    datasets = Datasets()
    data_malware, labels_malware = read_histogram_dll(path_dll=path_malware_dll, path_his=path_malware_his,
                                                      is_malware=True)
    data_benign, labels_benign = read_histogram_dll(path_dll=path_benign_dll, path_his=path_benign_his)

    data = np.concatenate((data_malware, data_benign), axis=0)
    labels = np.concatenate((labels_malware, labels_benign), axis=0)

    if one_hot_encode:
        labels = one_hot(labels, num_labels)
    data = np.reshape(data, (-1, sz_dll + sz_his))

    total_ins = data.shape[0]
    train_ins = int(total_ins * split_rate / 100)

    X_train = data[:train_ins]
    X_test = data[train_ins:]

    y_train = labels[:train_ins]
    y_test = labels[train_ins:]
    if is_train:
        datasets.train = Dataset(X_train, y_train)
        datasets.test = Dataset(X_test, y_test)
    else:
        datasets.test = Dataset(X_test, y_test)

    return datasets


# image + dll + histogram

def read_histogram_dll_(path_dll, path_his, path_img, is_malware=False):
    data = []
    for f in sorted(listdir(path_dll)):
        if isfile(path_his + f) == False or not isfile(path_img + f):
            continue
        file_his = join(path_his, f)
        file_dll = join(path_dll, f)
        fea_his = np.array(read_histogram_feature(file_his))
        fea_dll = np.array(read_dll_feature(file_dll))
        fea = np.concatenate((fea_his, fea_dll), axis=0)
        data.append(fea)
    data = np.array(data)
    labels_ = np.array([0] * data.shape[0]) if is_malware else np.array([1] * data.shape[0])
    return (data, labels_)


def read_histogram_dll_image(path_his, path_dll, path_img, is_malware=False):
    data_his_dll, labels = read_histogram_dll_(path_dll, path_his, path_img, is_malware)
    data_img = []
    for f in sorted(listdir(path_dll)):
        if not isfile(join(path_img, f)) or not isfile(path_his + f):
            continue
        file_img = join(path_img, f)
        fea_img = read_image_feature(file_img)
        data_img.append(fea_img)
    data_img = np.array(data_img)
    return data_his_dll, data_img, labels


# def read_histogram_dll_image(path_dll, path_his, path_img, is_malware=False):
#     data = []
#     data_img = []
#     for f in listdir(path_dll):
#         if isfile(path_his + f) == False or not isfile(path_img + f):
#             continue
#         file_his = join(path_his, f)
#         file_dll = join(path_dll, f)
#         file_img = join(path_img, f)
#         fea_img = np.array(read_image_feature(file_img))
#         fea_his = np.array(read_histogram_feature(file_his))
#         fea_dll = np.array(read_dll_feature(file_dll))
#         fea = np.concatenate((fea_his, fea_dll), axis=0)
#         data.append(fea)
#         data_img.append(fea_img)
#     data = np.array(data)
#     data_img = np.array(data_img)
#     labels_ = np.array([0] * data.shape[0]) if is_malware else np.array([1] * data.shape[0])
#     return (data, data_img, labels_)


def load_histogram_dll_image(path_malware_img, path_malware_his, path_malware_dll, path_benign_img, path_benign_his,
                             path_benign_dll, one_hot_encode=False, sz_img=64, sz_his=256, sz_dll=794, num_labels=2,
                             split_rate=85):
    class Datasets:
        pass

    datasets = Datasets()
    data_malware_his, data_malware_img, labels_malware = read_histogram_dll_image(path_img=path_malware_img,
                                                                                  path_his=path_malware_his,
                                                                                  path_dll=path_malware_dll,
                                                                                  is_malware=True)
    data_benign_his, data_benign_img, labels_benign = read_histogram_dll_image(path_img=path_benign_img,
                                                                               path_dll=path_benign_dll,
                                                                               path_his=path_benign_his)

    data_his = np.concatenate((data_malware_his, data_benign_his), axis=0)
    data_img = np.concatenate((data_malware_img, data_benign_img), axis=0)
    labels = np.concatenate((labels_malware, labels_benign), axis=0)
    shuffle = np.random.permutation(data_his.shape[0])

    data_his = data_his[shuffle]
    data_img = data_img[shuffle]
    labels = labels[shuffle]

    if one_hot_encode:
        labels = one_hot(labels, num_labels)
    data_his = np.reshape(data_his, (-1, sz_his + sz_dll))
    data_img = np.reshape(data_img, (-1, sz_img, sz_img, 1))

    total_ins = data_img.shape[0]
    train_ins = int(total_ins * split_rate / 100)

    X_train_his = data_his[:train_ins]
    X_train_img = data_img[:train_ins]

    X_test_his = data_his[train_ins:]
    X_test_img = data_img[train_ins:]

    y_train = labels[:train_ins]
    y_test = labels[train_ins:]

    datasets.train = Dataset_2d(X_train_img, X_train_his, y_train)
    datasets.test = Dataset_2d(X_test_img, X_test_his, y_test)

    return datasets


def read_all_feature(path_dll, path_his, path_img, sz_api=794, sz_his=256, sz_img=64, is_malware=False,
                     one_hot_encode=True, num_labels=2):
    data_his = []
    data_img = []
    data_api = []
    for f in listdir(path_dll):
        if isfile(path_his + f) == False or not isfile(path_img + f):
            continue
        file_his = join(path_his, f)
        file_dll = join(path_dll, f)
        file_img = join(path_img, f)
        fea_img = read_image_feature(file_img)
        fea_his = np.array(read_histogram_feature(file_his))
        fea_dll = np.array(read_dll_feature(file_dll))
        data_his.append(fea_his)
        data_img.append(fea_img)
        data_api.append(fea_dll)
    data_api = np.reshape(data_api, (-1, sz_api))
    data_his = np.reshape(data_his, (-1, sz_his))
    data_img = np.reshape(data_img, (-1, sz_img, sz_img, 1))
    labels_ = np.array([0] * data_his.shape[0]) if is_malware else np.array([1] * data_his.shape[0])
    if one_hot_encode:
        labels = one_hot(labels_, num_labels)
    return data_img, data_his, data_api, labels


def read_all_feature_malware_benign(path_malware_img, path_malware_his, path_malware_dll, path_benign_img,
                                    path_benign_his,
                                    path_benign_dll, n_ins=3500):
    class Datasets:
        pass

    datasets_api = Datasets()
    datasets_his = Datasets()
    datasets_img = Datasets()

    data_img_malware, data_his_malware, data_api_malware, labels_malware = read_all_feature(path_dll=path_malware_dll,
                                                                                            path_his=path_malware_his,
                                                                                            path_img=path_malware_img,
                                                                                            is_malware=True)
    data_img_benign, data_his_benign, data_api_benign, labels_benign = read_all_feature(path_dll=path_benign_dll,
                                                                                        path_his=path_benign_his,
                                                                                        path_img=path_benign_img,
                                                                                        is_malware=False)
    total_ins_malware = data_api_malware.shape[0]
    total_ins_benign = data_api_benign.shape[0]

    n_ins_mal = min(n_ins, total_ins_malware)
    n_ins_ben = min(n_ins, total_ins_benign)

    data_img_malware, data_his_malware, data_api_malware, labels_malware = data_img_malware[:n_ins_mal], data_his_malware[
                                                                                                     :n_ins_mal], data_api_malware[
                                                                                                              :n_ins_mal], labels_malware[
                                                                                                                :n_ins_mal]
    data_img_benign, data_his_benign, data_api_benign, labels_benign = data_img_benign[:n_ins_ben], data_his_benign[
                                                                                                     :n_ins_ben], data_api_benign[
                                                                                                              :n_ins_ben], labels_benign[
                                                                                                                :n_ins_ben]

    data_api = np.concatenate((data_api_malware, data_api_benign), axis=0)
    data_his = np.concatenate((data_his_malware, data_his_benign), axis=0)
    data_img = np.concatenate((data_img_malware, data_img_benign), axis=0)
    labels = np.concatenate((labels_malware, labels_benign), axis=0)

    shuffle = np.random.permutation(data_api.shape[0])
    data_api = data_api[shuffle]
    data_his = data_his[shuffle]
    data_img = data_img[shuffle]
    labels = labels[shuffle]

    datasets_api.test = Dataset(data_api, labels)
    datasets_his.test = Dataset(data_his, labels)
    datasets_img.test = Dataset(data_img, labels)

    return (datasets_img, datasets_his, datasets_api)
