import numpy as np

import pefile, zipfile, json, sys, re
import cv2 as cv
import math
from dataset import *
from ensembling import model_histogram, model_api, model_all
import tensorflow as tf


tf.app.flags.DEFINE_string('checkpoint_all', '', "check point directory")
tf.app.flags.DEFINE_string('checkpoint_api', '', "check point directory")
tf.app.flags.DEFINE_string('checkpoint_his', '', "check point directory")

FLAGS = tf.app.flags.FLAGS

def chooseWith(sz):
    w = 0

    if sz > 1000:
        w = 1024
    elif sz > 500:
        w = 768
    elif sz > 200:
        w = 512
    elif sz > 100:
        w = 384
    elif sz > 60:
        w = 256
    elif sz > 30:
        w = 128
    elif sz > 10:
        w = 64
    else:
        w = 32

    return w


def extract_image(file):
    with open(file, 'rb') as ff:
        content = ff.read()
        pixels = []
        h, w = 0, 0
        for byte in content:
            pixels.append(byte)
            sz = len(pixels)
            w = chooseWith(sz / 1024)
            h = int(sz / w)
    return pixels[: w * h], w, h


def extract_api(file):
    imps = []
    try:
        pe = pefile.PE(file)
        pe.parse_data_directories()
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                if imp.name == None:
                    continue
                imps.append(str(imp.name, "latin"))
    except Exception:
        print("not PE file")
    return imps


def resize_image(pixels, w, h, w_2, h_2):
    im = np.reshape(np.array(pixels), (h, w)).astype(np.uint8)
    thumbnail = cv.resize(im, (w_2, h_2), interpolation=cv.INTER_AREA)
    return list(np.reshape(thumbnail, (w_2 * h_2,)).astype(float))


def build_api_feature(apis, dic):
    feature = [0] * (len(dic))
    for api in apis:
        if api in dic:
            id = dic[api]
            feature[id] = 1
    return feature


def entropy(bytes):
    p = [0.0] * 256
    for byte in bytes:
        p[byte] += 1.0
    for i in range(len(p)):
        p[i] = p[i] / len(bytes)
    entropy = 0
    for i in range(len(p)):
        if p[i] != 0.0:
            entropy += p[i] * math.log2(p[i])
    return -entropy


def histogram_2d(pixels, window):
    histogram = [[0 for x in range(16)] for y in range(16)]
    start = 0
    while (start < len(pixels)):
        end = min(start + window, len(pixels))
        bytes = pixels[start: end]
        ent = entropy(bytes)
        ent = math.trunc(ent * 2)
        if ent == 16:
            ent = 15
        for byte in bytes:
            x = math.trunc(byte / 16)
            # print(x)
            # print(ent)
            histogram[x][ent] += 1
        start += window
    ans = []
    for i in range(16):
        for j in range(16):
            ans.append(histogram[i][j])
    return ans


def normalize(arr):
    s = sum(arr)
    out = [float(x) / s for x in arr]
    return np.array(out)

def one_hot(Y, C):
    return np.eye(C)[Y.reshape(-1)]

file =sys.argv[1]
is_malware = int(sys.argv[2])

api_file = open(sys.argv[3], 'r')
lines = api_file.read().split(',')
dic_api = {}

for i in range(len(lines)):
    dic_api[lines[i]] = i
if is_malware:
    labels = np.array([0])
else:
    labels = np.array([1])
labels = one_hot(labels, 2)


pixels, w, h = extract_image(file)

apis = extract_api(file)

fea_img = np.reshape(resize_image(pixels, w, h, 64, 64), (-1, 64, 64, 1))

fea_his = np.reshape(normalize(histogram_2d(pixels, 1024)), (-1, 256))

fea_api = np.reshape(build_api_feature(apis, dic_api), (-1, 794))

# print(fea_api)
class Datasets:
    pass

data_img = Datasets()
data_api = Datasets()
data_his = Datasets()
data_img.test = Dataset(fea_img, labels)
data_his.test = Dataset(fea_his, labels)
data_api.test = Dataset(fea_api, labels)

ensemble_his, labels = model_histogram(input_size=256, n_models=6, data=data_his, checkpoint_dir=FLAGS.checkpoint_his)

ensemble_api, labels_ = model_api(input_size=794, n_models=6, data=data_api, checkpoint_dir=FLAGS.checkpoint_api)

ensemble_all, labels_1 = model_all(data_img, data_his, data_api, checkpoint_dir=FLAGS.checkpoint_all, n_models=6)

print(ensemble_his)
print(ensemble_api)
print(ensemble_all)


