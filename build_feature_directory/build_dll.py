from random import Random

import numpy as np
import matplotlib.pyplot as plt
from sklearn import random_projection
import pefile, zipfile, json, sys, re
import cv2 as cv
import matplotlib.cm as cm

from sultan.api import Sultan
from os import listdir
from os.path import isfile, join
import math

def dll_feature(apis, dic):
        feature = [0]*(len(dic))
        for api in apis:
            if api in dic:
                id = dic[api]
                feature [id] = 1
        return feature

def build_dll_feature(src, dst, dic):
        cnt = 0
        for f in listdir(src):
            if isfile(dst + f):
                print(f + "exist")
                continue
            file = join(src, f)
            fea = dll_feature(file, dic)
            if len(fea)==0:
                continue
            with open(dst + f, 'w') as js:
                data = {'name': f, 'feature': fea}
                json.dump(data, js)
                js.close()
                cnt += 1
                print("file's id: %d, name = %s" % (cnt, f))

src = sys.argv[1]
dst = sys.argv[2]
text_file = open("/home/cuongdm30/Data/api.txt", 'r')
lines = text_file.read().split(',')
api = {}
for i in range(len(lines)):
    api[lines[i]] = i
build_dll_feature(src, dst, api)
