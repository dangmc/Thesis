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

class Histogram:

    def entropy(self, bytes):
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

    def histogram_2d(self, pixels, window):
        histogram = [[0 for x in range(16)] for y in range(16)]
        start = 0
        while (start < len(pixels)):
            end = min(start + window, len(pixels))
            bytes = pixels[start: end]
            ent = self.entropy(bytes)
            ent = math.trunc(ent * 2)
            if ent == 16:
                ent = 15
            for byte in bytes:
                x = math.trunc(byte / 16)
                #print(x)
                #print(ent)
                histogram[x][ent] += 1
            start += window
        ans = []
        for i in range(16):
            for j in range(16):
                ans.append(histogram[i][j])
        return ans
    
    def normalize(self, arr):
        s = sum(arr)
        out = [float(x)/s for x in arr]
        return np.array(out)      
    
    def extract_2d_histogram(self, src, dst, window):
        cnt = 0
        for f in listdir(src):
            file = join(src, f)
            if (isfile(dst+f)):
                print(f + "exist")
                continue
            try:
                with open(file, 'r') as js:
                    data = [json.loads(line) for line in js]
                    pixels = data[0]['pixels']
                    name = data[0]['name']
                    #ent = self.entropy(pixels)
                    histogram = self.histogram_2d(pixels, window)
                    s = sum(histogram)
                    # histogram = [x / s for x in histogram]
                    with open(dst + f, 'w') as j:
                        data = {'name': name, 'feature': histogram}
                        json.dump(data, j)
                        j.close()
                        cnt += 1
                        print("file's id: %d, name = %s" % (cnt, name))
                    js.close()
            except Exception:
                print('ERROR: ' + file)

his = Histogram()
src = sys.argv[1]
dst = sys.argv[2]
his.extract_2d_histogram(src, dst, 1024)

#file = "/home/cuongdm30/Data/PE/image_PE_train/8cb67aa9d67d31278db65eeedd5f9f5e02c49efc.json"
#with open(file, 'r') as js:
#   data = [json.loads(line) for line in js]
#   pixel = data[0]['pixels']
#   name = data[0]['name']
#   histogram = his.histogram_2d(pixel,1024)
#   print(histogram)
#   print(name)
#   with open( "/home/cuongdm30/Data/fuck.json", 'w') as j:
#                        data = {'name': name, 'feature': histogram}
#                        json.dump(data, j)
#                        j.close()
   #print(pixel)
#   js.close()

