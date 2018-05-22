
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import random_projection
import pefile, zipfile, json
import sys
import cv2 as cv

from os import listdir
from os.path import isfile, join

def resize_image(pixels, w, h, w_2, h_2):
    im = np.reshape(np.array(pixels), (h, w)).astype(np.uint8)
    thumbnail = cv.resize(im, (w_2, h_2), interpolation=cv.INTER_AREA)
    return list(np.reshape(thumbnail, (w_2 * h_2,)).astype(float))
