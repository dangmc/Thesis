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
        return pixels[: w*h], w, h
src = sys.argv[1]
dst = sys.argv[2]
extract_image_file_PE(src, dst)
