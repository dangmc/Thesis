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
def extract_dll(src, dst):
        cnt = 0
        for f in listdir(src):
                if isfile(dst + f + '.json'):
                    print("file exist")
                    continue
                file = join(src, f)
                dll = extract_import_lib(src + f)
                if len(dll) == 0:
                    continue
                cnt += 1
                with open(dst + f + '.json', 'w') as js:
                    data = {'name': f, 'feature': dll}
                    json.dump(data, js)
                print("file's id : %d, name= %s" % (cnt, f))

def extract_import_lib(file):
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

src = sys.argv[1]
dst = sys.argv[2]
extract_dll(src, dst)
