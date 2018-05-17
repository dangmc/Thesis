import  numpy as np
import json, sys, csv
from os.path import  join, isfile
from os import listdir


def read_data(path):
    data_ = []
    width = []
    height = []
    for f in listdir(path):
        file = join(path, f)
        if isfile(file):
            with open(file, 'r') as js:
                data = [json.loads(line) for line in js]
                data_.append(data[0]['pixels'])
                width.append(data[0]['width'])
                height.append(data[0]['height'])
                js.close()
    data_ = np.array(data_)
    width = np.array(width)
    height = np.array(height)
    return (data_, width, height)

path1 = sys.argv[1]
path2 = sys.argv[-1]

data, width, height = read_data(path1)

cnt_duplicate = 0
for f in listdir(path2):
    file = join(path2, f)
    if isfile(file):
        with open(file, 'r') as js:
            data_ = [json.loads(line) for line in js]
            p = data_[0]['pixels']
            w = data_[0]['width']
            h = data_[0]['height']
            for (p_, w_, h_) in zip(data, width, height):
                if (w_ == w) and (h_ == h):
                    dup = True
                    for i in range(len(p)):
                        if (p[i] != p_[i]):
                            dup = False
                            break
                    if dup:
                        print(file + " has exist")
                        cnt_duplicate += 1
                        break
            js.close()
print (cnt_duplicate)
