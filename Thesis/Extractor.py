from random import Random

import numpy as np
import matplotlib.pyplot as plt
from sklearn import random_projection

class Extractor:
    def chooseWith(self, sz):
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

    def readBinary(self, file):
        pixels = []
        h, w = 0, 0

        with open(file, 'r') as f:
            for line in f:
                sz = len(line)
                content = line[:sz - 1].split(' ')[1:]
                for byte in content:
                    if byte[0] == '?':
                        pixels.append(0)
                    else:
                        pixels.append(int('0x' + byte, 16))
        sz = len(pixels)
        w = self.chooseWith(sz / 1024)
        h = int(sz / w)
        return pixels[:h * w], h, w

    def readPE(self, file):
        pixels = []
        h, w = 0, 0
        with open(file, 'rb') as f:
            content = f.read()
            for x in content:
                pixels.append(x)
        sz = len(pixels)
        w = self.chooseWith(sz / 1024)
        h = int(sz / w)
        return pixels[: h * w], h, w

    def resizeBilinearGray(self, file, w_2, h_2, isBinary=True):
        if isBinary:
            pixels, h, w = self.readBinary(file)
        else:
            pixels, h, w = self.readPE(file)
        temp = []
        x_ratio = (int)(((w - 1) << 16) / w_2)
        y_ratio = (int)(((h - 1) << 16) / h_2)
        y = 0
        for i in range(h_2):
            yr = (int)(y >> 16)
            y_diff = y - (yr << 16)
            one_min_y_diff = 65536 - y_diff
            y_index = yr * w
            x = 0
            for j in range(w_2):
                xr = (int)(x >> 16)
                x_diff = x - (xr << 16)
                one_min_x_diff = 65536 - x_diff
                index = y_index + xr

                A = pixels[index] & 0xff
                B = pixels[index + 1] & 0xff
                C = pixels[index + w] & 0xff
                D = pixels[index + w + 1] & 0xff

                # Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + D(w)(h)
                gray = (int)((
                                 A * one_min_x_diff * one_min_y_diff + B * x_diff * one_min_y_diff + C * y_diff * one_min_x_diff + D * x_diff * y_diff) >> 32)

                temp.append(gray)
                x += x_ratio
            y += y_ratio;
        return temp

    def randomProjection(self, pixels, h, w, d):
        transfer = random_projection.SparseRandomProjection(d)
        X = np.reshape(pixels, (h, w))
        Y = transfer.fit_transform(X)
        return Y, h, d



    def display(self, pixels, w, h):
        g = np.reshape(pixels, (h, w))
        plt.imshow(g)
        plt.show()



ex = Extractor()
pixels, h, w = ex.readBinary("D:\\Data\\Malware\\dataSample\\0A32eTdBKayjCWhZqDOQ.bytes")
ex.display(pixels, w, h)
p, h, w = ex.randomProjection(pixels, h, w, 512)

ex.display(p, w, h)
# temp = ex.resizeBilinearGray("D:\\Data\\Malware\\dataSample\\0A32eTdBKayjCWhZqDOQ.bytes", 1024, 1024, True)
# ex.display(temp, 1024, 1024)
